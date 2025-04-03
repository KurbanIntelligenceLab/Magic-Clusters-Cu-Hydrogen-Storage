import torch
import torch.nn as nn
from transformers import AutoImageProcessor, ResNetForImageClassification
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_chans, embed_dim)
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GraphFeatureProcessor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 1024, 16], dropout=0.1):
        super().__init__()
        layers = []
        
        # First layer: input_dim -> hidden_dims[0]
        layers.extend([
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout)
        ])
        
        # Intermediate layers: hidden_dims[i] -> hidden_dims[i+1]
        for i in range(len(hidden_dims)-1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.LayerNorm(hidden_dims[i+1]),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
    def forward(self, x):
        return self.network(x)

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        # Ensure number of heads divides dimension evenly and is at least 1
        self.num_heads = min(num_heads, dim)
        # If dim is smaller than num_heads, reduce num_heads to largest factor of dim
        while self.num_heads > 1 and dim % self.num_heads != 0:
            self.num_heads -= 1
        
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        B_c, N_c, C_c = context.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B_c, N_c, 2, self.num_heads, C_c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MultimodalModel(nn.Module):
    def __init__(self, 
                 num_graph_features, 
                 num_outputs=1, 
                 image_dim=2048,  # ResNet-101 output dimension
                 graph_hidden_dims=[512, 1024, 16],  # Graph feature processing dimensions
                 dropout_rate=0.3):
        super(MultimodalModel, self).__init__()
        
        # Load pretrained ResNet-101 from Hugging Face
        self.resnet = ResNetForImageClassification.from_pretrained("microsoft/resnet-101")
        # Remove the classification head to get features
        self.resnet.classifier = nn.Identity()
        
        # Graph feature processing
        self.graph_processor = GraphFeatureProcessor(
            input_dim=num_graph_features,
            hidden_dims=graph_hidden_dims,
            dropout=dropout_rate
        )
        
        # Get the final graph feature dimension
        graph_dim = self.graph_processor.output_dim
        
        # Project image features to match graph dimension
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, graph_dim),
            nn.LayerNorm(graph_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Cross-attention for feature fusion
        self.cross_attention = CrossAttention(
            dim=graph_dim,  # Use graph dimension for cross-attention
            num_heads=8,
            qkv_bias=False,
            attn_drop=dropout_rate,
            proj_drop=dropout_rate
        )
        
        # Final prediction layers
        self.fusion_fc = nn.Sequential(
            nn.Linear(graph_dim + image_dim, graph_dim),  # Combine both feature types
            nn.LayerNorm(graph_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(graph_dim, graph_dim // 2),
            nn.LayerNorm(graph_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(graph_dim // 2, num_outputs)
        )
    
    def forward(self, image, graph_features):
        # Process image through pretrained ResNet
        outputs = self.resnet(image)
        # ResNet outputs shape: [batch, channels, height, width]
        # We need to flatten the spatial dimensions
        image_features = outputs.logits.squeeze(-1).squeeze(-1)  # shape: [batch, image_dim]
        
        # Process graph features
        graph_features = self.graph_processor(graph_features)  # shape: [batch, graph_dim]
        
        # Project image features to match graph dimension
        projected_image_features = self.image_projection(image_features)  # shape: [batch, graph_dim]
        
        # Cross-attention between image and graph features
        fused_features = self.cross_attention(
            rearrange(graph_features, 'b d -> b 1 d'),
            rearrange(projected_image_features, 'b d -> b 1 d')
        )
        fused_features = rearrange(fused_features, 'b 1 d -> b d')
        
        # Combine features
        combined = torch.cat([image_features, fused_features], dim=1)
        
        # Final prediction
        out = self.fusion_fc(combined)
        return out
