import torch
import torch.nn as nn
from torch_geometric.nn import SchNet, GCNConv, global_mean_pool
from torchvision import models
from torch_geometric.nn import global_mean_pool

class TabularGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return global_mean_pool(x, batch)

class MultiModalModel(nn.Module):
    def __init__(self, tabular_dim, gnn_hidden, gnn_out, schnet_out, resnet_out, fusion_dim, num_targets, dropout_rate=0.5):
        super().__init__()
        self.schnet = SchNet(
            hidden_channels=schnet_out, 
            num_filters=schnet_out, 
            num_interactions=2,  # Reduced from 3
            num_gaussians=10, 
            cutoff=10.0, 
            output_dim=schnet_out
        )
        
        # Create ResNet with regularization
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Sequential(
            nn.Linear(512, resnet_out),
            nn.BatchNorm1d(resnet_out),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.resnet = resnet
        
        self.tabular_gnn = TabularGNN(tabular_dim, gnn_hidden, gnn_out, dropout_rate)
        
        # Project all modalities to the same dimension for attention
        self.proj_dim = fusion_dim  # Use fusion_dim as the common dimension
        self.schnet_proj = nn.Linear(schnet_out, self.proj_dim)
        self.resnet_proj = nn.Linear(resnet_out, self.proj_dim)
        self.gnn_proj = nn.Linear(gnn_out, self.proj_dim)

        # Transformer-style attention: MultiheadAttention
        self.num_heads = 4  # You can increase this if desired
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.proj_dim, num_heads=self.num_heads, batch_first=True)
        
        self.fusion = nn.Sequential(
            nn.Linear(self.proj_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, num_targets)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, schnet_data, image_batch, tabular_x, edge_index, tabular_batch):
        schnet_feats = self.schnet(schnet_data.z, schnet_data.pos, batch=schnet_data.batch)
        img_feats = self.resnet(image_batch)
        gnn_feats = self.tabular_gnn(tabular_x, edge_index, tabular_batch)
        
        # Project to common dimension
        schnet_proj = self.schnet_proj(schnet_feats)
        img_proj = self.resnet_proj(img_feats)
        gnn_proj = self.gnn_proj(gnn_feats)
        
        # Stack: (batch, 3, proj_dim)
        feats = torch.stack([schnet_proj, img_proj, gnn_proj], dim=1)
        # MultiheadAttention expects (batch, seq, dim) if batch_first=True
        # Query, Key, Value are all the same for self-attention
        attn_output, attn_weights = self.multihead_attn(feats, feats, feats, need_weights=True)
        # attn_output: (batch, 3, proj_dim), attn_weights: (batch, num_heads, 3, 3)
        # Pool: mean over sequence (modalities)
        fused = attn_output.mean(dim=1)
        return self.fusion(fused)