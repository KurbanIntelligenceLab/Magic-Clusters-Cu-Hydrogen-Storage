import torch
import torch.nn as nn
from torch_geometric.nn import SchNet, GCNConv, global_mean_pool
from torchvision import models
from torch_geometric.nn import global_mean_pool

class TabularGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        return global_mean_pool(x, batch)

class MultiModalModel(nn.Module):
    def __init__(self, tabular_dim, gnn_hidden, gnn_out, schnet_out, resnet_out, fusion_dim, num_targets):
        super().__init__()
        self.schnet = SchNet(hidden_channels=schnet_out, num_filters=schnet_out, num_interactions=3, num_gaussians=25, cutoff=10.0, output_dim=schnet_out)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        self.resnet = resnet
        self.tabular_gnn = TabularGNN(tabular_dim, gnn_hidden, gnn_out)
        fusion_input_dim = schnet_out + resnet_out + gnn_out
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, num_targets)
        )
        self.attn = nn.Parameter(torch.ones(3))  # One for each modality

    def forward(self, schnet_data, image_batch, tabular_x, edge_index, tabular_batch):
        schnet_feats = self.schnet(schnet_data.z, schnet_data.pos, batch=schnet_data.batch)
        img_feats = self.resnet(image_batch)
        gnn_feats = self.tabular_gnn(tabular_x, edge_index, tabular_batch)
        # Attention-weighted fusion
        feats = [schnet_feats, img_feats, gnn_feats]
        attn_weights = torch.softmax(self.attn, dim=0)
        fused = torch.cat([w * f for w, f in zip(attn_weights, feats)], dim=1)
        return self.fusion(fused)