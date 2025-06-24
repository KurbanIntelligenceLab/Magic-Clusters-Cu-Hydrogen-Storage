from typing import Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv, SchNet
from torchvision import models


class TabularGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv1: GCNConv = GCNConv(in_dim, hidden_dim)
        self.conv2: GCNConv = GCNConv(hidden_dim, out_dim)
        self.relu: nn.ReLU = nn.ReLU()
        # Add projection layers for residual connections
        self.input_proj: nn.Linear = nn.Linear(in_dim, out_dim)
        # Removed dropout and batch norm for tabular branch

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        # Store original input for residual connection
        x_orig = x
        # GNN processing
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # Add residual connection to preserve node identity
        x_residual = self.input_proj(x_orig)
        x = x + x_residual
        return x  # Return node-level features instead of pooled features


class MultiModalModel(nn.Module):
    def __init__(
        self,
        tabular_dim: int,
        gnn_hidden: int,
        gnn_out: int,
        schnet_out: int,
        resnet_out: int,
        fusion_dim: int,
        num_targets: int,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.schnet: SchNet = SchNet(
            hidden_channels=schnet_out,
            num_filters=schnet_out,
            num_interactions=2,
            num_gaussians=10,
            cutoff=10.0,
            output_dim=schnet_out,
        )
        resnet: nn.Module = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.fc = nn.Sequential(
            nn.Linear(2048, resnet_out),
            nn.BatchNorm1d(resnet_out),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.resnet: nn.Module = resnet
        self.tabular_gnn: TabularGNN = TabularGNN(tabular_dim, gnn_hidden, gnn_out)
        self.proj_dim: int = fusion_dim
        self.schnet_proj: nn.Sequential = nn.Sequential(
            nn.Linear(schnet_out, self.proj_dim), nn.LayerNorm(self.proj_dim)
        )
        self.resnet_proj: nn.Sequential = nn.Sequential(
            nn.Linear(resnet_out, self.proj_dim), nn.LayerNorm(self.proj_dim)
        )
        self.gnn_proj: nn.Sequential = nn.Sequential(
            nn.Linear(gnn_out, self.proj_dim), nn.LayerNorm(self.proj_dim)
        )
        # Enhanced attention mechanism with temperature scaling and minimum weights
        self.attention: nn.Sequential = nn.Sequential(
            nn.Linear(self.proj_dim, self.proj_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.proj_dim // 2, 1),
        )
        # Temperature parameter for softmax (lower = sharper distribution)
        self.temperature: nn.Parameter = nn.Parameter(torch.ones(1) * 1.0)
        # Minimum weight constraint
        self.min_weight: float = 0.1  # Ensure each modality gets at least 10% weight
        # Alternative: Gated fusion mechanism
        self.gate_networks: nn.ModuleList = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.proj_dim, self.proj_dim // 2),
                    nn.ReLU(),
                    nn.Linear(self.proj_dim // 2, 1),
                    nn.Sigmoid(),
                )
                for _ in range(3)  # One gate per modality
            ]
        )
        # Residual connection weights
        self.residual_weights: nn.Parameter = nn.Parameter(
            torch.ones(3) * 0.3
        )  # Equal initial weights
        self.fusion: nn.Sequential = nn.Sequential(
            nn.Linear(self.proj_dim, num_targets)
        )
        # --- Auxiliary heads for each modality ---
        self.tabular_head: nn.Linear = nn.Linear(gnn_out, num_targets)
        self.image_head: nn.Linear = nn.Linear(resnet_out, num_targets)
        self.schnet_head: nn.Linear = nn.Linear(schnet_out, num_targets)
        # Initialize weights
        self.apply(self._init_weights)

    # Initialize weights
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        schnet_data: Any,
        image_batch: Tensor,
        tabular_x: Tensor,
        edge_index: Tensor,
        tabular_batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        schnet_feats: Tensor = self.schnet(
            schnet_data.z, schnet_data.pos, batch=schnet_data.batch
        )
        img_feats: Tensor = self.resnet(image_batch)
        gnn_feats: Tensor = self.tabular_gnn(tabular_x, edge_index, tabular_batch)
        # Project to common dimension
        schnet_proj: Tensor = self.schnet_proj(schnet_feats)
        img_proj: Tensor = self.resnet_proj(img_feats)
        gnn_proj: Tensor = self.gnn_proj(gnn_feats)
        # Stack: (batch, 3, proj_dim)
        feats: Tensor = torch.stack([schnet_proj, img_proj, gnn_proj], dim=1)
        # Method 1: Enhanced attention with temperature scaling
        attention_scores: Tensor = self.attention(feats).squeeze(-1)  # (batch, 3)
        attention_scores = attention_scores / torch.clamp(
            self.temperature, min=0.1, max=10.0
        )
        attention_weights: Tensor = torch.softmax(attention_scores, dim=1)  # (batch, 3)
        # Apply minimum weight constraint
        min_weights: Tensor = torch.ones_like(attention_weights) * self.min_weight
        attention_weights = torch.maximum(attention_weights, min_weights)
        attention_weights = attention_weights / attention_weights.sum(
            dim=1, keepdim=True
        )
        # Method 2: Gated fusion
        gates = []
        for i, gate_net in enumerate(self.gate_networks):
            gate = gate_net(feats[:, i, :])  # (batch, 1)
            gates.append(gate)
        gates_tensor: Tensor = torch.cat(gates, dim=1)  # (batch, 3)
        # Combine attention and gated fusion
        combined_weights: Tensor = 0.7 * attention_weights + 0.3 * gates_tensor
        # Method 3: Residual connections
        residual_contrib: Tensor = self.residual_weights.unsqueeze(0)  # (1, 3)
        final_weights: Tensor = combined_weights + 0.1 * residual_contrib
        final_weights = final_weights / final_weights.sum(dim=1, keepdim=True)
        # Weighted sum: (batch, 3, proj_dim) * (batch, 3, 1) -> (batch, proj_dim)
        fused: Tensor = torch.sum(feats * final_weights.unsqueeze(-1), dim=1)
        fusion_out: Tensor = self.fusion(fused)
        _tabular: Tensor = self.tabular_head(gnn_feats)
        _image: Tensor = self.image_head(img_feats)
        _schnet: Tensor = self.schnet_head(schnet_feats)
        return fusion_out, _tabular, _image, _schnet, final_weights
