from typing import Any

import torch
import torch.nn as nn
from torch_geometric.nn import SchNet


class SchNetOnlyModel(nn.Module):
    """SchNet-only model for predicting material properties from atomic coordinates."""

    def __init__(
        self,
        schnet_out: int,
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
        # Direct prediction head from SchNet features
        self.predictor: nn.Sequential = nn.Sequential(
            nn.Linear(schnet_out, num_targets),
        )
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for linear layers."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, schnet_data: Any) -> torch.Tensor:
        """
        Forward pass using only SchNet geometric features.
        
        Args:
            schnet_data: PyTorch Geometric data with atomic numbers (z) and positions (pos)
        
        Returns:
            Predictions tensor of shape (batch_size, num_targets)
        """
        schnet_feats: torch.Tensor = self.schnet(
            schnet_data.z, schnet_data.pos, batch=schnet_data.batch
        )
        predictions: torch.Tensor = self.predictor(schnet_feats)
        return predictions

