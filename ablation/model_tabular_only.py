import torch
import torch.nn as nn
from torch import Tensor


class TabularOnlyModel(nn.Module):
    """Tabular-only model for predicting material properties from tabular features."""

    def __init__(
        self,
        tabular_dim: int,
        num_targets: int,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        # MLP for tabular features
        self.predictor: nn.Sequential = nn.Sequential(
            nn.Linear(tabular_dim, num_targets),
            nn.LayerNorm(tabular_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(tabular_dim * 2, tabular_dim),
            nn.LayerNorm(tabular_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(tabular_dim, tabular_dim // 2),
            nn.LayerNorm(tabular_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(tabular_dim // 2, num_targets),
        )
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for linear layers."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tabular_x: Tensor) -> torch.Tensor:
        """
        Forward pass using only tabular features.
        
        Args:
            tabular_x: Tabular features tensor of shape (batch_size, tabular_dim)
        
        Returns:
            Predictions tensor of shape (batch_size, num_targets)
        """
        predictions: torch.Tensor = self.predictor(tabular_x)
        return predictions

