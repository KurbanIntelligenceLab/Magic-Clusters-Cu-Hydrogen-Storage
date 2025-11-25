from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models


class ResNetOnlyModel(nn.Module):
    """ResNet-only model for predicting material properties from structure images."""

    def __init__(
        self,
        resnet_out: int,
        num_targets: int,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        # Use ResNet50 as backbone
        resnet: nn.Module = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.fc = nn.Sequential(
            nn.Linear(2048, resnet_out),
            nn.BatchNorm1d(resnet_out),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.resnet: nn.Module = resnet
        
        # Direct prediction head from ResNet features
        self.predictor: nn.Sequential = nn.Sequential(
            nn.Linear(resnet_out, num_targets),
        )
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for linear layers."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, image_batch: Tensor) -> torch.Tensor:
        """
        Forward pass using only ResNet image features.
        
        Args:
            image_batch: Image tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Predictions tensor of shape (batch_size, num_targets)
        """
        img_feats: torch.Tensor = self.resnet(image_batch)
        predictions: torch.Tensor = self.predictor(img_feats)
        return predictions

