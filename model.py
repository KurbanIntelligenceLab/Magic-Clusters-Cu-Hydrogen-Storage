import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalModel(nn.Module):
    def __init__(self, num_graph_features, num_graph_outputs, num_outputs=1, hidden_dimension=128, dropout_rate=0.3):
        super(MultimodalModel, self).__init__()
        
        # Image branch: Use pretrained ResNet-101
        self.cnn = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        # Remove the final classification layer to get feature embeddings
        self.cnn.fc = nn.Identity()
        cnn_out_features = 2048  # For ResNet-101
        
        # Graph branch: Simple fully connected network for material features
        self.graph_fc = nn.Sequential(
            nn.Linear(num_graph_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_graph_outputs),
        )
        
        # Fusion and final prediction
        self.fusion_fc = nn.Sequential(
            nn.Linear(cnn_out_features + num_graph_outputs, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dimension, hidden_dimension//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dimension//2, num_outputs)
        )
    
    def forward(self, image, graph_features):
        # Process image
        image_features = self.cnn(image)  # shape: [batch, 2048]
        
        # Process graph features
        graph_features = self.graph_fc(graph_features)  # shape: [batch, num_graph_outputs]
        
        # Fuse features by concatenation
        combined = torch.cat((image_features, graph_features), dim=1)
        out = self.fusion_fc(combined)
        return out
