import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalModel(nn.Module):
    def __init__(self, num_graph_features, num_graph_outputs, num_outputs=1, hidden_dimension=128, dropout_rate=0.3):
        super(MultimodalModel, self).__init__()
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # with torch.no_grad():
        #     original_weights = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).conv1.weight
        #     self.cnn.conv1.weight = nn.Parameter(original_weights.mean(dim=1, keepdim=True))

        self.cnn.fc = nn.Identity()
        cnn_out_features = 2048  # For ResNet-50
        
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
