import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedMultimodalUnit(nn.Module):
    def __init__(self, spatial_dim=128, tabular_dim=64, output_dim=128):
        super(GatedMultimodalUnit, self).__init__()
        """
        The GMU dynamically weights the spatial and tabular branches.
        This prevents 'Spectral Bias' by allowing the model to suppress 
        the weather signal if the satellite imagery provides conflicting info.
        """
        # Feature transformation layers
        self.spatial_transform = nn.Linear(spatial_dim, output_dim)
        self.tabular_transform = nn.Linear(tabular_dim, output_dim)
        
        # The Gate: deciding which modality to trust
        # It takes concatenated features and outputs a weight between 0 and 1
        self.gate = nn.Linear(spatial_dim + tabular_dim, output_dim)

    def forward(self, x_spatial, x_tabular):
        # 1. Get latent representations
        h_spatial = torch.tanh(self.spatial_transform(x_spatial))
        h_tabular = torch.tanh(self.tabular_transform(x_tabular))
        
        # 2. Calculate the gate value (the 'z' gate)
        z = torch.sigmoid(self.gate(torch.cat([x_spatial, x_tabular], dim=1)))
        
        # 3. Dynamic Fusion: gate * spatial + (1 - gate) * tabular
        gated_fusion = z * h_spatial + (1 - z) * h_tabular
        
        return gated_fusion, z

class RobustAgriSightNet(nn.Module):
    def __init__(self, num_classes=3):
        super(RobustAgriSightNet, self).__init__()
        
        # Spatial Branch (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten() # Output: 32
        )
        
        # Tabular Branch
        self.tab_mlp = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU()
        )
        
        # Gated Fusion Layer
        self.gmu = GatedMultimodalUnit(spatial_dim=32, tabular_dim=16, output_dim=64)
        
        # Final Classifier
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, spatial, tabular):
        feat_spatial = self.cnn(spatial)
        feat_tabular = self.tab_mlp(tabular)
        
        # Fusion and gate visualization
        fused_rep, gate_values = self.gmu(feat_spatial, feat_tabular)
        
        logits = self.classifier(fused_rep)
        return logits, gate_values

if __name__ == "__main__":
    # Test with dummy data
    model = RobustAgriSightNet(num_classes=3)
    dummy_img = torch.randn(1, 4, 32, 32)
    dummy_tab = torch.randn(1, 6)
    
    output, gate = model(dummy_img, dummy_tab)
    print(f"Fusion Output Shape: {output.shape}")
    print(f"Gate Value (Importance of Imagery): {gate.mean().item():.4f}")
