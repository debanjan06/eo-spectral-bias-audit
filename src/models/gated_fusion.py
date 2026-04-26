import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedMultimodalUnit(nn.Module):
    def __init__(self, spatial_dim=128, tabular_dim=64, output_dim=128):
        """
        Learns dynamic weights to fuse spatial and tabular modalities.
        The gate mechanism prevents 'Spectral Bias' by allowing the model to 
        suppress biased meteorological signals when they conflict with spatial evidence.
        """
        super(GatedMultimodalUnit, self).__init__()
        
        # Latent feature transformations
        self.spatial_transform = nn.Linear(spatial_dim, output_dim)
        self.tabular_transform = nn.Linear(tabular_dim, output_dim)
        
        # The Sigmoid Gate: calculates the trust ratio between modalities
        self.gate = nn.Linear(spatial_dim + tabular_dim, output_dim)

    def forward(self, x_spatial, x_tabular):
        # Generate non-linear latent representations
        h_spatial = torch.tanh(self.spatial_transform(x_spatial))
        h_tabular = torch.tanh(self.tabular_transform(x_tabular))
        
        # Calculate gate value z (importance of spatial vs tabular)
        z = torch.sigmoid(self.gate(torch.cat([x_spatial, x_tabular], dim=1)))
        
        # Gated fusion: z acts as the spatial weight, (1-z) as the tabular weight
        fused_rep = z * h_spatial + (1 - z) * h_tabular
        
        return fused_rep, z

class RobustAgriSightNet(nn.Module):
    def __init__(self, num_classes=3):
        """
        Multi-modal architecture optimized for OOD robustness using GMU.
        """
        super(RobustAgriSightNet, self).__init__()
        
        # Spatial branch: extracts features from 4-channel imagery
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten() # Feature dimension: 32
        )
        
        # Tabular branch: extracts features from meteorological vectors
        self.tab_mlp = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU()
        )
        
        # Gated Fusion Layer
        self.gmu = GatedMultimodalUnit(spatial_dim=32, tabular_dim=16, output_dim=64)
        
        # Final classification head
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, spatial, tabular):
        feat_spatial = self.cnn(spatial)
        feat_tabular = self.tab_mlp(tabular)
        
        # Dynamic fusion and gate extraction for interpretability
        fused_rep, gate_values = self.gmu(feat_spatial, feat_tabular)
        
        logits = self.classifier(fused_rep)
        return logits, gate_values

if __name__ == "__main__":
    # Diagnostic test for architecture integrity
    print("LOG: Initializing RobustAgriSightNet architecture test...")
    model = RobustAgriSightNet(num_classes=3)
    
    dummy_img = torch.randn(1, 4, 32, 32)
    dummy_tab = torch.randn(1, 6)
    
    output, gate = model(dummy_img, dummy_tab)
    print(f"SUCCESS: Output tensor shape: {output.shape}")
    print(f"REPORT: Mean Gate Value (Spatial Reliance): {gate.mean().item():.4f}")
