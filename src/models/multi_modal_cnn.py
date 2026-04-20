import torch
import torch.nn as nn
import pytorch_lightning as pl

class MultiModalCNN(pl.LightningModule):
    def __init__(self, num_classes=3, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        # 1. SPATIAL PATHWAY (Sentinel-2 10m Imagery)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),  # Assuming 32x32 input patches
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 2. TABULAR PATHWAY (Open-Meteo Weather + Vegetation Indices)
        # Input size is 6: [NDVI, EVI, SAVI, temp_max_c, temp_min_c, rainfall_mm]
        self.tabular_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        # 3. LATE FUSION LAYER (Where the Australia Bias Occurs!)
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 + 32, 64), # Spatial (128) + Tabular (32)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        self.criterion = nn.CrossEntropyLoss()
            
    def forward(self, spatial_tensor, tabular_tensor):
        # Extract feature vectors
        spatial_features = self.spatial_encoder(spatial_tensor)
        tabular_features = self.tabular_encoder(tabular_tensor)
        
        # Fuse the modalities
        fused_features = torch.cat([spatial_features, tabular_features], dim=1)
        
        # Final classification
        output = self.fusion_layer(fused_features)
        return output