# src/model.py
import torch
import torch.nn as nn
import pytorch_lightning as pl

class AgriCNN(pl.LightningModule):
    def __init__(self, num_classes=3, lr=5e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Spatial encoder
        self.spatial = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Spectral encoder
        self.spectral = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, spatial, spectral):
        spatial_feat = self.spatial(spatial)
        spectral_feat = self.spectral(spectral)
        combined = torch.cat([spatial_feat, spectral_feat], dim=1)
        return self.fusion(combined)
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch['spatial'], batch['spectral'])
        loss = self.criterion(outputs, batch['label'])
        acc = (outputs.argmax(1) == batch['label']).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch['spatial'], batch['spectral'])
        loss = self.criterion(outputs, batch['label'])
        acc = (outputs.argmax(1) == batch['label']).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}