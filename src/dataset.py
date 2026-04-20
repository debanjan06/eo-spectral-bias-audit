# src/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path

class EOSpectralDataset(Dataset):
    def __init__(self, metadata_df, patches_dir, transform=None):
        self.metadata = metadata_df.copy()
        self.patches_dir = Path(patches_dir)
        self.transform = transform
        
        # Encoding labels for the PyTorch CrossEntropyLoss function
        self.label_map = {'healthy': 0, 'stressed': 1, 'diseased': 2}
        self.metadata['label_encoded'] = self.metadata['label'].map(self.label_map)
        
        print(f"✅ Global Dataset Initialized: {len(self.metadata)} samples")
        print(self.metadata['label'].value_counts())
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # ==========================================
        # 1. SPATIAL PATHWAY (Sentinel-2 Imagery)
        # ==========================================
        patch_path = self.patches_dir / f"{row['patch_id']}.npy"
        patch = np.load(patch_path)
        
        # Normalize Sentinel-2 10m Surface Reflectance
        patch = patch.astype(np.float32) / 10000.0  
        
        if self.transform:
            patch = self.transform(patch)
            
        spatial_tensor = torch.from_numpy(patch).float()
        
        # ==========================================
        # 2. TABULAR PATHWAY (Indices + Weather)
        # ==========================================
        # This is the exact tensor that feeds the MLP and caused the Australia bias!
        tabular_features = [
            row['ndvi_mean'],
            row['evi_mean'],
            row['savi_mean'],          # The key to diagnosing the arid-zone bias
            row['temp_max_c'],         # Open-Meteo Maximum Temperature
            row['temp_min_c'],         # Open-Meteo Minimum Temperature
            row['rainfall_mm']         # Open-Meteo Precipitation
        ]
        
        tabular_tensor = torch.tensor(tabular_features, dtype=torch.float32)
        label = torch.tensor(row['label_encoded'], dtype=torch.long)
        
        return {
            'spatial': spatial_tensor,
            'tabular': tabular_tensor,
            'label': label
        }