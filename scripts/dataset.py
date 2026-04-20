# src/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path

class RealSatelliteDataset(Dataset):
    def __init__(self, metadata_df, patches_dir, transform=None):
        self.metadata = metadata_df.copy()
        self.patches_dir = Path(patches_dir)
        self.transform = transform
        
        # Encode labels
        self.label_map = {'healthy': 0, 'stressed': 1, 'diseased': 2}
        self.metadata['label_encoded'] = self.metadata['label'].map(self.label_map)
        
        print(f"Dataset size: {len(self.metadata)}")
        print(self.metadata['label'].value_counts())
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load patch
        patch = np.load(self.patches_dir / f"{row['patch_id']}.npy")
        patch = patch.astype(np.float32) / 10000.0  # Normalize
        
        if self.transform:
            patch = self.transform(patch)
        
        patch = torch.from_numpy(patch).float()
        label = torch.tensor(row['label_encoded'], dtype=torch.long)
        
        # Spectral features
        spectral = torch.tensor([
            row['ndvi_mean'],
            row['evi_mean'],
            row['savi_mean'],
            row.get('rep_mean', 715.0)
        ], dtype=torch.float32)
        
        return {
            'spatial': patch,
            'spectral': spectral,
            'label': label
        }