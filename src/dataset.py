import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path

class AgriSightDataset(Dataset):
    def __init__(self, metadata_df, patches_dir, transform=None):
        """
        Custom Dataset for multi-modal Earth Observation analysis.
        metadata_df: DataFrame containing patch metadata and weather features.
        patches_dir: Directory path containing .npy spatial patches.
        """
        self.metadata = metadata_df.copy()
        self.patches_dir = Path(patches_dir)
        self.transform = transform
        
        # Mapping string labels to integers for CrossEntropyLoss
        self.label_map = {'healthy': 0, 'stressed': 1, 'diseased': 2}
        if 'label' in self.metadata.columns:
            self.metadata['label_encoded'] = self.metadata['label'].map(self.label_map)
        
        print(f"LOG: Dataset initialized with {len(self.metadata)} samples.")

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # 1. Spatial Pathway: Sentinel-2 Imagery
        # Loads 4-channel (RGB-NIR) patches
        patch_path = self.patches_dir / f"{row['patch_id']}.npy"
        patch = np.load(patch_path)
        
        # Normalize 16-bit reflectance to [0, 1]
        patch = patch.astype(np.float32) / 10000.0  
        
        if self.transform:
            patch = self.transform(patch)
            
        spatial_tensor = torch.from_numpy(patch).float()
        
        # 2. Tabular Pathway: Meteorological Features
        # These features were identified as the primary source of Spectral Bias
        tabular_features = [
            row.get('ndvi_mean', 0.0),
            row.get('evi_mean', 0.0),
            row.get('savi_mean', 0.0),
            row.get('temp_max_c', 0.0),
            row.get('temp_min_c', 0.0),
            row.get('rainfall_mm', 0.0)
        ]
        
        tabular_tensor = torch.tensor(tabular_features, dtype=torch.float32)
        
        # Handle cases where labels might be missing (e.g., during raw audit)
        label = torch.tensor(row.get('label_encoded', -1), dtype=torch.long)
        
        return {
            'spatial': spatial_tensor,
            'tabular': tabular_tensor,
            'label': label
        }
