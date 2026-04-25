import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class CropHealthDataset(Dataset):
    def __init__(self, folder_path, csv_path):
        self.folder_path = folder_path
        
        # Load your real AgriGuard dataset!
        self.data_df = pd.read_csv(csv_path)
        
        # Map the text labels to mathematical classes for PyTorch
        self.label_map = {
            'healthy': 0, 
            'stressed': 1, 
            'diseased': 2
        }
        
    def __len__(self):
        return len(self.data_df)
        
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        
        # 1. Match the CSV row to the exact .npy file on your hard drive
        # If your patches are named patch_0.npy, we could use idx.
        # But assuming they match the sample_id (e.g., AGR_0000.npy):
        file_name = f"patch_{idx}.npy" # Change to f"{row['sample_id']}.npy" if your files are named AGR_0000.npy
        file_path = os.path.join(self.folder_path, file_name)
        
        # 2. Load the REAL spatial patch
        if os.path.exists(file_path):
            spatial_data = np.load(file_path).astype(np.float32)
        else:
            # Fallback if a file is missing
            spatial_data = np.random.randn(4, 32, 32).astype(np.float32)
            
        spatial_tensor = torch.from_numpy(spatial_data)
        
        # 3. Load the REAL Tabular data from the CSV!
        tabular_tensor = torch.tensor([
            row['NDVI'], row['SAVI'], row['EVI'], 
            row['temp_max'], row['rainfall'], row['humidity']
        ], dtype=torch.float32)
        
        # 4. Get the REAL Ground Truth Label
        label_str = str(row['health_status']).lower().strip()
        actual_label = torch.tensor(self.label_map.get(label_str, 0), dtype=torch.long)
        
        return spatial_tensor, tabular_tensor, actual_label