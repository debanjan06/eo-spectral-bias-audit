import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def extract_patches(tiff_path, n_patches=500, patch_size=32):
    
    print(f"Reading {tiff_path}")
    
    with rasterio.open(tiff_path) as src:
        print(f"Shape: {src.shape}")
        print(f"Bands: {src.count} (B2, B3, B4, B8)")
        
        data = src.read()
        height, width = src.shape
        
        patches_data = []
        output_dir = Path('data/real_patches')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(42)
        
        extracted = 0
        attempts = 0
        
        pbar = tqdm(total=n_patches)
        
        while extracted < n_patches and attempts < n_patches * 3:
            attempts += 1
            
            y = np.random.randint(0, height - patch_size)
            x = np.random.randint(0, width - patch_size)
            
            patch = data[:, y:y+patch_size, x:x+patch_size]
            
            # Skip invalid patches
            if patch.sum() == 0 or (patch == 0).sum() > (patch.size * 0.1):
                continue
            
            # Calculate indices
            b2 = patch[0].astype(float)
            b3 = patch[1].astype(float)
            b4 = patch[2].astype(float)
            b8 = patch[3].astype(float)
            
            # NDVI
            ndvi = np.where((b8 + b4) != 0, (b8 - b4) / (b8 + b4), 0)
            
            # EVI
            evi = np.where(
                (b8 + 6*b4 - 7.5*b2 + 10000) != 0,
                2.5 * (b8 - b4) / (b8 + 6*b4 - 7.5*b2 + 10000),
                0
            )
            
            # SAVI
            savi = np.where(
                (b8 + b4 + 5000) != 0,
                1.5 * (b8 - b4) / (b8 + b4 + 5000),
                0
            )
            
            ndvi_mean = np.mean(ndvi)
            
            # Skip invalid NDVI
            if ndvi_mean < -0.5 or ndvi_mean > 1.0:
                continue
            
            # Get coordinates
            lon, lat = rasterio.transform.xy(src.transform, y + patch_size//2, x + patch_size//2)
            
            # Save
            patch_id = f'patch_{extracted:04d}'
            np.save(output_dir / f'{patch_id}.npy', patch)
            
            patches_data.append({
                'patch_id': patch_id,
                'ndvi_mean': float(np.mean(ndvi)),
                'evi_mean': float(np.mean(evi)),
                'savi_mean': float(np.mean(savi)),
                'ndvi_std': float(np.std(ndvi)),
                'lat': lat,
                'lon': lon
            })
            
            extracted += 1
            pbar.update(1)
        
        pbar.close()
        
        df = pd.DataFrame(patches_data)
        df.to_csv(output_dir / 'metadata.csv', index=False)
        
        print(f"\n✓ Extracted {len(df)} patches")
        print(f"\nNDVI distribution:")
        print(df['ndvi_mean'].describe())
        
        return df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python 2_extract_patches.py <tiff_path>")
        sys.exit(1)
    
    tiff_path = sys.argv[1]
    extract_patches(tiff_path, n_patches=500)