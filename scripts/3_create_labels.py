# scripts/3_create_labels.py
import pandas as pd
import numpy as np

def create_labels():
    
    df = pd.read_csv('data/real_patches/metadata.csv')
    
    print(f"Total patches: {len(df)}")
    print(f"\nNDVI distribution:")
    print(df['ndvi_mean'].describe())
    
    # Rule-based labeling
    conditions = [
        (df['ndvi_mean'] >= 0.5),
        (df['ndvi_mean'] >= 0.3) & (df['ndvi_mean'] < 0.5),
        (df['ndvi_mean'] < 0.3)
    ]
    choices = ['healthy', 'stressed', 'diseased']
    
    df['label'] = np.select(conditions, choices)
    df['confidence'] = 3
    df['labeling_method'] = 'ndvi_threshold'
    
    # Mark edge cases for review
    df['needs_review'] = False
    df.loc[(df['ndvi_mean'] > 0.28) & (df['ndvi_mean'] < 0.32), 'needs_review'] = True
    df.loc[(df['ndvi_mean'] > 0.48) & (df['ndvi_mean'] < 0.52), 'needs_review'] = True
    
    df.to_csv('data/real_patches/metadata.csv', index=False)
    
    print(f"\n✓ Labels created")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    print(f"\nEdge cases needing review: {df['needs_review'].sum()}")
    
    return df

if __name__ == "__main__":
    create_labels()