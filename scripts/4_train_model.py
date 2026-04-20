import sys
sys.path.append('.')

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.dataset import RealSatelliteDataset
from src.model import AgriCNN

def train():
    
    # Load data
    metadata = pd.read_csv(r"C:\Users\DEBANJAN SHIL\Documents\AgriGuard\scripts\data\real_patches\metadata.csv")
    metadata = metadata[metadata['label'].isin(['healthy', 'stressed', 'diseased'])]
    
    print(f"Total samples: {len(metadata)}")
    print(metadata['label'].value_counts())
    
    # Split
    train_df, val_df = train_test_split(
        metadata, 
        test_size=0.2, 
        stratify=metadata['label'],
        random_state=42
    )
    
    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}")
    
    # Datasets
    train_dataset = RealSatelliteDataset(train_df, 'data/real_patches')
    val_dataset = RealSatelliteDataset(val_df, 'data/real_patches')
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=0)
    
    # Model
    model = AgriCNN(num_classes=3, lr=5e-4)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=30,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=7),
            pl.callbacks.ModelCheckpoint(
                dirpath='models',
                filename='agriguard_{epoch}_{val_acc:.3f}',
                monitor='val_acc',
                mode='max',
                save_top_k=1
            )
        ],
        log_every_n_steps=5
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Save final
    Path('models').mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'class_dist': train_df['label'].value_counts().to_dict(),
        'best_val_acc': float(trainer.checkpoint_callback.best_model_score)
    }, 'models/agriguard_final.pth')
    
    print(f"\n✓ Training complete!")
    print(f"Best val_acc: {trainer.checkpoint_callback.best_model_score:.3f}")

if __name__ == "__main__":
    train()