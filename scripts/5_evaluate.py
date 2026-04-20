import sys
sys.path.append('.')

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.dataset import RealSatelliteDataset
from src.model import AgriCNN
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def evaluate():
    
    # Load data
    metadata = pd.read_csv('data/real_patches/metadata.csv')
    metadata = metadata[metadata['label'].isin(['healthy', 'stressed', 'diseased'])]
    
    _, val_df = train_test_split(metadata, test_size=0.2, stratify=metadata['label'], random_state=42)
    
    # Dataset
    val_dataset = RealSatelliteDataset(val_df, 'data/real_patches')
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    # Load model
    checkpoint = torch.load('models/agriguard_final.pth')
    model = AgriCNN(num_classes=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Predict
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch['spatial'], batch['spectral'])
            preds = outputs.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())
    
    # Metrics
    classes = ['healthy', 'stressed', 'diseased']
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm / cm.sum(axis=1, keepdims=True), 
                annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title('Confusion Matrix (Normalized)')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    
    Path('results').mkdir(exist_ok=True)
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved confusion matrix to results/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()