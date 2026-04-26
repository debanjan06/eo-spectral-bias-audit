import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from dataset import AgriSightDataset
from models.multi_modal_cnn import MultiModalCNN
from sklearn.metrics import classification_report

def evaluate_baseline():
    print("LOG: Initiating baseline evaluation on California control group...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Updated CSV path to reflect AgriSight branding
    metadata_path = 'data/processed/agrisight_training_dataset.csv'
    patches_dir = 'data/processed/california_patches'
    
    try:
        metadata_df = pd.read_csv(metadata_path)
    except FileNotFoundError:
        print(f"ERROR: Metadata file not found at {metadata_path}")
        return

    dataset = AgriSightDataset(metadata_df=metadata_df, patches_dir=patches_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model = MultiModalCNN(num_classes=3).to(device)
    
    try:
        model.load_state_dict(torch.load('models/best_baseline_model.pth', map_location=device, weights_only=True))
        print("SUCCESS: Baseline weights loaded successfully.")
    except FileNotFoundError:
        print("ERROR: Models weights not found. Ensure train.py has been executed.")
        return

    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            spatial = batch['spatial'].to(device)
            tabular = batch['tabular'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(spatial, tabular)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    target_names = ['Healthy', 'Stressed', 'Diseased']
    print("\n--- BASELINE METRICS REPORT ---")
    print(classification_report(all_labels, all_preds, target_names=target_names))

if __name__ == "__main__":
    evaluate_baseline()
