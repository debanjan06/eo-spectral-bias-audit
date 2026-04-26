import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from dataset import AgriSightDataset
from models.multi_modal_cnn import MultiModalCNN
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_baseline():
    print("LOG: Initiating baseline evaluation on control distribution...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Data Preparation
    # Loading the California-based training/validation metadata
    metadata_df = pd.read_csv('data/processed/agriguard_training_dataset.csv')
    patches_dir = 'data/processed/california_patches'
    
    dataset = AgriSightDataset(metadata_df=metadata_df, patches_dir=patches_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 2. Model Initialization
    model = MultiModalCNN(num_classes=3).to(device)
    
    # Secure weight loading
    try:
        model.load_state_dict(torch.load('models/best_baseline_model.pth', map_location=device, weights_only=True))
        print("SUCCESS: Model weights loaded for baseline evaluation.")
    except FileNotFoundError:
        print("ERROR: Weights not found at models/best_baseline_model.pth. Run train.py first.")
        return

    model.eval()
    
    all_preds = []
    all_labels = []
    
    # 3. Inference Loop
    print("LOG: Running inference on validation samples...")
    with torch.no_grad():
        for batch in loader:
            spatial = batch['spatial'].to(device)
            tabular = batch['tabular'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(spatial, tabular)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 4. Metrics Reporting
    target_names = ['Healthy', 'Stressed', 'Diseased']
    
    print("\n--- BASELINE PERFORMANCE REPORT (CALIFORNIA) ---")
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    # Calculating specific Healthy Prediction Rate for comparison with Audit
    correct_healthy = sum((np.array(all_preds) == 0) & (np.array(all_labels) == 0))
    total_healthy_ground_truth = sum(np.array(all_labels) == 0)
    
    healthy_accuracy = (correct_healthy / total_healthy_ground_truth) * 100 if total_healthy_ground_truth > 0 else 0
    print(f"REPORT: Recall for 'Healthy' class: {healthy_accuracy:.2f}%")
    print("------------------------------------------------\n")

if __name__ == "__main__":
    evaluate_baseline()
