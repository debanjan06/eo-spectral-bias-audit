import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from models.multi_modal_cnn import MultiModalCNN

class AuditDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        self.data_df = pd.read_csv(csv_path)
    def __len__(self):
        return len(self.data_df)
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        # Using dummy spatial data to isolate the tabular "Weather Prior" effect
        spatial_tensor = torch.randn(4, 32, 32)
        
        # Load the Punjab tabular data
        tabular_tensor = torch.tensor([
            row.get('NDVI', 0.4), row.get('SAVI', 0.3), row.get('EVI', 0.25), 
            row.get('temp_max', 28.0), row.get('rainfall', 5.0), row.get('humidity', 45.0)
        ], dtype=torch.float32)
        return spatial_tensor, tabular_tensor, torch.tensor(-1, dtype=torch.long)

def run_punjab_audit():
    print("🌾 Initiating Punjab Wheat Belt Stress Test...")
    os.makedirs('results', exist_ok=True)
    
    # Load model weights
    model = MultiModalCNN(num_classes=3)
    model.load_state_dict(torch.load('models/best_baseline_model.pth', weights_only=True))
    model.eval()
    
    # Point to Punjab CSV
    dataset = AuditDataset(csv_path='data/raw/punjab_wheat_belt_tile_weather.csv')
    audit_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    healthy_count = 0
    total = 0

    with torch.no_grad():
        for spatial, tabular, _ in audit_loader:
            outputs = model(spatial, tabular)
            _, predicted = torch.max(outputs, 1)
            healthy_count += (predicted == 0).sum().item()
            total += spatial.size(0)

    bias_rate = (healthy_count / total) * 100
    
    # --- GENERATE COMPARISON PLOT ---
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Comparison: California (72.3%) vs Punjab result
    categories = ['California (Control)', 'Punjab (Audit)']
    values = [72.3, bias_rate]
    
    ax = sns.barplot(x=categories, y=values, palette=['#2ecc71', '#3498db'])
    plt.title('Cross-Continental Audit: California vs. Punjab (Wheat Belt)', fontsize=14)
    plt.ylabel('Percentage of "Healthy" Predictions', fontsize=12)
    plt.ylim(0, 110)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', weight='bold')

    plot_path = 'results/punjab_bias_audit_plot.png'
    plt.savefig(plot_path)
    print(f"✅ Punjab audit plot saved to: {plot_path}")
    print(f"🚨 Punjab 'Healthy' Prediction Rate: {bias_rate:.2f}%")

if __name__ == "__main__":
    run_punjab_audit()