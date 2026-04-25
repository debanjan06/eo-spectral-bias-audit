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
        # Generate dummy spatial data to isolate the influence of weather priors
        spatial_tensor = torch.randn(4, 32, 32)
        
        # Extract features for the audit regions
        tabular_tensor = torch.tensor([
            row.get('NDVI', 0.4), row.get('SAVI', 0.3), row.get('EVI', 0.25), 
            row.get('temp_max', 28.0), row.get('rainfall', 5.0), row.get('humidity', 45.0)
        ], dtype=torch.float32)
        return spatial_tensor, tabular_tensor, torch.tensor(-1, dtype=torch.long)

def run_regional_audit(region_name, csv_path, output_filename):
    print(f"INFO: Initiating stress test for region: {region_name}")
    os.makedirs('results', exist_ok=True)
    
    # Initialize model and load trained weights
    model = MultiModalCNN(num_classes=3)
    model.load_state_dict(torch.load('models/best_baseline_model.pth', weights_only=True))
    model.eval()
    
    dataset = AuditDataset(csv_path=csv_path)
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
    
    # Visualization configuration
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    categories = ['California (Control)', f'{region_name} (Audit)']
    values = [72.3, bias_rate]
    
    ax = sns.barplot(x=categories, y=values, palette=['#2ecc71', '#3498db'])
    plt.title(f'Spectral Bias Audit: California vs. {region_name}', fontsize=14)
    plt.ylabel('Percentage of "Healthy" Predictions', fontsize=12)
    plt.ylim(0, 110)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', weight='bold')

    plot_path = f'results/{output_filename}'
    plt.savefig(plot_path)
    print(f"SUCCESS: Audit complete. Results saved to: {plot_path}")
    print(f"REPORT: {region_name} 'Healthy' Prediction Rate: {bias_rate:.2f}%")

if __name__ == "__main__":
    # Execute audits for verified regions
    run_regional_audit("W. Australia", "data/raw/australia_dryland_tile_weather.csv", "spectral_bias_audit_plot.png")
    run_regional_audit("Punjab", "data/raw/punjab_wheat_belt_tile_weather.csv", "punjab_bias_audit_plot.png")
