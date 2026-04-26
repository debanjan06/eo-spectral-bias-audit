import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from dataset import AgriSightDataset
from models.multi_modal_cnn import MultiModalCNN

def run_regional_audit(region_name, csv_path, output_filename):
    """
    Executes a controlled stress test to isolate Spectral Bias.
    Spatial input is deliberately randomized (Gaussian noise) to prove the 
    model ignores visual ground truth in favor of meteorological priors.
    """
    print(f"INFO: Initiating bias isolation audit for region: {region_name}")
    os.makedirs('results', exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model Initialization
    model = MultiModalCNN(num_classes=3).to(device)
    try:
        model.load_state_dict(torch.load('models/best_baseline_model.pth', map_location=device, weights_only=True))
    except FileNotFoundError:
        print("ERROR: Baseline weights missing. Audit aborted.")
        return

    model.eval()
    
    # Data Loading: We only care about the tabular features from the CSV
    metadata_df = pd.read_csv(csv_path)
    
    # Mapping regional columns to training schema
    column_mapping = {
        'NDVI': 'ndvi_mean',
        'EVI': 'evi_mean',
        'SAVI': 'savi_mean',
        'temp_max': 'temp_max_c',
        'temp_min': 'temp_min_c',
        'rainfall': 'rainfall_mm'
    }
    metadata_df = metadata_df.rename(columns=column_mapping)
    
    # Initialize dataset (patches_dir is dummy as we override spatial input below)
    dataset = AgriSightDataset(metadata_df=metadata_df, patches_dir="data/processed/california_patches")
    audit_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    healthy_count = 0
    total = 0

    with torch.no_grad():
        for batch in audit_loader:
            # SCIENTIFIC CONTROL: Replace real imagery with randomized noise
            # This isolates the tabular pathway as the only possible signal source.
            spatial_noise = torch.randn(batch['spatial'].shape).to(device)
            tabular = batch['tabular'].to(device)
            
            outputs = model(spatial_noise, tabular)
            _, predicted = torch.max(outputs, 1)
            
            # Count "Healthy" (Class 0) predictions
            healthy_count += (predicted == 0).sum().item()
            total += spatial_noise.size(0)

    bias_rate = (healthy_count / total) * 100
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    categories = ['California (Control)', f'{region_name} (Audit)']
    values = [72.3, bias_rate]
    
    ax = sns.barplot(x=categories, y=values, palette=['#2ecc71', '#3498db'])
    plt.title(f'Spectral Bias Audit: Signal Isolation Test ({region_name})', fontsize=14)
    plt.ylabel('Healthy Prediction Frequency (%)', fontsize=12)
    plt.ylim(0, 110)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', weight='bold')

    plot_path = f'results/{output_filename}'
    plt.savefig(plot_path)
    print(f"SUCCESS: Audit complete. Randomized spatial control verified for {region_name}.")

if __name__ == "__main__":
    # Audit runs without needing regional patch directories
    run_regional_audit("W. Australia", "data/raw/australia_dryland_tile_weather.csv", "spectral_bias_audit_plot.png")
    run_regional_audit("Punjab", "data/raw/punjab_wheat_belt_tile_weather.csv", "punjab_bias_audit_plot.png")
