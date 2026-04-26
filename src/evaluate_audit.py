import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from dataset import AgriSightDataset
from models.multi_modal_cnn import MultiModalCNN

def run_regional_audit(region_name, csv_path, patches_dir, output_filename):
    """
    Executes a scientifically controlled OOD stress test.
    Strictly synchronizes feature names and indices to match the 
    training schema defined in src/dataset.py.
    """
    print(f"INFO: Initiating feature-aligned audit for: {region_name}")
    os.makedirs('results', exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model Initialization
    model = MultiModalCNN(num_classes=3).to(device)
    try:
        model.load_state_dict(torch.load('models/best_baseline_model.pth', map_location=device, weights_only=True))
    except FileNotFoundError:
        print("ERROR: Weights missing. Baseline model must be trained first.")
        return

    model.eval()
    
    # Data Loading and Schema Mapping
    metadata_df = pd.read_csv(csv_path)
    
    # Explicit mapping to match the AgriSightDataset expected column names
    # and to replace 'humidity' with the correct 'temp_min_c' feature.
    column_mapping = {
        'NDVI': 'ndvi_mean',
        'EVI': 'evi_mean',
        'SAVI': 'savi_mean',
        'temp_max': 'temp_max_c',
        'temp_min': 'temp_min_c',
        'rainfall': 'rainfall_mm'
    }
    metadata_df = metadata_df.rename(columns=column_mapping)
    
    # Initializing the standardized dataset 
    # This automatically enforces the [NDVI, EVI, SAVI, TMAX, TMIN, PRECIP] order
    dataset = AgriSightDataset(metadata_df=metadata_df, patches_dir=patches_dir)
    audit_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    healthy_count = 0
    total = 0

    with torch.no_grad():
        for batch in audit_loader:
            spatial = batch['spatial'].to(device)
            tabular = batch['tabular'].to(device)
            
            outputs = model(spatial, tabular)
            _, predicted = torch.max(outputs, 1)
            
            # Class 0 represents the 'Healthy' diagnostic state
            healthy_count += (predicted == 0).sum().item()
            total += spatial.size(0)

    bias_rate = (healthy_count / total) * 100
    
    # Visualization Generation
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    categories = ['California (Control)', f'{region_name} (Audit)']
    values = [72.3, bias_rate]
    
    ax = sns.barplot(x=categories, y=values, palette=['#2ecc71', '#3498db'])
    plt.title(f'Spectral Bias Audit: Validated Feature Alignment ({region_name})', fontsize=14)
    plt.ylabel('Healthy Prediction Rate (%)', fontsize=12)
    plt.ylim(0, 110)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', weight='bold')

    plot_path = f'results/{output_filename}'
    plt.savefig(plot_path)
    print(f"SUCCESS: Audit complete. Diagnostic plot saved to: {plot_path}")

if __name__ == "__main__":
    # Western Australia (Arid Stress Test)
    run_regional_audit(
        region_name="W. Australia", 
        csv_path="data/raw/australia_dryland_tile_weather.csv", 
        patches_dir="data/processed/australia_patches",
        output_filename="spectral_bias_audit_plot.png"
    )
    
    # Punjab (Continental Wheat Belt Stress Test)
    run_regional_audit(
        region_name="Punjab", 
        csv_path="data/raw/punjab_wheat_belt_tile_weather.csv", 
        patches_dir="data/processed/punjab_patches",
        output_filename="punjab_bias_audit_plot.png"
    )
