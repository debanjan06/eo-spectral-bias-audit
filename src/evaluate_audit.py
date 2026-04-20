import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_thesis_plot():
    print("📊 Generating official thesis visualization...")
    os.makedirs('results', exist_ok=True)
    
    # 1. The Exact Empirical Results from your V1.0 Audit
    audit_results = [
        {
            'Validation Zone': 'California (Baseline)',
            'Predicted Class': 'Healthy',
            'Model Confidence Score': 0.978,
            'Expected Reality': 'Healthy Crop'
        },
        {
            'Validation Zone': 'Punjab (Scale Test)',
            'Predicted Class': 'Healthy',
            'Model Confidence Score': 0.982,
            'Expected Reality': 'Healthy Crop'
        },
        {
            'Validation Zone': 'W. Australia (Stress Test)',
            'Predicted Class': 'Healthy', # THIS IS THE BIAS!
            'Model Confidence Score': 1.000,
            'Expected Reality': 'Bare Earth / Stubble'
        }
    ]
    
    df = pd.DataFrame(audit_results)
    
    # 2. Plot Configuration
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotting Confidence across regions
    sns.barplot(
        data=df, 
        x='Validation Zone', 
        y='Model Confidence Score', 
        hue='Predicted Class', 
        palette={'Healthy': '#2ecc71', 'Stressed/Diseased': '#e74c3c'}, 
        ax=ax
    )
    
    # 3. Typography and Labels
    plt.title('Multi-Modal Architecture: Cross-Continental Confidence Audit', fontsize=14, pad=20, fontweight='bold')
    plt.ylabel('Model Confidence Score', fontsize=12, fontweight='bold')
    plt.xlabel('Validation Zone', fontsize=12, fontweight='bold')
    plt.ylim(0, 1.15) # Giving room for the annotation
    
    # 4. The Smoking Gun Annotation
    plt.annotate(
        'SPECTRAL BIAS DETECTED\n(Bare Earth predicted as Healthy)', 
        xy=(2, 1.01), xytext=(1.4, 1.08),
        arrowprops=dict(facecolor='#e74c3c', shrink=0.05, width=3, headwidth=10),
        fontsize=11, color='#c0392b', fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#e74c3c", alpha=0.9, lw=1.5)
    )
                 
    plt.tight_layout()
    plt.savefig('results/spectral_bias_audit_plot.png', dpi=300)
    print("✅ Success! Check results/spectral_bias_audit_plot.png")

if __name__ == "__main__":
    generate_thesis_plot()