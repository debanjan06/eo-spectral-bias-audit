import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

def generate_comprehensive_training_data():
    """Generate comprehensive training data based on real satellite statistics"""
    print("GENERATING COMPREHENSIVE TRAINING DATASET")
    print("=" * 50)
    
    # Use your actual satellite statistics as baseline
    baseline_stats = {
        'NDVI_healthy': 0.493,
        'EVI_healthy': 0.337,
        'SAVI_healthy': 0.295,
        'REP_healthy': 717.616
    }
    
    print("Using Real Satellite Statistics as Baseline:")
    for key, value in baseline_stats.items():
        print(f"  {key}: {value:.3f}")
    
    # Generate comprehensive dataset
    np.random.seed(42)
    training_data = []
    
    # Create 1000 samples across different scenarios
    n_samples = 1000
    print(f"\nGenerating {n_samples} training samples...")
    
    for i in range(n_samples):
        # Random date
        start_date = datetime(2024, 1, 1)
        random_days = np.random.randint(0, 365)
        sample_date = start_date + timedelta(days=random_days)
        month = sample_date.month
        
        # Seasonal adjustments
        seasonal_factor = get_seasonal_factor(month)
        
        # Weather conditions
        temp_max, temp_min, humidity, rainfall = generate_weather_for_date(sample_date)
        
        # Determine health status and disease type
        disease_prob = calculate_disease_probability(temp_max, temp_min, humidity, rainfall, month)
        
        if disease_prob > 0.7:
            health_status = 'diseased'
            disease_type = select_disease_type(month, temp_max, humidity, rainfall)
            severity = np.random.uniform(0.6, 1.0)
        elif disease_prob > 0.4:
            health_status = 'stressed'
            disease_type = 'environmental_stress'
            severity = np.random.uniform(0.3, 0.6)
        else:
            health_status = 'healthy'
            disease_type = 'none'
            severity = 0.0
        
        # Generate vegetation indices based on health status
        ndvi, evi, savi, rep = generate_vegetation_indices(
            baseline_stats, health_status, disease_type, severity, seasonal_factor
        )
        
        # Additional spectral bands (simulate satellite bands)
        blue = np.random.normal(0.08, 0.02)
        green = np.random.normal(0.12, 0.03)
        red = np.random.normal(0.15, 0.04)
        nir = red * (1 + ndvi) / (1 - ndvi) if ndvi < 0.99 else red * 10
        
        # Ensure realistic ranges
        blue = max(0.01, min(0.3, blue))
        green = max(0.01, min(0.4, green))
        red = max(0.01, min(0.5, red))
        nir = max(0.01, min(0.8, nir))
        
        training_data.append({
            'sample_id': f'AGR_{i:04d}',
            'date': sample_date,
            'month': month,
            'season': get_season_name(month),
            'temp_max': temp_max,
            'temp_min': temp_min,
            'humidity': humidity,
            'rainfall': rainfall,
            'disease_probability': disease_prob,
            'health_status': health_status,
            'disease_type': disease_type,
            'severity': severity,
            'NDVI': ndvi,
            'EVI': evi,
            'SAVI': savi,
            'REP': rep,
            'blue_band': blue,
            'green_band': green,
            'red_band': red,
            'nir_band': nir,
            # Updated to California Central Valley bounds
            'lat': np.random.uniform(36.5, 37.5),  
            'lon': np.random.uniform(-120.5, -119.5)
        })
    
    df = pd.DataFrame(training_data)
    
    # Save comprehensive dataset
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/agriguard_training_dataset.csv', index=False)
    
    print(f"Comprehensive dataset saved: data/processed/agriguard_training_dataset.csv")
    
    # Dataset statistics
    print(f"\nDATASET STATISTICS:")
    print(f"  Total samples: {len(df)}")
    print(f"  Healthy: {len(df[df['health_status'] == 'healthy'])} ({len(df[df['health_status'] == 'healthy'])/len(df)*100:.1f}%)")
    print(f"  Stressed: {len(df[df['health_status'] == 'stressed'])} ({len(df[df['health_status'] == 'stressed'])/len(df)*100:.1f}%)")
    print(f"  Diseased: {len(df[df['health_status'] == 'diseased'])} ({len(df[df['health_status'] == 'diseased'])/len(df)*100:.1f}%)")
    
    print(f"\nDISEASE BREAKDOWN:")
    disease_counts = df['disease_type'].value_counts()
    for disease, count in disease_counts.items():
        print(f"  - {disease}: {count} samples")
    
    print(f"\nSEASONAL DISTRIBUTION:")
    seasonal_counts = df['season'].value_counts()
    for season, count in seasonal_counts.items():
        print(f"  - {season}: {count} samples")
    
    return df

def get_seasonal_factor(month):
    """Get seasonal adjustment factor for vegetation"""
    if month in [6, 7, 8, 9]:
        return 1.2
    elif month in [10, 11]:
        return 1.1
    elif month in [12, 1, 2]:
        return 0.9
    else:
        return 0.8

def generate_weather_for_date(date):
    """Generate realistic weather for given date"""
    month = date.month
    
    # Temperature patterns
    if month in [12, 1, 2]:
        temp_max = np.random.normal(25, 3)
        temp_min = np.random.normal(15, 3)
    elif month in [3, 4, 5]:
        temp_max = np.random.normal(35, 4)
        temp_min = np.random.normal(22, 3)
    elif month in [6, 7, 8, 9]:
        temp_max = np.random.normal(28, 3)
        temp_min = np.random.normal(20, 2)
    else:
        temp_max = np.random.normal(30, 3)
        temp_min = np.random.normal(18, 3)
    
    # Humidity patterns
    if month in [6, 7, 8, 9]:
        humidity = np.random.normal(85, 8)
    else:
        humidity = np.random.normal(65, 15)
    
    # Rainfall patterns
    if month in [6, 7, 8, 9]:
        rainfall = np.random.exponential(5) if np.random.random() < 0.4 else 0
    else:
        rainfall = np.random.exponential(1) if np.random.random() < 0.1 else 0
    
    return max(temp_min + 2, temp_max), temp_min, max(30, min(95, humidity)), rainfall

def calculate_disease_probability(temp_max, temp_min, humidity, rainfall, month):
    """Calculate disease probability based on weather"""
    prob = 0.0
    
    if humidity > 85: prob += 0.4
    elif humidity > 75: prob += 0.2
    
    if rainfall > 10: prob += 0.3
    elif rainfall > 2: prob += 0.1
    
    if temp_max > 38 or temp_min < 10: prob += 0.2
    
    if month in [7, 8, 9]: prob += 0.2
    
    if 25 < temp_max < 32 and humidity > 80: prob += 0.3
    
    return max(0, min(1, prob + np.random.normal(0, 0.1)))

def select_disease_type(month, temp_max, humidity, rainfall):
    """Select disease type based on conditions"""
    if month in [7, 8] and humidity > 80:
        return np.random.choice(['early_blight', 'leaf_spot'], p=[0.6, 0.4])
    elif month in [9, 10] and rainfall > 5:
        return 'late_blight'
    elif month in [4, 5] and temp_max > 32:
        return 'bacterial_wilt'
    elif humidity > 85:
        return 'fungal_infection'
    else:
        return 'viral_infection'

def generate_vegetation_indices(baseline_stats, health_status, disease_type, severity, seasonal_factor):
    """Generate vegetation indices based on plant health"""
    ndvi_base = baseline_stats['NDVI_healthy'] * seasonal_factor
    evi_base = baseline_stats['EVI_healthy'] * seasonal_factor
    savi_base = baseline_stats['SAVI_healthy'] * seasonal_factor
    rep_base = baseline_stats['REP_healthy']
    
    if health_status == 'healthy':
        ndvi = ndvi_base + np.random.normal(0, 0.05)
        evi = evi_base + np.random.normal(0, 0.03)
        savi = savi_base + np.random.normal(0, 0.04)
        rep = rep_base + np.random.normal(0, 5)
        
    elif health_status == 'stressed':
        ndvi = ndvi_base * (1 - severity * 0.3) + np.random.normal(0, 0.03)
        evi = evi_base * (1 - severity * 0.25) + np.random.normal(0, 0.02)
        savi = savi_base * (1 - severity * 0.2) + np.random.normal(0, 0.03)
        rep = rep_base - severity * 20 + np.random.normal(0, 3)
        
    else:
        disease_effects = {
            'early_blight': {'ndvi': 0.4, 'evi': 0.35, 'savi': 0.3, 'rep': 25},
            'late_blight': {'ndvi': 0.5, 'evi': 0.45, 'savi': 0.4, 'rep': 30},
            'bacterial_wilt': {'ndvi': 0.6, 'evi': 0.5, 'savi': 0.45, 'rep': 35},
            'leaf_spot': {'ndvi': 0.3, 'evi': 0.25, 'savi': 0.2, 'rep': 20},
            'fungal_infection': {'ndvi': 0.35, 'evi': 0.3, 'savi': 0.25, 'rep': 22},
            'viral_infection': {'ndvi': 0.25, 'evi': 0.2, 'savi': 0.15, 'rep': 18}
        }
        
        effects = disease_effects.get(disease_type, disease_effects['fungal_infection'])
        
        ndvi = ndvi_base * (1 - severity * effects['ndvi']) + np.random.normal(0, 0.02)
        evi = evi_base * (1 - severity * effects['evi']) + np.random.normal(0, 0.015)
        savi = savi_base * (1 - severity * effects['savi']) + np.random.normal(0, 0.02)
        rep = rep_base - severity * effects['rep'] + np.random.normal(0, 2)
    
    ndvi = max(-1, min(1, ndvi))
    evi = max(-1, min(1, evi))
    savi = max(-1, min(1, savi))
    rep = max(680, min(750, rep))
    
    return round(ndvi, 4), round(evi, 4), round(savi, 4), round(rep, 2)

def get_season_name(month):
    """Get season name"""
    if month in [12, 1, 2]: return "Winter"
    elif month in [3, 4, 5]: return "Summer"
    elif month in [6, 7, 8, 9]: return "Monsoon"
    else: return "Post-Monsoon"

def create_visualization(df):
    """Create comprehensive visualizations"""
    print("\nCREATING VISUALIZATIONS...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    health_counts = df['health_status'].value_counts()
    colors = ['lightgreen', 'gold', 'lightcoral']
    axes[0,0].pie(health_counts.values, labels=health_counts.index, autopct='%1.1f%%', colors=colors)
    axes[0,0].set_title('Health Status Distribution')
    
    sns.boxplot(data=df, x='health_status', y='NDVI', ax=axes[0,1])
    axes[0,1].set_title('NDVI by Health Status')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    seasonal_disease = pd.crosstab(df['season'], df['health_status'])
    seasonal_disease.plot(kind='bar', stacked=True, ax=axes[0,2], color=colors)
    axes[0,2].set_title('Seasonal Disease Patterns')
    axes[0,2].tick_params(axis='x', rotation=45)
    
    axes[1,0].scatter(df['humidity'], df['disease_probability'], alpha=0.6, c=df['disease_probability'], cmap='Reds')
    axes[1,0].set_xlabel('Humidity (%)')
    axes[1,0].set_ylabel('Disease Probability')
    axes[1,0].set_title('Humidity vs Disease Risk')
    
    indices_data = df[['NDVI', 'EVI', 'SAVI']].corr()
    sns.heatmap(indices_data, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
    axes[1,1].set_title('Vegetation Indices Correlation')
    
    diseased_data = df[df['health_status'] == 'diseased']
    if len(diseased_data) > 0:
        axes[1,2].hist(diseased_data['severity'], bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[1,2].set_xlabel('Disease Severity')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].set_title('Disease Severity Distribution')
    
    plt.tight_layout()
    plt.savefig('data/processed/dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved: data/processed/dataset_analysis.png")

def main():
    """Generate comprehensive training dataset"""
    print("AGRIGUARD TRAINING DATA GENERATION")
    print("=" * 60)
    
    df = generate_comprehensive_training_data()
    create_visualization(df)
    
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TRAINING DATASET READY!")
    print("=" * 60)
    
    return df

if __name__ == "__main__":
    df = main()