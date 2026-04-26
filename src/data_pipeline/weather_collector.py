import pandas as pd
import numpy as np

def process_weather_data(file_path):
    """
    Standardizes regional meteorological time-series data.
    Implements forward-fill to handle missing observations in weather station logs.
    """
    print(f"INFO: Loading weather data from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
        return None

    # Resolve temporal gaps via forward propagation
    df = df.ffill()
    
    # Secondary check for leading nulls
    if df.isnull().values.any():
        df = df.bfill()

    print("SUCCESS: Data cleaning and gap resolution complete.")
    return df

def aggregate_regional_stats(df):
    """
    Calculates mean meteorological features for tabular pathway injection.
    Synchronized with the model feature vector: [NDVI, EVI, SAVI, TMAX, TMIN, PRECIP].
    """
    # Aligned with the latest AgriSight training schema
    features = [
        'ndvi_mean', 
        'evi_mean', 
        'savi_mean', 
        'temp_max_c', 
        'temp_min_c', 
        'rainfall_mm'
    ]
    
    available_features = [f for f in features if f in df.columns]
    
    if not available_features:
        print("ERROR: No matching features found in the provided DataFrame.")
        return None
        
    summary = df[available_features].mean()
    print("LOG: Regional feature aggregation complete.")
    return summary

if __name__ == "__main__":
    # Example diagnostic execution
    sample_data = "data/raw/punjab_wheat_belt_tile_weather.csv"
    processed_df = process_weather_data(sample_data)
    if processed_df is not None:
        stats = aggregate_regional_stats(processed_df)
        if stats is not None:
            print(stats)
