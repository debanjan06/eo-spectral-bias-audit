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

    # Replace deprecated fillna(method='ffill') with modern syntax
    # This handles intermittent sensor failures by propagating the last valid observation
    df = df.ffill()
    
    # Secondary check for leading nulls that ffill cannot reach
    if df.isnull().values.any():
        df = df.bfill()

    print("SUCCESS: Temporal gaps resolved via forward/backward propagation.")
    return df

def aggregate_regional_stats(df):
    """
    Calculates mean meteorological features used in the tabular pathway.
    """
    features = ['ndvi', 'temp_max', 'rainfall', 'humidity']
    available_features = [f for f in features if f in df.columns]
    
    summary = df[available_features].mean()
    print("LOG: Regional feature aggregation complete.")
    return summary

if __name__ == "__main__":
    # Example diagnostic run
    sample_data = "data/raw/punjab_wheat_belt_tile_weather.csv"
    processed_df = process_weather_data(sample_data)
    if processed_df is not None:
        print(processed_df.head())
