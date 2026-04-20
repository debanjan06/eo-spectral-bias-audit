import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import ee
import pandas as pd
import numpy as np
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
# --- UPDATE IN scripts/start_data_collection.py ---
# --- UPDATE IN scripts/start_data_collection.py ---
REGIONS = {
    'australia_dryland_tile': [118.2, -31.7, 118.7, -31.2], # Merredin, WA
}

TARGET_REGION_NAME = 'australia_dryland_tile'
TARGET_BOUNDS = REGIONS[TARGET_REGION_NAME]

# Note: In April, Australia is in the 'Pre-Sowing' or 'Early Emergence' phase.
# This is a perfect test for "False Positives."
START_DATE = '2026-04-01' 
END_DATE = '2026-04-20'
MAX_CLOUD_PRCT = 5

def initialize_earth_engine():
    try:
        ee.Initialize()
        print("Earth Engine initialized successfully")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def collect_satellite_data(bounds, region_name):
    print(f"\nCOLLECTING SATELLITE DATA FOR: {region_name.upper()}")
    region = ee.Geometry.Rectangle(bounds)
    
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(region)
                 .filterDate(START_DATE, END_DATE)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_PRCT))
                 .sort('system:time_start'))
    
    def calculate_indices(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        evi = image.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {'NIR': image.select('B8').divide(10000), 'RED': image.select('B4').divide(10000), 'BLUE': image.select('B2').divide(10000)}).rename('EVI')
        savi = image.expression('((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
            {'NIR': image.select('B8').divide(10000), 'RED': image.select('B4').divide(10000)}).rename('SAVI')
        return image.addBands([ndvi, evi, savi])
    
    return collection.map(calculate_indices), region

def start_satellite_export(collection, region, region_name):
    print("\nSTARTING SATELLITE DATA EXPORT")
    composite = collection.median()
    export_bands = ['B4', 'B3', 'B2', 'B8', 'NDVI', 'EVI', 'SAVI']
    
    # CRITICAL FIX: Cast to float32 for compatibility
    image_to_export = composite.select(export_bands).toFloat()
    
    task = ee.batch.Export.image.toDrive(
        image=image_to_export,
        description=f'agriguard_{region_name}_data',
        folder='AgriGuard_Data',
        region=region,
        scale=10, 
        crs='EPSG:4326',
        maxPixels=1e10,  # <-- Change from 1e9 to 1e10
        fileFormat='GeoTIFF'
    )
    task.start()
    print(f"Export task initiated! Task ID: {task.id}")
    return task

def generate_generalized_weather_data(region_name):
    """Restored function to generate regional weather csv"""
    print("\nGENERATING ALIGNED WEATHER DATA")
    dates = pd.date_range(START_DATE, END_DATE, freq='D')
    np.random.seed(42)
    weather_data = []
    
    for date in dates:
        temp_max = np.random.normal(32, 4)
        temp_min = np.random.normal(15, 3)
        humidity = np.random.normal(45, 10)
        rainfall = np.random.exponential(0.5) if np.random.random() < 0.05 else 0
        
        weather_data.append({
            'date': date,
            'temp_max': round(temp_max, 1),
            'temp_min': round(temp_min, 1),
            'humidity': round(humidity, 1),
            'rainfall': round(rainfall, 2),
            'disease_risk': round(np.random.random() * 0.4, 3)
        })
    
    weather_df = pd.DataFrame(weather_data)
    os.makedirs('data/raw', exist_ok=True)
    output_file = f'data/raw/{region_name}_weather.csv'
    weather_df.to_csv(output_file, index=False)
    print(f"Weather data saved to: {output_file}")
    return weather_df

def main():
    print("=" * 60)
    print("AGRIGUARD: DATA COLLECTION PIPELINE")
    print("=" * 60)
    
    if not initialize_earth_engine(): return
        
    try:
        # 1. Generate/Check Weather Data
        generate_generalized_weather_data(TARGET_REGION_NAME)

        # 2. TRIGGER SATELLITE COLLECTION
        collection, region = collect_satellite_data(TARGET_BOUNDS, TARGET_REGION_NAME)
        
        count = collection.size().getInfo()
        if count > 0:
            print(f"Found {count} pristine images for Punjab.")
            # 3. TRIGGER EXPORT TO DRIVE
            task = start_satellite_export(collection, region, TARGET_REGION_NAME)
            print(f"\n✅ SUCCESS: Check Google Drive for 'agriguard_{TARGET_REGION_NAME}_data.tif' in 15 mins.")
            print(f"Task ID: {task.id}")
        else:
            print("❌ No images found for these dates/cloud cover. Try expanding the date range.")
            
    except Exception as e:
        print(f"\nExecution Error: {e}")
if __name__ == "__main__":
    main()