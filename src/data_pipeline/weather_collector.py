import requests
import pandas as pd
from datetime import datetime

class EOMeteorologicalCollector:
    def __init__(self):
        """
        Initialize the Meteorological Data Collector for Global Audit.
        Utilizes Open-Meteo Historical Archive API (No API key required).
        """
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        
        # Coordinates for the three validation zones
        self.validation_zones = {
            'california_baseline': {'lat': 37.0, 'lon': -120.0},
            'punjab_scale': {'lat': 31.0, 'lon': 75.5},
            'australia_stress': {'lat': -31.5, 'lon': 116.5}
        }

    def get_historical_weather(self, region_name, start_date, end_date):
        """
        Fetch real historical weather data for the MLP pathway of the fusion model.
        """
        if region_name not in self.validation_zones:
            raise ValueError(f"Region {region_name} not found in validation zones.")
            
        coords = self.validation_zones[region_name]
        
        # API Parameters aligned with your PyTorch Tabular Tensor inputs
        params = {
            "latitude": coords['lat'],
            "longitude": coords['lon'],
            "start_date": start_date,
            "end_date": end_date,
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
            "timezone": "auto"
        }

        print(f"📡 Fetching historical climate data for {region_name}...")
        response = requests.get(self.base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Convert JSON response into a clean Pandas DataFrame
            df = pd.DataFrame({
                'date': pd.to_datetime(data['daily']['time']),
                'temp_max_c': data['daily']['temperature_2m_max'],
                'temp_min_c': data['daily']['temperature_2m_min'],
                'rainfall_mm': data['daily']['precipitation_sum'],
                'region': region_name
            })
            
            # Handle any missing data from the API
            df = df.fillna(method='ffill')
            return df
        else:
            raise ConnectionError(f"API Request failed with status code {response.status_code}")

    def engineer_fusion_features(self, weather_df):
        """
        Format the raw weather data into the scaled tabular tensor format
        expected by the PyTorch MultiModalCNN.
        """
        # Calculate derived features
        weather_df['temp_range'] = weather_df['temp_max_c'] - weather_df['temp_min_c']
        
        # In a real pipeline, you would apply Min-Max scaling here 
        # before concatenating with the Sentinel-2 data.
        
        return weather_df

# --- Pipeline Execution ---
if __name__ == "__main__":
    weather_collector = EOMeteorologicalCollector()
    
    # Example: Fetching the exact weather during the Australia Stress Test
    australia_weather = weather_collector.get_historical_weather(
        region_name='australia_stress',
        start_date='2025-01-01', # Australian Summer (High Albedo / High Temp)
        end_date='2025-03-31'
    )
    
    processed_weather = weather_collector.engineer_fusion_features(australia_weather)
    
    print("\n✅ Real Meteorological Data Collected Successfully:")
    print(processed_weather.head())