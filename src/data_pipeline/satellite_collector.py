import ee
from datetime import datetime, timedelta

class EOSpectralDataCollector:
    def __init__(self):
        """Initialize Earth Engine for Global Spectral Bias Audit"""
        ee.Initialize()
        # Using the exact Sentinel-2 Surface Reflectance dataset from your thesis
        self.s2_collection = 'COPERNICUS/S2_SR_HARMONIZED'
        
    def define_study_areas(self):
        """Define the cross-continental validation regions"""
        study_areas = {
            # High-Biomass Baseline Training Area
            'california_baseline': ee.Geometry.Rectangle([-120.5, 36.5, -119.5, 37.5]),
            # 1.1 Billion Pixel Scalability Test Area
            'punjab_scale': ee.Geometry.Rectangle([75.0, 30.5, 76.0, 31.5]),
            # High-Albedo Arid Stress Test Area (Where the bias was found)
            'australia_stress': ee.Geometry.Rectangle([116.0, -32.0, 117.0, -31.0])
        }
        return study_areas
    
    def calculate_vegetation_indices(self, image):
        """Calculate the multi-modal spectral pathway features"""
        # NDVI - Normalized Difference Vegetation Index
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # EVI - Enhanced Vegetation Index
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {
                'NIR': image.select('B8').divide(10000),
                'RED': image.select('B4').divide(10000),
                'BLUE': image.select('B2').divide(10000)
            }
        ).rename('EVI')
        
        # SAVI - Soil Adjusted Vegetation Index (Crucial for the Australia audit)
        savi = image.expression(
            '((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
            {
                'NIR': image.select('B8').divide(10000),
                'RED': image.select('B4').divide(10000)
            }
        ).rename('SAVI')
        
        return image.addBands([ndvi, evi, savi])
    
    def collect_temporal_data(self, geometry, start_date, end_date):
        """Collect cloud-free monthly temporal composites"""
        
        collection = (ee.ImageCollection(self.s2_collection)
                     .filterBounds(geometry)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
                     .map(self.calculate_vegetation_indices))
        
        months = []
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        current = start
        while current < end:
            month_start = current.strftime('%Y-%m-%d')
            # Calculate end of month safely
            next_month = current.replace(day=28) + timedelta(days=4)
            month_end = (next_month - timedelta(days=next_month.day)).strftime('%Y-%m-%d')
            
            monthly_composite = (collection
                               .filterDate(month_start, month_end)
                               .median() # Median filtering removes cloud shadows/noise
                               .set('system:time_start', ee.Date(month_start).millis()))
            
            months.append(monthly_composite)
            current = next_month.replace(day=1)
        
        return ee.ImageCollection(months)
    
    def export_inference_tensors(self, region_name, geometry, start_date, end_date):
        """Export data for the Streamlit Docker deployment"""
        time_series = self.collect_temporal_data(geometry, start_date, end_date)
        
        task = ee.batch.Export.image.toDrive(
            image=time_series.toBands(),
            description=f'eo_audit_{region_name}_{start_date}_{end_date}',
            folder='AgriSight_Raw_Tiffs',
            region=geometry,
            scale=10,  # 10m Sentinel-2 resolution
            crs='EPSG:4326',
            maxPixels=1e10 # Expanded for the Punjab billion-pixel run
        )
        
        task.start()
        print(f"Export started to Google Drive for: {region_name}")
        return task

# Test the collector
if __name__ == "__main__":
    collector = EOSpectralDataCollector()
    study_areas = collector.define_study_areas()
    
    # Test with the California Baseline
    california = study_areas['california_baseline']
    
    # Collect a test composite
    time_series = collector.collect_temporal_data(
        california, 
        '2025-06-01', 
        '2025-08-01'
    )
    
    print(f"Validated pipeline. Generated {time_series.size().getInfo()} temporal composites.")