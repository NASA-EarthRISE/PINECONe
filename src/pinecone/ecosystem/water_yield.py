"""
Water yield calculation for ecosystem services valuation.
Calculates water yield from precipitation and evapotranspiration.
"""

import ee
from typing import Dict, Tuple, Optional
from datetime import datetime


class WaterYieldCalculator:
    """
    Calculate water yield for ecosystem services valuation.
    
    Water Yield = Precipitation - Evapotranspiration
    """
    
    # Constants
    M2_PER_ACRE = 4046.86
    KG_TO_KL = 0.001  # 1 kg water = 0.001 kL
    ET_SCALE_FACTOR = 0.1  # MODIS ET scaling factor
    WATER_PRICE_PER_KL = 0.018  # $/kL (default, can be changed)
    
    def __init__(
        self,
        water_price_per_kl: float = 0.018,
        et_scale_factor: float = 0.1
    ):
        """
        Initialize water yield calculator.
        
        Args:
            water_price_per_kl: Price per kiloliter of water ($/kL)
            et_scale_factor: Scaling factor for MODIS ET data
        """
        self.water_price = water_price_per_kl
        self.et_scale_factor = et_scale_factor
    
    def calculate_water_yield(
        self,
        aoi: ee.FeatureCollection,
        start_date: str,
        end_date: str,
        scale: int = 500
    ) -> Dict:
        """
        Calculate water yield for an area of interest.
        
        Args:
            aoi: Area of interest (FeatureCollection or Geometry)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            scale: Scale for reduction (meters)
            
        Returns:
            Dictionary with water yield statistics
        """
        # Get geometry
        if isinstance(aoi, ee.FeatureCollection):
            geometry = aoi.geometry()
        else:
            geometry = aoi
        
        # Load MODIS Evapotranspiration
        modis_et = (ee.ImageCollection('MODIS/NTSG/MOD16A2/105')
            .filterDate(start_date, end_date)
            .filterBounds(geometry)
            .select('ET'))
        
        # Load PRISM Precipitation
        prism = (ee.ImageCollection('OREGONSTATE/PRISM/ANm')
            .filterDate(start_date, end_date)
            .filterBounds(geometry)
            .select('ppt'))
        
        # Sum to get annual totals
        annual_et = modis_et.sum()
        annual_precip = prism.sum()
        
        # Calculate mean values for AOI
        et_stats = annual_et.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=scale,
            maxPixels=1e13
        )
        
        precip_stats = annual_precip.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=scale,
            maxPixels=1e13
        )
        
        # Check if we got valid data
        et_info = et_stats.getInfo()
        precip_info = precip_stats.getInfo()
        
        if 'ET' not in et_info or et_info['ET'] is None:
            raise ValueError("No ET data available for the specified date range and area")
        
        if 'ppt' not in precip_info or precip_info['ppt'] is None:
            raise ValueError("No precipitation data available for the specified date range and area")
        
        # Get values
        et_mean = ee.Number(et_info['ET'])
        precip_mean = ee.Number(precip_info['ppt'])
        
        # Calculate per-acre values ($/acre)
        precip_per_acre = (precip_mean
            .multiply(self.M2_PER_ACRE)
            .multiply(self.KG_TO_KL)
            .multiply(self.water_price))
        
        et_per_acre = (et_mean
            .multiply(self.M2_PER_ACRE)
            .multiply(self.et_scale_factor)  # Apply scaling factor
            .multiply(self.KG_TO_KL)
            .multiply(self.water_price))
        
        water_yield_per_acre = precip_per_acre.subtract(et_per_acre)
        
        # Calculate standard deviation
        per_pixel_precip = (annual_precip
            .multiply(self.M2_PER_ACRE)
            .multiply(self.KG_TO_KL)
            .multiply(self.water_price))
        
        per_pixel_et = (annual_et
            .multiply(self.M2_PER_ACRE)
            .multiply(self.et_scale_factor)
            .multiply(self.KG_TO_KL)
            .multiply(self.water_price))
        
        per_pixel_yield = per_pixel_precip.subtract(per_pixel_et)
        
        yield_std = per_pixel_yield.reduceRegion(
            reducer=ee.Reducer.stdDev(),
            geometry=geometry,
            scale=scale,
            maxPixels=1e13,
            bestEffort=True
        )
        
        # Calculate total values
        area_m2 = ee.Number(geometry.area(maxError=1))
        area_acres = area_m2.divide(self.M2_PER_ACRE)
        
        # Total precipitation and ET (kg/year)
        total_precip_kg = precip_mean.multiply(area_m2)
        total_et_kg = et_mean.multiply(area_m2).multiply(self.et_scale_factor)
        
        # Total water yield
        total_water_yield_kg = total_precip_kg.subtract(total_et_kg)
        total_water_yield_kl = total_water_yield_kg.multiply(self.KG_TO_KL)
        total_water_yield_cost = total_water_yield_kl.multiply(self.water_price)
        
        # Get info
        results = {
            'area_m2': area_m2.getInfo(),
            'area_acres': area_acres.getInfo(),
            'precipitation_per_acre_usd': precip_per_acre.getInfo(),
            'et_per_acre_usd': et_per_acre.getInfo(),
            'water_yield_per_acre_usd': water_yield_per_acre.getInfo(),
            'water_yield_std_per_acre_usd': ee.Number(yield_std.get('ppt')).getInfo() if yield_std.contains('ppt').getInfo() else 0,
            'total_precipitation_kg': total_precip_kg.getInfo(),
            'total_et_kg': total_et_kg.getInfo(),
            'total_water_yield_kg': total_water_yield_kg.getInfo(),
            'total_water_yield_kl': total_water_yield_kl.getInfo(),
            'total_water_yield_usd': total_water_yield_cost.getInfo(),
            'period': f"{start_date} to {end_date}"
        }
        
        return results
    
    def calculate_for_multiple_aois(
        self,
        aois: Dict[str, ee.FeatureCollection],
        start_date: str,
        end_date: str,
        scale: int = 500
    ) -> Dict[str, Dict]:
        """
        Calculate water yield for multiple AOIs.
        
        Args:
            aois: Dictionary of {name: FeatureCollection}
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            scale: Scale for reduction (meters)
            
        Returns:
            Dictionary of results for each AOI
        """
        results = {}
        
        for aoi_name, aoi_fc in aois.items():
            print(f"  Calculating water yield for {aoi_name}...")
            
            try:
                aoi_results = self.calculate_water_yield(
                    aoi=aoi_fc,
                    start_date=start_date,
                    end_date=end_date,
                    scale=scale
                )
                
                results[aoi_name] = aoi_results
                
                print(f"    ✓ Water yield: ${aoi_results['water_yield_per_acre_usd']:.2f}/acre")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                results[aoi_name] = None
        
        return results
    
    def export_timeseries(
        self,
        aoi: ee.FeatureCollection,
        start_date: str,
        end_date: str,
        output_type: str = 'et'  # 'et' or 'precip'
    ) -> ee.FeatureCollection:
        """
        Create time series of ET or precipitation for visualization.
        
        Args:
            aoi: Area of interest
            start_date: Start date
            end_date: End date
            output_type: Type of data ('et' or 'precip')
            
        Returns:
            FeatureCollection with time series data
        """
        geometry = aoi.geometry() if isinstance(aoi, ee.FeatureCollection) else aoi
        
        if output_type == 'et':
            collection = (ee.ImageCollection('MODIS/061/MOD16A2GF')
                .filterDate(start_date, end_date)
                .filterBounds(geometry)
                .select('ET'))
            band_name = 'ET'
        else:  # precipitation
            collection = (ee.ImageCollection('OREGONSTATE/PRISM/ANm')
                .filterDate(start_date, end_date)
                .filterBounds(geometry)
                .select('ppt'))
            band_name = 'ppt'
        
        def extract_value(image):
            stats = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=500
            )
            return ee.Feature(None, {
                'date': image.date().format(),
                'value': stats.get(band_name)
            })
        
        return collection.map(extract_value)


# Integration with TEV Calculator - helper function
def create_water_yield_params(
    water_yield_results: Dict,
    as_tuple: bool = True
) -> Dict:
    """
    Convert water yield results to TEV calculator parameters.
    
    Args:
        water_yield_results: Results from WaterYieldCalculator
        as_tuple: Return as (mean, std) tuples for Monte Carlo
        
    Returns:
        Dictionary ready for TEV calculator
    """
    mean_value = water_yield_results['water_yield_per_acre_usd']
    std_value = water_yield_results.get('water_yield_std_per_acre_usd', mean_value * 0.1)
    
    if as_tuple:
        return {
            'water_quality_value': (mean_value, std_value)
        }
    else:
        return {
            'water_quality_value': mean_value,
            'water_quality_std': std_value
        }