"""
Biomass statistics calculator for polygons.
"""

import ee
from typing import Optional


class BiomassStatsCalculator:
    """
    Calculate biomass statistics for polygons.
    """
    
    def __init__(self, biomass_data):
        """
        Initialize with biomass data.
        
        Args:
            biomass_data: BiomassData instance
        """
        self.biomass_data = biomass_data
        self.biomass_per_pixel = biomass_data.get_biomass_per_pixel(convert_to_tons=True)
    
    def calculate_stats(
        self,
        feature_collection: ee.FeatureCollection,
        zone_name: str,
        scale: int = 100
    ) -> ee.FeatureCollection:
        """
        Calculate biomass statistics for each polygon.
        
        Args:
            feature_collection: Polygons to analyze
            zone_name: Name for this zone/group
            scale: Scale in meters for reduction
            
        Returns:
            FeatureCollection with biomass statistics
        """
        # Reduce biomass to regions
        reduced = self.biomass_per_pixel.reduceRegions(
            collection=feature_collection,
            reducer=(ee.Reducer.sum()
                .combine(reducer2=ee.Reducer.mean(), sharedInputs=True)
                .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)),
            scale=scale
        )
        
        # Filter out empty polygons
        reduced = reduced.filter(ee.Filter.notNull(['sum']))
        
        # Add computed fields
        def add_fields(feature):
            geom = feature.geometry()
            area_m2 = ee.Algorithms.If(geom, ee.Number(geom.area(1)), 0)
            area_acres = ee.Number(area_m2).divide(4046.86)
            
            agb_sum = ee.Number(feature.get('sum'))
            agb_mean = ee.Number(feature.get('mean'))
            agb_std = ee.Number(feature.get('stdDev'))
            
            agb_per_acre = ee.Algorithms.If(
                area_acres.neq(0),
                agb_sum.divide(area_acres),
                None
            )
            agb_std_per_acre = ee.Algorithms.If(
                area_acres.neq(0),
                agb_std.divide(area_acres),
                None
            )
            
            return feature.set({
                'area_acres': area_acres,
                'AOI_Zone': zone_name,
                'AGB_Total_tons': agb_sum,
                'AGB_per_acre_tons': agb_per_acre,
                'AGB_Mean_tons_per_pixel': agb_mean,
                'AGB_StdDev_tons_per_pixel': agb_std,
                'AGB_StdDev_per_acre_tons': agb_std_per_acre
            })
        
        return reduced.map(add_fields)
    
    def calculate_zone_summary(
        self,
        stats_fc: ee.FeatureCollection
    ) -> ee.FeatureCollection:
        """
        Calculate summary statistics by zone.
        
        Args:
            stats_fc: FeatureCollection with individual polygon stats
            
        Returns:
            FeatureCollection with one feature per zone
        """
        zones = stats_fc.aggregate_array('AOI_Zone').distinct()
        
        def summarize_zone(zone_name):
            zone_name = ee.String(zone_name)
            fc_zone = stats_fc.filter(ee.Filter.eq('AOI_Zone', zone_name))
            
            mean_agb = fc_zone.aggregate_mean('AGB_per_acre_tons')
            mean_std = fc_zone.aggregate_mean('AGB_StdDev_per_acre_tons')
            total_area = fc_zone.aggregate_sum('area_acres')
            total_biomass = fc_zone.aggregate_sum('AGB_Total_tons')
            
            return ee.Feature(None, {
                'AOI_Zone': zone_name,
                'mean_AGB_per_acre_tons': mean_agb,
                'mean_StdDev_per_acre_tons': mean_std,
                'total_area_acres': total_area,
                'total_biomass_tons': total_biomass
            })
        
        return ee.FeatureCollection(zones.map(summarize_zone))