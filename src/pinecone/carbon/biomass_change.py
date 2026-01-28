"""
Biomass change detection and carbon emissions calculation.
Handles pre/post fire biomass comparison with quality control.
"""

import ee
from typing import Optional, Dict, Tuple
from datetime import datetime


class BiomassChangeCalculator:
    """
    Calculate biomass change and carbon emissions between two time periods.
    
    Uses ESA-CCI biomass change products with quality flags.
    """
    
    def __init__(
        self,
        biomass_data,  # BiomassData instance
        focal_species=None,  # Optional FocalSpeciesLayer
        carbon_fraction: float = 0.51,
        credit_price_per_ton: float = 1.0
    ):
        """
        Initialize biomass change calculator.
        
        Args:
            biomass_data: BiomassData instance
            focal_species: Optional FocalSpeciesLayer to clip to
            carbon_fraction: Carbon fraction of biomass (default 0.51)
            credit_price_per_ton: Price per ton CO2e in USD
        """
        self.biomass_data = biomass_data
        self.focal_species = focal_species
        self.carbon_fraction = carbon_fraction
        self.credit_price_per_ton = credit_price_per_ton
        
        # Results storage
        self.agb_pre = None
        self.agb_post = None
        self.agb_diff = None
        self.co2_diff = None
        self.quality_flag = None
        self.quality_mask = None
        self.change_sd = None
    
    def calculate_change(
        self,
        pre_date_start: str,
        pre_date_end: str,
        post_date_start: str,
        post_date_end: str,
        aoi: ee.FeatureCollection,
        resolution: int = 100,
        apply_quality_filter: bool = True
    ) -> Dict:
        """
        Calculate biomass change between two time periods.
        
        Args:
            pre_date_start: Pre-fire start date (YYYY-MM-DD)
            pre_date_end: Pre-fire end date (YYYY-MM-DD)
            post_date_start: Post-fire start date (YYYY-MM-DD)
            post_date_end: Post-fire end date (YYYY-MM-DD)
            aoi: Area of interest (FeatureCollection)
            resolution: Resolution in meters for calculations
            apply_quality_filter: Whether to apply quality flag filtering
            
        Returns:
            Dictionary with change statistics
        """
        # Get geometry
        if isinstance(aoi, ee.FeatureCollection):
            geometry = aoi.geometry()
        else:
            geometry = aoi
        
        # Clip to focal species if provided
        if self.focal_species:
            clip_geom = self.focal_species.vectorize(aoi).geometry()
        else:
            clip_geom = geometry
        
        # Load biomass data for both periods
        agb_collection = ee.ImageCollection('projects/sat-io/open-datasets/ESA/ESA_CCI_AGB').select(['AGB'])
        
        # Get pre-fire biomass
        agb_pre_ha = (agb_collection
            .filterDate(pre_date_start, pre_date_end)
            .first()
            .reproject(crs='EPSG:32616', scale=resolution)
            .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=10000, bestEffort=True))
        
        # Get post-fire biomass
        agb_post_ha = (agb_collection
            .filterDate(post_date_start, post_date_end)
            .first()
            .reproject(crs='EPSG:32616', scale=resolution)
            .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=10000, bestEffort=True))
        
        # Convert from tons/ha to tons/acre
        self.agb_pre = agb_pre_ha.divide(2.47105)
        self.agb_post = agb_post_ha.divide(2.47105)
        
        # Load quality flag and standard deviation
        start_year = datetime.strptime(pre_date_start, '%Y-%m-%d').year
        
        if apply_quality_filter:
            self._load_quality_data(start_year, resolution)
        
        # Calculate change statistics
        results = self._calculate_emissions_stats(clip_geom, resolution)
        
        # Add metadata
        results['pre_period'] = f"{pre_date_start} to {pre_date_end}"
        results['post_period'] = f"{post_date_start} to {post_date_end}"
        results['carbon_fraction'] = self.carbon_fraction
        results['credit_price_per_ton'] = self.credit_price_per_ton
        results['quality_filter_applied'] = apply_quality_filter
        
        return results
    
    def _load_quality_data(self, start_year: int, resolution: int):
        """Load quality flag and standard deviation data."""
        
        # Load Quality Flag
        agb_change_qf_collection = ee.ImageCollection(
            "projects/sat-io/open-datasets/ESA/ESA_CCI_AGB_DIFF"
        ).select(['AGB_DIFF_QF'])
        
        self.quality_flag = (agb_change_qf_collection
            .filter(ee.Filter.eq('start_year', ee.String(str(start_year))))
            .first()
            .reproject(crs='EPSG:32616', scale=resolution)
            .reduceResolution(reducer=ee.Reducer.mode(), maxPixels=10000, bestEffort=True))
        
        # Create mask: exclude pixels with value 3 (improbable change)
        self.quality_mask = self.quality_flag.neq(3)
        
        # Load Standard Deviation
        agb_change_sd_collection = ee.ImageCollection(
            "projects/sat-io/open-datasets/ESA/ESA_CCI_AGB_DIFF"
        ).select(['AGB_DIFF_SD'])
        
        self.change_sd = (agb_change_sd_collection
            .filter(ee.Filter.eq('start_year', ee.String(str(start_year))))
            .first()
            .reproject(crs='EPSG:32616', scale=resolution)
            .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=10000, bestEffort=True))
    
    def _calculate_emissions_stats(
        self,
        geometry: ee.Geometry,
        resolution: int
    ) -> Dict:
        """Calculate emissions statistics."""
        
        # Calculate AGB difference (tons per acre)
        pixel_area_acres = ee.Image.pixelArea().divide(4046.86)
        
        agb_pre_total = self.agb_pre.multiply(pixel_area_acres)
        agb_post_total = self.agb_post.multiply(pixel_area_acres)
        
        self.agb_diff = agb_post_total.subtract(agb_pre_total)
        
        # Calculate CO2 difference
        # CO2 = AGB * CF * (44/12)  [converting C to CO2]
        self.co2_diff = self.agb_diff.multiply(self.carbon_fraction).multiply(44).divide(12)
        
        # Apply quality mask if available
        if self.quality_mask is not None:
            co2_masked = self.co2_diff.updateMask(self.quality_mask)
        else:
            co2_masked = self.co2_diff
        
        # Calculate mean CO2 change
        co2_mean = ee.Number(
            co2_masked.reduceRegion(
                geometry=geometry,
                reducer=ee.Reducer.mean(),
                scale=resolution,
                maxPixels=1e18
            ).get('AGB')
        )
        
        # Calculate standard deviation if available
        if self.change_sd is not None:
            # Convert AGB SD to CO2 SD
            co2_sd_raster = self.change_sd.multiply(self.carbon_fraction).multiply(44).divide(12)
            
            # Calculate combined SD: sqrt(mean(variance))
            co2_variance = co2_sd_raster.pow(2)
            
            if self.quality_mask is not None:
                co2_variance = co2_variance.updateMask(self.quality_mask)
            
            co2_std = ee.Number(
                co2_variance.reduceRegion(
                    geometry=geometry,
                    reducer=ee.Reducer.mean(),
                    scale=resolution,
                    maxPixels=1e18
                ).get('AGB_DIFF_SD')
            ).sqrt()
        else:
            co2_std = None
        
        # Calculate carbon credits
        credit_mean_per_acre = co2_mean.multiply(self.credit_price_per_ton)
        credit_std_per_acre = co2_std.multiply(self.credit_price_per_ton) if co2_std else None
        
        # Calculate total area (with error margin)
        area_acres = ee.Number(geometry.area(maxError=1)).divide(4046.86)
        
        # Calculate totals
        co2_total = co2_mean.multiply(area_acres)
        credit_total = credit_mean_per_acre.multiply(area_acres)
        
        results = {
            'area_acres': area_acres.getInfo(),
            'CO2_mean_tons_per_acre': co2_mean.getInfo(),
            'CO2_total_tons': co2_total.getInfo(),
            'credit_mean_usd_per_acre': credit_mean_per_acre.getInfo(),
            'credit_total_usd': credit_total.getInfo(),
        }
        
        if co2_std is not None:
            results['CO2_std_tons_per_acre'] = co2_std.getInfo()
            results['credit_std_usd_per_acre'] = credit_std_per_acre.getInfo()
        
        return results
    
    def calculate_for_multiple_aois(
        self,
        aois: Dict[str, ee.FeatureCollection],
        pre_date_start: str,
        pre_date_end: str,
        post_date_start: str,
        post_date_end: str,
        resolution: int = 100,
        apply_quality_filter: bool = True
    ) -> ee.FeatureCollection:
        """
        Calculate emissions for multiple AOIs.
        
        Args:
            aois: Dictionary of {name: FeatureCollection}
            pre_date_start: Pre-fire start date
            pre_date_end: Pre-fire end date
            post_date_start: Post-fire start date
            post_date_end: Post-fire end date
            resolution: Resolution in meters
            apply_quality_filter: Apply quality filtering
            
        Returns:
            FeatureCollection with results for all AOIs
        """
        features = []
        
        for aoi_name, aoi_fc in aois.items():
            print(f"   Processing {aoi_name}...")
            
            # Calculate change for this AOI
            results = self.calculate_change(
                pre_date_start=pre_date_start,
                pre_date_end=pre_date_end,
                post_date_start=post_date_start,
                post_date_end=post_date_end,
                aoi=aoi_fc,
                resolution=resolution,
                apply_quality_filter=apply_quality_filter
            )
            
            # Create feature
            properties = {
                'AOI_Zone': aoi_name,
                **results
            }
            
            feature = ee.Feature(None, properties)
            features.append(feature)
            
            print(f"      ✓ {aoi_name}: {results['CO2_total_tons']:.2f} tons CO2")
        
        return ee.FeatureCollection(features)
    
    def export_results(
        self,
        results_fc: ee.FeatureCollection,
        description: str = 'Carbon_Emissions_Stats'
    ):
        """
        Export results to Google Drive.
        
        Args:
            results_fc: FeatureCollection with results
            description: Export task description
        """
        task = ee.batch.Export.table.toDrive(
            collection=results_fc,
            description=description,
            fileFormat='CSV'
        )
        task.start()
        print(f"Export task started: {description}")
        return task