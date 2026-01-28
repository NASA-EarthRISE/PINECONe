"""
Focal species layer handler with dynamic selection.
"""

import ee
from typing import Union


class FocalSpeciesLayer:
    """
    Handler for focal species layers with dynamic selection.
    """
    
    def __init__(
        self,
        layer_source: Union[str, ee.Image],
        binary_threshold: float = 0,
        name: str = 'Focal Species'
    ):
        """
        Initialize focal species layer.
        
        Args:
            layer_source: EE asset path or ee.Image object
            binary_threshold: Threshold for binary mask (values > threshold = 1)
            name: Display name for layer
        """
        self.layer_source = layer_source
        self.binary_threshold = binary_threshold
        self.name = name
        
        self.layer = None
        self.binary_mask = None
        
        self._load_layer()
    
    def _load_layer(self):
        """Load the focal species layer."""
        
        if isinstance(self.layer_source, ee.Image):
            self.layer = self.layer_source
        elif isinstance(self.layer_source, str):
            # Load as EE asset
            try:
                self.layer = ee.Image(self.layer_source)
            except:
                raise ValueError(f"Could not load layer from: {self.layer_source}")
        else:
            raise ValueError(f"Unsupported layer_source type: {type(self.layer_source)}")
        
        # Create binary mask (all values > threshold -> 1, background -> 0)
        self.binary_mask = self.layer.gt(self.binary_threshold).selfMask()
    
    def clip_to_aoi(self, aoi: ee.Geometry) -> ee.Image:
        """Clip focal species layer to AOI."""
        return self.binary_mask.clip(aoi)
    
    def vectorize(
        self,
        aoi: Union[ee.Geometry, ee.FeatureCollection],
        scale: int = 30,
        max_pixels: int = 1e13
    ) -> ee.FeatureCollection:
        """
        Vectorize the focal species layer within AOI.
        
        Args:
            aoi: Area of interest (geometry or feature collection)
            scale: Scale in meters
            max_pixels: Maximum pixels to process
            
        Returns:
            FeatureCollection of focal species polygons
        """
        # Get geometry
        if isinstance(aoi, ee.FeatureCollection):
            geometry = aoi.geometry()
        else:
            geometry = aoi
            
        clipped = self.clip_to_aoi(geometry)
        
        vectors = clipped.reduceToVectors(
            geometry=geometry,
            scale=scale,
            eightConnected=True,
            geometryType='polygon',
            bestEffort=True,
            maxPixels=max_pixels
        )
        
        return vectors
    
    def get_non_species_areas(
        self,
        aoi: ee.FeatureCollection,
        dissolved_species_geom: ee.Geometry
    ) -> ee.FeatureCollection:
        """
        Get areas within AOI that are NOT focal species.
        
        Args:
            aoi: Area of interest FeatureCollection
            dissolved_species_geom: Dissolved geometry of species polygons
            
        Returns:
            FeatureCollection of non-species areas
        """
        def subtract_species(feature):
            # Use geometry().difference() instead of feature.difference()
            return ee.Feature(
                feature.geometry().difference(dissolved_species_geom, 1),
                feature.toDictionary()
            )
        
        non_species = aoi.map(subtract_species)
        return non_species