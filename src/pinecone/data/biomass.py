"""
Biomass data handler with dynamic source selection.
"""

import ee
from typing import Optional, Dict, List
from datetime import datetime


class BiomassData:
    """
    Handler for multiple biomass data sources.
    
    Supports:
    - ESA-CCI AGB (Above Ground Biomass)
    - GEDI L4B
    - WHRC
    - Custom user layers
    """
    
    # Available biomass products
    PRODUCTS = {
        'esa_cci_agb': {
            'collection': 'projects/sat-io/open-datasets/ESA/ESA_CCI_AGB',
            'band': 'AGB',
            'units': 'Mg/ha',
            'description': 'ESA CCI Above Ground Biomass'
        },
        'gedi_l4b': {
            'collection': 'LARSE/GEDI/GEDI04_B_002',
            'band': 'MU',
            'units': 'Mg/ha',
            'description': 'GEDI L4B Aboveground Biomass'
        },
        'whrc': {
            'collection': 'WHRC/biomass/tropical',
            'band': 'b1',
            'units': 'Mg/ha',
            'description': 'Woods Hole Research Center Tropical Biomass'
        }
    }
    
    def __init__(
        self,
        product: str = 'esa_cci_agb',
        year: Optional[int] = None,
        custom_image: Optional[ee.Image] = None,
        custom_band: Optional[str] = None
    ):
        """
        Initialize biomass data source.
        
        Args:
            product: Biomass product name ('esa_cci_agb', 'gedi_l4b', 'whrc', 'custom')
            year: Year to select (if applicable)
            custom_image: Custom biomass image (if product='custom')
            custom_band: Band name for custom image
        """
        self.product = product
        self.year = year or 2019  # Default to 2019
        self.custom_image = custom_image
        self.custom_band = custom_band
        
        self.image = None
        self.band_name = None
        self.units = None
        
        self._load_biomass()
    
    def _load_biomass(self):
        """Load the biomass image based on selected product."""
        
        if self.product == 'custom':
            if self.custom_image is None:
                raise ValueError("custom_image required when product='custom'")
            self.image = self.custom_image
            self.band_name = self.custom_band or self.custom_image.bandNames().get(0).getInfo()
            self.units = 'Mg/ha'
            
        elif self.product not in self.PRODUCTS:
            raise ValueError(
                f"Unknown product: {self.product}. "
                f"Available: {list(self.PRODUCTS.keys())}"
            )
        else:
            product_info = self.PRODUCTS[self.product]
            
            if self.product == 'esa_cci_agb':
                # ESA-CCI AGB by year
                collection = ee.ImageCollection(product_info['collection'])
                start_date = f'{self.year}-01-01'
                end_date = f'{self.year}-12-31'
                
                self.image = (collection
                    .filterDate(start_date, end_date)
                    .first()
                    .select([product_info['band']]))
                
            elif self.product == 'gedi_l4b':
                # GEDI L4B
                collection = ee.ImageCollection(product_info['collection'])
                self.image = collection.select([product_info['band']]).mosaic()
                
            elif self.product == 'whrc':
                # WHRC single image
                self.image = ee.Image(product_info['collection']).select([product_info['band']])
            
            self.band_name = product_info['band']
            self.units = product_info['units']
    
    def get_biomass_per_pixel(self, convert_to_tons: bool = True) -> ee.Image:
        """
        Get biomass in tons per pixel.
        
        Args:
            convert_to_tons: Convert Mg/ha to US tons per pixel
            
        Returns:
            Biomass image in tons per pixel
        """
        # Pixel area in hectares
        pixel_area_ha = ee.Image.pixelArea().divide(10000)
        
        # Biomass in Mg per pixel
        biomass_per_pixel = self.image.multiply(pixel_area_ha)
        
        if convert_to_tons:
            # Convert Mg to US tons (1 Mg = 1.10231 US tons)
            biomass_per_pixel = biomass_per_pixel.multiply(1.10231)
        
        return biomass_per_pixel.rename('biomass_tons_per_pixel')
    
    @classmethod
    def list_available_products(cls) -> List[str]:
        """List all available biomass products."""
        return list(cls.PRODUCTS.keys()) + ['custom']