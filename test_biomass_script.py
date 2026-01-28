"""
Test script to run the converted biomass analysis.
This replicates your original JavaScript workflow.
"""

import ee

# Import our new modules
from src.pinecone.data.biomass import BiomassData
from src.pinecone.data.focal_species import FocalSpeciesLayer
from src.pinecone.carbon.biomass_stats import BiomassStatsCalculator

# Initialize Earth Engine
ee.Initialize()

print("="*60)
print("PINECONe Biomass Analysis - Script 1")
print("="*60)

# ---------------------
# DYNAMIC INPUTS
# ---------------------

# 1. DYNAMIC: Focal Species Layer
print("\n1. Loading focal species layer...")
focal_species = FocalSpeciesLayer(
    layer_source="projects/servir-sco-assets/assets/Rx_Fire/EO_Inputs/LEO_extantLLP_significance",
    binary_threshold=0,
    name="Longleaf Pine (LLP)"
)
print(f"   ✓ Loaded: {focal_species.name}")

# 2. DYNAMIC: AOIs
print("\n2. Loading AOIs...")
EIA_CS1 = ee.FeatureCollection('projects/servir-sco-assets/assets/Rx_Fire/Vector_Data/EIA_CS1')
EIA_CS2 = ee.FeatureCollection('projects/servir-sco-assets/assets/Rx_Fire/Vector_Data/EIA_CS2')
EIA_CS3 = ee.FeatureCollection('projects/servir-sco-assets/assets/Rx_Fire/Vector_Data/EIA_CS3')
aois = {
    'EIA_CS1': EIA_CS1,
    'EIA_CS2': EIA_CS2,
    'EIA_CS3': EIA_CS3
}
print(f"   ✓ Loaded {len(aois)} AOIs")

# 3. DYNAMIC: Biomass Product
print("\n3. Loading biomass data...")
biomass = BiomassData(
    product='esa_cci_agb',  # CHANGEABLE: 'esa_cci_agb', 'gedi_l4b', 'whrc'
    year=2019               # CHANGEABLE: any year
)
print(f"   ✓ Loaded: {biomass.product} ({biomass.year})")

# 4. Create calculator
calculator = BiomassStatsCalculator(biomass)

# ---------------------
# PROCESS EACH AOI
# ---------------------

print("\n4. Processing AOIs...")
all_stats_list = []

for aoi_name, aoi_fc in aois.items():
    print(f"\n   Processing {aoi_name}...")
    
    # Vectorize focal species in this AOI
    print(f"      - Vectorizing {focal_species.name}...")
    species_vectors = focal_species.vectorize(aoi_fc, scale=30)
    
    # Dissolve species polygons
    species_dissolved = species_vectors.union(1)
    
    # Get non-species areas
    print(f"      - Calculating non-{focal_species.name} areas...")
    non_species = focal_species.get_non_species_areas(
        aoi_fc,
        species_dissolved.geometry()
    )
    
    # Calculate stats for species areas
    print(f"      - Calculating biomass stats for {focal_species.name}...")
    stats_species = calculator.calculate_stats(
        species_vectors,
        f"{aoi_name}_LLP"
    )
    
    # Calculate stats for non-species areas
    print(f"      - Calculating biomass stats for non-{focal_species.name}...")
    stats_non_species = calculator.calculate_stats(
        non_species,
        f"{aoi_name}_NonLLP"
    )
    
    # Merge
    aoi_stats = stats_species.merge(stats_non_species)
    all_stats_list.append(aoi_stats)
    print(f"      ✓ {aoi_name} complete")

# Combine all zones
print("\n5. Merging all results...")
all_stats = all_stats_list[0]
for stats in all_stats_list[1:]:
    all_stats = all_stats.merge(stats)

print("   ✓ All zones merged")

# Calculate zone summaries
print("\n6. Calculating zone summaries...")
zone_summary = calculator.calculate_zone_summary(all_stats)
print("   ✓ Zone summaries calculated")

# Print first few results
print("\n" + "="*60)
print("RESULTS PREVIEW")
print("="*60)
print("\nDetailed Stats (first 5 features):")
print(all_stats.limit(5).getInfo())

print("\nZone Summary:")
print(zone_summary.getInfo())

# ---------------------
# EXPORT (optional)
# ---------------------
print("\n7. Ready to export...")
print("\nTo export to Google Drive, uncomment these lines:")
print("""
# Export detailed stats
# task1 = ee.batch.Export.table.toDrive(
#     collection=all_stats,
#     description='Biomass_Detailed_Stats',
#     fileFormat='CSV'
# )
# task1.start()

# Export zone summary
# task2 = ee.batch.Export.table.toDrive(
#     collection=zone_summary,
#     description='Biomass_Zone_Summary',
#     fileFormat='CSV'
# )
# task2.start()
""")

print("\n" + "="*60)
print("✓ Script complete!")
print("="*60)