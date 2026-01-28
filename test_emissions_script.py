"""
Test script for biomass change and carbon emissions calculation.
This replicates your second JavaScript workflow.
"""

import ee

# Import modules
from src.pinecone.data.focal_species import FocalSpeciesLayer
from src.pinecone.carbon.biomass_change import BiomassChangeCalculator

# Initialize Earth Engine
ee.Initialize()

print("="*60)
print("PINECONe Carbon Emissions - Script 2")
print("="*60)

# ---------------------
# DYNAMIC INPUTS
# ---------------------

# 1. DYNAMIC: AOIs (same as Script 1)
print("\n1. Loading AOIs...")
EIA_CS1 = ee.FeatureCollection('projects/servir-sco-assets/assets/Rx_Fire/Vector_Data/EIA_CS1')
EIA_CS2 = ee.FeatureCollection('projects/servir-sco-assets/assets/Rx_Fire/Vector_Data/EIA_CS2')
EIA_CS3 = ee.FeatureCollection('projects/servir-sco-assets/assets/Rx_Fire/Vector_Data/EIA_CS3')

aois = {
    'EIA_CS1': EIA_CS1,
    'EIA_CS2': EIA_CS2,
    'EIA_CS3': EIA_CS3
}
print(f"   ✓ Loaded {len(aois)} AOIs")

# 2. DYNAMIC: Focal Species Layer (same as Script 1)
print("\n2. Loading focal species layer...")
focal_species = FocalSpeciesLayer(
    layer_source="projects/servir-sco-assets/assets/Rx_Fire/EO_Inputs/LEO_extantLLP_significance",
    binary_threshold=0,
    name="Longleaf Pine (LLP)"
)
print(f"   ✓ Loaded: {focal_species.name}")

# 3. DYNAMIC: Date Ranges
print("\n3. Configuring date ranges...")
# Pre-fire period
pre_date_start = '2018-01-01'  # CHANGEABLE
pre_date_end = '2018-12-31'    # CHANGEABLE

# Post-fire period
post_date_start = '2019-01-01'  # CHANGEABLE
post_date_end = '2019-12-31'    # CHANGEABLE

print(f"   Pre-fire:  {pre_date_start} to {pre_date_end}")
print(f"   Post-fire: {post_date_start} to {post_date_end}")

# 4. DYNAMIC: Emissions Constants
print("\n4. Configuring emissions parameters...")
carbon_fraction = 0.51  # CHANGEABLE: Carbon fraction of biomass
credit_price = 1.0      # CHANGEABLE: Carbon credit price ($/ton CO2e)

print(f"   Carbon fraction: {carbon_fraction}")
print(f"   Credit price: ${credit_price}/ton CO2e")

# 5. Create calculator
print("\n5. Initializing calculator...")
calculator = BiomassChangeCalculator(
    biomass_data=None,  # Not needed for this workflow
    focal_species=focal_species,
    carbon_fraction=carbon_fraction,
    credit_price_per_ton=credit_price
)
print("   ✓ Calculator initialized")

# ---------------------
# CALCULATE EMISSIONS
# ---------------------

print("\n6. Calculating carbon emissions for all AOIs...")
print("   (This may take a few minutes)")

results_fc = calculator.calculate_for_multiple_aois(
    aois=aois,
    pre_date_start=pre_date_start,
    pre_date_end=pre_date_end,
    post_date_start=post_date_start,
    post_date_end=post_date_end,
    resolution=100,
    apply_quality_filter=True  # CHANGEABLE: Apply quality flag filtering
)

print("\n   ✓ All AOIs processed")

# ---------------------
# DISPLAY RESULTS
# ---------------------

print("\n" + "="*60)
print("RESULTS")
print("="*60)

results_info = results_fc.getInfo()

for feature in results_info['features']:
    props = feature['properties']
    print(f"\n{props['AOI_Zone']}:")
    print(f"  Area: {props['area_acres']:.2f} acres")
    print(f"  CO2 emissions: {props['CO2_mean_tons_per_acre']:.2f} tons/acre")
    print(f"  Total CO2: {props['CO2_total_tons']:.2f} tons")
    
    if 'CO2_std_tons_per_acre' in props:
        print(f"  Std Dev: ±{props['CO2_std_tons_per_acre']:.2f} tons/acre")
    
    print(f"  Carbon credit: ${props['credit_mean_usd_per_acre']:.2f}/acre")
    print(f"  Total credit: ${props['credit_total_usd']:.2f}")
    
    if 'credit_std_usd_per_acre' in props:
        print(f"  Credit std: ±${props['credit_std_usd_per_acre']:.2f}/acre")

# ---------------------
# EXPORT (optional)
# ---------------------

print("\n" + "="*60)
print("EXPORT")
print("="*60)
print("\nTo export to Google Drive, uncomment these lines:")
print("""
# Export results
# task = calculator.export_results(
#     results_fc,
#     description='Carbon_Emissions_Stats_2018_2019'
# )
# print(f"Export task started: {task.status()}")
""")

print("\n" + "="*60)
print("✓ Script complete!")
print("="*60)

# ---------------------
# TEST DIFFERENT SCENARIOS
# ---------------------

print("\n" + "="*60)
print("TESTING DIFFERENT SCENARIOS")
print("="*60)

print("\nYou can easily test different scenarios:")
print("""
# Different years
calculator.calculate_for_multiple_aois(
    aois=aois,
    pre_date_start='2017-01-01',
    pre_date_end='2017-12-31',
    post_date_start='2020-01-01',
    post_date_end='2020-12-31',
    ...
)

# Different carbon credit price
calculator.credit_price_per_ton = 15.0  # $15/ton

# Different carbon fraction
calculator.carbon_fraction = 0.47

# Without quality filtering
results = calculator.calculate_for_multiple_aois(
    ...,
    apply_quality_filter=False
)
""")