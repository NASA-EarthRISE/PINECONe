"""
Complete PINECONe Workflow - Scripts 1, 2, and 3 integrated.
Shows how biomass stats → carbon emissions → TEV calculation.
"""

import ee
import pandas as pd

# Import all modules
from src.pinecone.data.biomass import BiomassData
from src.pinecone.data.focal_species import FocalSpeciesLayer
from src.pinecone.carbon.biomass_stats import BiomassStatsCalculator
from src.pinecone.carbon.biomass_change import BiomassChangeCalculator
from src.pinecone.economics.tev_calculator import TEVCalculator, DEFAULT_ECONOMIC_PARAMS

# Initialize Earth Engine
ee.Initialize()

print("="*70)
print("PINECONe Complete Workflow - Scripts 1, 2, 3")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

print("\n📋 Configuration")
print("-"*70)

# AOIs (used in all scripts)
aois = {
    'EIA_CS1': ee.FeatureCollection('projects/servir-sco-assets/assets/Rx_Fire/Vector_Data/EIA_CS1'),
    'EIA_CS2': ee.FeatureCollection('projects/servir-sco-assets/assets/Rx_Fire/Vector_Data/EIA_CS2'),
    'EIA_CS3': ee.FeatureCollection('projects/servir-sco-assets/assets/Rx_Fire/Vector_Data/EIA_CS3')
}

# Focal species (used in all scripts)
focal_species = FocalSpeciesLayer(
    layer_source="projects/servir-sco-assets/assets/Rx_Fire/EO_Inputs/LEO_extantLLP_significance",
    binary_threshold=0,
    name="Longleaf Pine (LLP)"
)

# Date ranges for emissions
pre_date_start = '2018-01-01'
pre_date_end = '2018-12-31'
post_date_start = '2019-01-01'
post_date_end = '2019-12-31'

# Economic parameters
carbon_credit_price = 10.0  # $/ton CO2e
case_acres = {
    'EIA_CS1_LLP': 651.06,
    'EIA_CS2_LLP': 937.02,
    'EIA_CS3_LLP': 544.03
}

print(f"✓ {len(aois)} AOIs configured")
print(f"✓ Focal species: {focal_species.name}")
print(f"✓ Pre-fire: {pre_date_start} to {pre_date_end}")
print(f"✓ Post-fire: {post_date_start} to {post_date_end}")
print(f"✓ Carbon credit price: ${carbon_credit_price}/ton CO2e")

# ============================================================================
# SCRIPT 1: BIOMASS STATISTICS
# ============================================================================

print("\n" + "="*70)
print("📊 SCRIPT 1: Biomass Statistics (LLP vs Non-LLP)")
print("="*70)

# Configure biomass data
biomass = BiomassData(product='esa_cci_agb', year=2019)
calculator_biomass = BiomassStatsCalculator(biomass)

# Calculate stats for all AOIs
print("\nProcessing AOIs...")
all_stats_list = []

for aoi_name, aoi_fc in aois.items():
    print(f"  Processing {aoi_name}...")
    
    # Vectorize focal species
    species_vectors = focal_species.vectorize(aoi_fc, scale=30)
    species_dissolved = species_vectors.union(1)
    
    # Get non-species areas
    non_species = focal_species.get_non_species_areas(aoi_fc, species_dissolved.geometry())
    
    # Calculate stats
    stats_species = calculator_biomass.calculate_stats(species_vectors, f"{aoi_name}_LLP")
    stats_non_species = calculator_biomass.calculate_stats(non_species, f"{aoi_name}_NonLLP")
    
    all_stats_list.append(stats_species.merge(stats_non_species))

# Merge all zones
all_stats = all_stats_list[0]
for stats in all_stats_list[1:]:
    all_stats = all_stats.merge(stats)

# Calculate zone summaries
zone_summary = calculator_biomass.calculate_zone_summary(all_stats)
biomass_results = zone_summary.getInfo()

print("\n✓ Biomass statistics calculated")
print("\nResults (LLP only):")
biomass_stats_dict = {}
for feature in biomass_results['features']:
    props = feature['properties']
    zone = props['AOI_Zone']
    if 'LLP' in zone:
        print(f"  {zone}: {props['mean_AGB_per_acre_tons']:.2f} ± {props['mean_StdDev_per_acre_tons']:.2f} tons/acre")
        biomass_stats_dict[zone] = {
            'AGB_per_acre_tons': props['mean_AGB_per_acre_tons'],
            'AGB_StdDev_per_acre_tons': props['mean_StdDev_per_acre_tons']
        }

# ============================================================================
# SCRIPT 2: CARBON EMISSIONS
# ============================================================================

print("\n" + "="*70)
print("💨 SCRIPT 2: Carbon Emissions Calculation")
print("="*70)

# Configure emissions calculator
emissions_calculator = BiomassChangeCalculator(
    biomass_data=None,
    focal_species=focal_species,
    carbon_fraction=0.51,
    credit_price_per_ton=1.0  # Will multiply later
)

# Calculate emissions for all AOIs
print("\nCalculating emissions...")
emissions_fc = emissions_calculator.calculate_for_multiple_aois(
    aois=aois,
    pre_date_start=pre_date_start,
    pre_date_end=pre_date_end,
    post_date_start=post_date_start,
    post_date_end=post_date_end,
    resolution=100,
    apply_quality_filter=True
)

emissions_results = emissions_fc.getInfo()

print("\n✓ Carbon emissions calculated")
print("\nResults:")
emissions_stats_dict = {}
for feature in emissions_results['features']:
    props = feature['properties']
    zone = props['AOI_Zone']
    zone_llp = f"{zone}_LLP"
    print(f"  {zone_llp}: {props['CO2_mean_tons_per_acre']:.2f} ± {props.get('CO2_std_tons_per_acre', 0):.2f} tons CO2/acre")
    emissions_stats_dict[zone_llp] = {
        'CO2_mean_tons_per_acre': props['CO2_mean_tons_per_acre'],
        'CO2_std_tons_per_acre': props.get('CO2_std_tons_per_acre', 0)
    }

# ============================================================================
# SCRIPT 3: TOTAL ECONOMIC VALUE (TEV)
# ============================================================================

print("\n" + "="*70)
print("💰 SCRIPT 3: Total Economic Value (TEV) Calculation")
print("="*70)

# Initialize TEV calculator
tev_calculator = TEVCalculator(random_seed=42)

# Create input DataFrame from Scripts 1 & 2
print("\nCombining biomass and emissions data...")
input_df = tev_calculator.create_input_dataframe(
    biomass_stats=biomass_stats_dict,
    emissions_stats=emissions_stats_dict,
    carbon_credit_price=carbon_credit_price
)

print("Input data:")
print(input_df)

# Run Monte Carlo simulations
print("\nRunning Monte Carlo simulations (10,000 iterations)...")
tev_results = tev_calculator.run_monte_carlo(
    input_df=input_df,
    base_cases=DEFAULT_ECONOMIC_PARAMS,
    case_acres=case_acres,
    num_simulations=10000
)

print("\n✓ TEV calculations complete")
print("\nResults:")
print(tev_results)

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("📈 Generating Visualizations")
print("="*70)

# Plot distributions
print("\nPlotting TEV distributions...")
tev_calculator.plot_distributions()

# Plot boxplots
print("\nPlotting TEV comparison...")
tev_calculator.plot_boxplots()

# ============================================================================
# EXPORT
# ============================================================================

print("\n" + "="*70)
print("💾 Export Results")
print("="*70)

# Export TEV results
tev_calculator.export_results('tev_results.csv')

print("\n" + "="*70)
print("✓ Complete workflow finished!")
print("="*70)

print("\n📋 Summary:")
print(f"  • Processed {len(aois)} AOIs")
print(f"  • Calculated biomass statistics for LLP and Non-LLP areas")
print(f"  • Calculated carbon emissions from {pre_date_start} to {post_date_start}")
print(f"  • Ran 10,000 Monte Carlo simulations for TEV")
print(f"  • Results exported to tev_results.csv")

print("\n🎉 PINECONe analysis complete!")