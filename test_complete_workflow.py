"""
Complete PINECONe Workflow with Dynamic TEV Parameterization
Integrates Scripts 1, 2, 3, and 4 with user-customizable economic parameters.
"""

import ee
import pandas as pd

# Import all modules
from src.pinecone.data.biomass import BiomassData
from src.pinecone.data.focal_species import FocalSpeciesLayer
from src.pinecone.carbon.biomass_stats import BiomassStatsCalculator
from src.pinecone.carbon.biomass_change import BiomassChangeCalculator
from src.pinecone.ecosystem.water_yield import WaterYieldCalculator
from src.pinecone.economics.tev_calculator import (
    TEVCalculator, 
    create_user_params_template
)

# Initialize Earth Engine
ee.Initialize()

print("="*70)
print("PINECONe Complete Workflow - All Scripts with Dynamic Parameterization")
print("="*70)

# ============================================================================
# STEP 1: CONFIGURATION
# ============================================================================

print("\n📋 Step 1: Configuration")
print("-"*70)

# AOIs (used in all scripts)
aois = {
    'EIA_CS1': ee.FeatureCollection('projects/servir-sco-assets/assets/Rx_Fire/Vector_Data/EIA_CS1'),
    'EIA_CS2': ee.FeatureCollection('projects/servir-sco-assets/assets/Rx_Fire/Vector_Data/EIA_CS2'),
    'EIA_CS3': ee.FeatureCollection('projects/servir-sco-assets/assets/Rx_Fire/Vector_Data/EIA_CS3')
}

# Focal species
focal_species = FocalSpeciesLayer(
    layer_source="projects/servir-sco-assets/assets/Rx_Fire/EO_Inputs/LEO_extantLLP_significance",
    binary_threshold=0,
    name="Longleaf Pine (LLP)"
)

# Biomass year
biomass_year = 2019

# Emissions date ranges
pre_date_start = '2018-01-01'
pre_date_end = '2018-12-31'
post_date_start = '2019-01-01'
post_date_end = '2019-12-31'

# Water yield date range (post-fire period)
water_yield_start = '2019-03-20'
water_yield_end = '2020-03-20'

# Economic parameters
carbon_credit_price = 10.0  # $/ton CO2e
water_price_per_kl = 0.018  # $/kL

# Case acres
case_acres = {
    'EIA_CS1_LLP': 651.06,
    'EIA_CS2_LLP': 937.02,
    'EIA_CS3_LLP': 544.03
}

print(f"✓ Configuration complete")

# ============================================================================
# STEP 2: SCRIPT 1 - BIOMASS STATISTICS
# ============================================================================

print("\n" + "="*70)
print("📊 Step 2: Script 1 - Biomass Statistics")
print("="*70)

biomass = BiomassData(product='esa_cci_agb', year=biomass_year)
calculator_biomass = BiomassStatsCalculator(biomass)

print("\nCalculating biomass statistics...")
all_stats_list = []

for aoi_name, aoi_fc in aois.items():
    print(f"  Processing {aoi_name}...")
    
    species_vectors = focal_species.vectorize(aoi_fc, scale=30)
    species_dissolved = species_vectors.union(1)
    non_species = focal_species.get_non_species_areas(aoi_fc, species_dissolved.geometry())
    
    stats_species = calculator_biomass.calculate_stats(species_vectors, f"{aoi_name}_LLP")
    stats_non_species = calculator_biomass.calculate_stats(non_species, f"{aoi_name}_NonLLP")
    
    all_stats_list.append(stats_species.merge(stats_non_species))

all_stats = all_stats_list[0]
for stats in all_stats_list[1:]:
    all_stats = all_stats.merge(stats)

zone_summary = calculator_biomass.calculate_zone_summary(all_stats)
biomass_results = zone_summary.getInfo()

# Extract biomass stats for LLP zones
biomass_stats_dict = {}
for feature in biomass_results['features']:
    props = feature['properties']
    zone = props['AOI_Zone']
    if 'LLP' in zone:
        biomass_stats_dict[zone] = {
            'AGB_per_acre_tons': props['mean_AGB_per_acre_tons'],
            'AGB_StdDev_per_acre_tons': props['mean_StdDev_per_acre_tons']
        }
        print(f"  ✓ {zone}: {props['mean_AGB_per_acre_tons']:.2f} ± {props['mean_StdDev_per_acre_tons']:.2f} tons/acre")

# ============================================================================
# STEP 3: SCRIPT 2 - CARBON EMISSIONS
# ============================================================================

print("\n" + "="*70)
print("💨 Step 3: Script 2 - Carbon Emissions")
print("="*70)

emissions_calculator = BiomassChangeCalculator(
    biomass_data=None,
    focal_species=focal_species,
    carbon_fraction=0.51,
    credit_price_per_ton=1.0
)

print("\nCalculating carbon emissions...")
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

# Extract emissions stats
emissions_stats_dict = {}
for feature in emissions_results['features']:
    props = feature['properties']
    zone = props['AOI_Zone']
    zone_llp = f"{zone}_LLP"
    emissions_stats_dict[zone_llp] = {
        'CO2_mean_tons_per_acre': props['CO2_mean_tons_per_acre'],
        'CO2_std_tons_per_acre': props.get('CO2_std_tons_per_acre', 0)
    }
    print(f"  ✓ {zone_llp}: {props['CO2_mean_tons_per_acre']:.2f} ± {props.get('CO2_std_tons_per_acre', 0):.2f} tons CO2/acre")

# ============================================================================
# STEP 4: SCRIPT 4 - WATER YIELD
# ============================================================================

print("\n" + "="*70)
print("💧 Step 4: Script 4 - Water Yield (Ecosystem Services)")
print("="*70)

water_calculator = WaterYieldCalculator(
    water_price_per_kl=water_price_per_kl,
    et_scale_factor=0.1
)

print("\nCalculating water yield...")
print("(Note: May fail if ET/precipitation data unavailable for date range)")

try:
    water_yield_results = water_calculator.calculate_for_multiple_aois(
        aois=aois,
        start_date=water_yield_start,
        end_date=water_yield_end,
        scale=500
    )
    
    # Check if any succeeded
    successful_calcs = sum(1 for v in water_yield_results.values() if v is not None)
    
    if successful_calcs > 0:
        print(f"\n✓ Water yield calculated for {successful_calcs}/{len(aois)} AOIs")
    else:
        print("\n⚠️  Water yield calculation failed for all AOIs")
        print("    Will use default water quality values in TEV calculation")
        water_yield_results = None
        
except Exception as e:
    print(f"\n⚠️  Water yield calculation failed: {e}")
    print("    Will use default water quality values in TEV calculation")
    water_yield_results = None

# ============================================================================
# STEP 5: USER-DEFINED ECONOMIC PARAMETERS
# ============================================================================

print("\n" + "="*70)
print("⚙️ Step 5: Define Custom Economic Parameters")
print("="*70)

# Create a template
zone_names = list(biomass_stats_dict.keys())
user_params_template = create_user_params_template(zone_names)

# Customize parameters for each zone
# Users can modify these values!
user_params = {
    "EIA_CS1_LLP": {
        'E_Pt': (7.50, 1.0),       # CUSTOMIZABLE: Stumpage price ($/ton) ± std
        'g': (375, 50),             # CUSTOMIZABLE: Regeneration cost ($/acre) ± std
        'endangered_species_WTP': (13.37 * 0.1, 1),  # CUSTOMIZABLE: RCW WTP
        'R_t_lease': [50, 20, 0, 0, 0],  # CUSTOMIZABLE: Annual lease revenues
        'T_lease': 5,               # CUSTOMIZABLE: Lease period (years)
        'r_lease': 0.06,            # CUSTOMIZABLE: Discount rate
        'epsilon_t': 0              # CUSTOMIZABLE: Price shock (optional)
    },
    "EIA_CS2_LLP": {
        'E_Pt': (21, 3),
        'g': (200, 30),
        'endangered_species_WTP': (13.37 * 0.5, 2),
        'R_t_lease': [200, 100, 50, 20, 10],
        'T_lease': 5,
        'r_lease': 0.055,
        'epsilon_t': 0
    },
    "EIA_CS3_LLP": {
        'E_Pt': (36, 5),
        'g': (50, 10),
        'endangered_species_WTP': (13.37 * 1, 3),
        'R_t_lease': [700, 700, 700, 700, 700],
        'T_lease': 5,
        'r_lease': 0.05,
        'epsilon_t': 0
    }
}

print("\nEconomic parameters defined:")
for zone_name in user_params.keys():
    params = user_params[zone_name]
    print(f"\n  {zone_name}:")
    print(f"    Stumpage price: ${params['E_Pt'][0]:.2f}/ton ± ${params['E_Pt'][1]:.2f}")
    print(f"    Regeneration cost: ${params['g'][0]:.2f}/acre ± ${params['g'][1]:.2f}")
    print(f"    Species WTP: ${params['endangered_species_WTP'][0]:.2f}/acre")
    print(f"    Lease revenues: {params['R_t_lease']}")

# ============================================================================
# STEP 6: SCRIPT 3 - TOTAL ECONOMIC VALUE (TEV)
# ============================================================================

print("\n" + "="*70)
print("💰 Step 6: Script 3 - Total Economic Value (TEV)")
print("="*70)

tev_calculator = TEVCalculator(random_seed=42)

print("\nRunning Monte Carlo simulations (10,000 iterations)...")
print("Integrating: Biomass + Emissions + Water Yield + User Parameters")

# Run TEV with all integrated data
tev_results = tev_calculator.run_monte_carlo(
    biomass_stats=biomass_stats_dict,
    emissions_stats=emissions_stats_dict,
    case_acres=case_acres,
    water_yield_stats=water_yield_results,  # Integrated from Script 4!
    user_params=user_params,  # User-customized parameters!
    carbon_credit_price=carbon_credit_price,
    num_simulations=10000
)

print("\n✓ TEV calculations complete")

# ============================================================================
# STEP 7: DISPLAY RESULTS
# ============================================================================

print("\n" + "="*70)
print("📊 RESULTS SUMMARY")
print("="*70)

print("\nTotal Economic Value (TEV) by Scenario:")
print(tev_results.to_string(index=False))

# ============================================================================
# STEP 8: VISUALIZATIONS
# ============================================================================

print("\n" + "="*70)
print("📈 Generating Visualizations")
print("="*70)

tev_calculator.plot_distributions()
tev_calculator.plot_boxplots()

# ============================================================================
# STEP 9: EXPORT RESULTS
# ============================================================================

print("\n" + "="*70)
print("💾 Exporting Results")
print("="*70)

tev_calculator.export_results('tev_results_integrated.csv')

# ============================================================================
# STEP 10: DETAILED BREAKDOWN
# ============================================================================

print("\n" + "="*70)
print("📋 COMPONENT BREAKDOWN")
print("="*70)

print("\nFor each zone, TEV includes:")
print("\n1. Timber Value (PV) = Stumpage Price × Biomass Volume - Regeneration Cost")
print("2. Carbon Benefits (PVC) = CO2 Change × Carbon Credit Price")
print("3. Ecosystem Services (PE) = Water Yield Value + Species WTP")
print("4. Land Value (L) = NPV of Lease Revenues")

# Only show LLP zones (NonLLP doesn't have emissions data)
for zone_name in [k for k in biomass_stats_dict.keys() if 'LLP' in k and 'NonLLP' not in k]:
    print(f"\n{zone_name}:")
    print(f"  Biomass (from Script 1): {biomass_stats_dict[zone_name]['AGB_per_acre_tons']:.2f} tons/acre")
    print(f"  Emissions (from Script 2): {emissions_stats_dict[zone_name]['CO2_mean_tons_per_acre']:.2f} tons CO2/acre")
    
    base_zone = zone_name.replace('_LLP', '')
    if water_yield_results and base_zone in water_yield_results and water_yield_results[base_zone]:
        water_value = water_yield_results[base_zone]['water_yield_per_acre_usd']
        print(f"  Water Yield (from Script 4): ${water_value:.2f}/acre")
    else:
        print(f"  Water Yield (default used): $100.00/acre")
    
    print(f"  Stumpage Price (user param): ${user_params[zone_name]['E_Pt'][0]:.2f}/ton")
    print(f"  Regen Cost (user param): ${user_params[zone_name]['g'][0]:.2f}/acre")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("✓ COMPLETE WORKFLOW FINISHED!")
print("="*70)

print(f"""
📋 What was calculated:
  ✓ Biomass statistics (Script 1)
  ✓ Carbon emissions (Script 2)
  ✓ Water yield ecosystem services (Script 4)
  ✓ Total Economic Value with uncertainty (Script 3)
  
🎯 Key Features:
  ✓ All calculated values automatically integrated
  ✓ User parameters easily customizable per zone
  ✓ Monte Carlo uncertainty quantification
  ✓ Professional visualizations
  ✓ Results exported to CSV
  
💡 To modify parameters:
  1. Edit the 'user_params' dictionary in Step 5
  2. Rerun from Step 6 onwards
  3. Compare different scenarios
""")

print("\n🎉 PINECONe analysis complete!")