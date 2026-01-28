"""
Test script for water yield calculation (Script 4).
Demonstrates integration with TEV calculator.
"""

import ee
from src.pinecone.ecosystem.water_yield import WaterYieldCalculator, create_water_yield_params

# Initialize Earth Engine
ee.Initialize()

print("="*70)
print("PINECONe Water Yield Calculation - Script 4")
print("="*70)

# ============================================================================
# CONFIGURATION - DYNAMIC PARAMETERS
# ============================================================================

print("\n📋 Configuration")
print("-"*70)

# 1. DYNAMIC: AOIs (same as Scripts 1, 2, 3)
aois = {
    'EIA_CS1': ee.FeatureCollection('projects/servir-sco-assets/assets/Rx_Fire/Vector_Data/EIA_CS1'),
    'EIA_CS2': ee.FeatureCollection('projects/servir-sco-assets/assets/Rx_Fire/Vector_Data/EIA_CS2'),
    'EIA_CS3': ee.FeatureCollection('projects/servir-sco-assets/assets/Rx_Fire/Vector_Data/EIA_CS3')
}

# Example for additional case studies mentioned in your script
# Uncomment and add these if you have them:
# aois['CS4'] = ee.FeatureCollection('projects/your/CS4')  # Managed - 1272 acres
# aois['CS7'] = ee.FeatureCollection('projects/your/CS7')  # Unmanaged - 2672 acres
# aois['CS8'] = ee.FeatureCollection('projects/your/CS8')  # Prescribed - 1028 acres

# 2. DYNAMIC: Date range (post-fire period)
# This should match your fire timing from Scripts 1 & 2
start_date = '2019-03-20'  # CHANGEABLE: Post-fire start
end_date = '2020-03-20'    # CHANGEABLE: One year post-fire

# 3. DYNAMIC: Water price
water_price_per_kl = 0.018  # CHANGEABLE: $/kL (default from original script)

print(f"✓ {len(aois)} AOIs configured")
print(f"✓ Analysis period: {start_date} to {end_date}")
print(f"✓ Water price: ${water_price_per_kl}/kL")

# ============================================================================
# CALCULATE WATER YIELD
# ============================================================================

print("\n" + "="*70)
print("💧 Calculating Water Yield")
print("="*70)

# Initialize calculator
calculator = WaterYieldCalculator(
    water_price_per_kl=water_price_per_kl,
    et_scale_factor=0.1  # MODIS ET scaling factor
)

# Calculate for all AOIs
print("\nProcessing AOIs...")
water_yield_results = calculator.calculate_for_multiple_aois(
    aois=aois,
    start_date=start_date,
    end_date=end_date,
    scale=500
)

print("\n✓ Water yield calculations complete")

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

print("\n" + "="*70)
print("📊 WATER YIELD RESULTS")
print("="*70)

for aoi_name, results in water_yield_results.items():
    if results:
        print(f"\n{aoi_name}:")
        print(f"  Area: {results['area_acres']:.2f} acres")
        print(f"  Precipitation: ${results['precipitation_per_acre_usd']:.2f}/acre")
        print(f"  Evapotranspiration: ${results['et_per_acre_usd']:.2f}/acre")
        print(f"  Water Yield: ${results['water_yield_per_acre_usd']:.2f}/acre ± ${results['water_yield_std_per_acre_usd']:.2f}")
        print(f"  Total Water Yield: {results['total_water_yield_kl']:.2f} kL")
        print(f"  Total Value: ${results['total_water_yield_usd']:.2f}")

# ============================================================================
# INTEGRATION WITH TEV CALCULATOR
# ============================================================================

print("\n" + "="*70)
print("🔗 Integration with TEV Calculator")
print("="*70)

print("\nConverting to TEV parameters...")

# Convert water yield results to TEV parameters
tev_water_params = {}
for aoi_name, results in water_yield_results.items():
    if results:
        params = create_water_yield_params(results, as_tuple=True)
        tev_water_params[aoi_name] = params
        print(f"  {aoi_name}: {params['water_quality_value']}")

print("\n✓ Parameters ready for TEV calculator")

# ============================================================================
# EXAMPLE: USE IN TEV CALCULATOR
# ============================================================================

print("\n" + "="*70)
print("💡 Example: Using Water Yield in TEV Calculator")
print("="*70)

print("""
# Update your TEV economic parameters with calculated water yield values:

from src.pinecone.economics.tev_calculator import DEFAULT_ECONOMIC_PARAMS

# Example for EIA_CS1
updated_params = DEFAULT_ECONOMIC_PARAMS["EIA_CS1_LLP"].copy()

# Replace water_quality_value with calculated value
water_yield_params = create_water_yield_params(
    water_yield_results['EIA_CS1'],
    as_tuple=True
)

updated_params.update(water_yield_params)

# Now use in TEV calculator:
# tev_results = tev_calculator.run_monte_carlo(
#     input_df=input_df,
#     base_cases={'EIA_CS1_LLP': updated_params},
#     case_acres={'EIA_CS1_LLP': 651.06},
#     num_simulations=10000
# )
""")

# ============================================================================
# VISUALIZATION (OPTIONAL)
# ============================================================================

print("\n" + "="*70)
print("📈 Time Series Visualization")
print("="*70)

print("\nTo visualize time series data, use:")
print("""
# Export ET time series
et_timeseries = calculator.export_timeseries(
    aoi=aois['EIA_CS1'],
    start_date=start_date,
    end_date=end_date,
    output_type='et'
)

# Export precipitation time series
precip_timeseries = calculator.export_timeseries(
    aoi=aois['EIA_CS1'],
    start_date=start_date,
    end_date=end_date,
    output_type='precip'
)

# Export to CSV
ee.batch.Export.table.toDrive(
    collection=et_timeseries,
    description='ET_TimeSeries',
    fileFormat='CSV'
).start()
""")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("✓ Script Complete!")
print("="*70)

print(f"""
📋 Summary:
  • Processed {len(water_yield_results)} AOIs
  • Analysis period: {start_date} to {end_date}
  • Water price: ${water_price_per_kl}/kL
  • Results ready for TEV calculator integration
  
💡 Next Steps:
  1. Use water yield values to update TEV economic parameters
  2. Run complete TEV analysis with updated water quality values
  3. Compare scenarios with different management strategies
""")

print("\n" + "="*70)
print("🌊 Water Yield Analysis Complete!")
print("="*70)