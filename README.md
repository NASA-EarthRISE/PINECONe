# 🌲 PINECONe

**Pine Ecosystem Carbon and Economics RX Fire Analysis**
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18419563.svg)](https://doi.org/10.5281/zenodo.18419563)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NASA-EarthRISE/PINECONe/blob/main/PINECONe_Demo.ipynb)

A Python package for analyzing prescribed fire impacts on forest ecosystems, combining remote sensing data, carbon accounting, and economic valuation.

---

Mayer, T., Walker, K., & Basu, R. (2026). NASA-EarthRISE/PINECONe: v0.1.0 (v0.1.0). Zenodo. https://doi.org/10.5281/zenodo.18419563

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Complete Workflow](#complete-workflow)
- [Package Structure](#package-structure)
- [Documentation](#documentation)
- [Examples](#examples)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

PINECONe integrates three key components for prescribed fire analysis:

1. **📊 Biomass Statistics** - Calculate above-ground biomass for focal species (e.g., Longleaf Pine) and surrounding ecosystems
2. **💨 Carbon Emissions** - Estimate CO₂ emissions or sequestration from biomass changes over time
3. **💰 Economic Valuation** - Calculate Total Economic Value (TEV) including timber, carbon credits, ecosystem services, and land value

### Key Capabilities

- **Dynamic Configuration** - Easily switch between data sources, time periods, and study areas
- **Multi-Source Remote Sensing** - Supports ESA-CCI AGB, GEDI L4B, WHRC biomass products
- **Quality Control** - Built-in quality filtering for biomass change detection
- **Uncertainty Quantification** - Monte Carlo simulation for robust economic estimates
- **Scalable** - Analyze individual plots to landscape-scale regions

---

## ✨ Features

### 🛰️ Remote Sensing Integration
- **Google Earth Engine** backend for cloud-based processing
- Multiple biomass products: ESA-CCI AGB, GEDI L4B, WHRC
- Support for Landsat, Sentinel-2, MODIS imagery
- Automated cloud masking and compositing

### 🌲 Ecological Analysis
- **Focal species mapping** - Identify and quantify specific tree species
- **Biomass statistics** - Calculate per-acre and total biomass with uncertainty
- **Change detection** - Track biomass gains/losses over time
- **Quality flagging** - Filter unreliable change estimates

### 💰 Economic Analysis
- **Carbon credits** - Calculate marketable carbon offsets
- **Multiple methodologies** - VCS, CAR, Gold Standard, IPCC
- **Timber valuation** - Estimate stumpage value and regeneration costs
- **Ecosystem services** - Include water quality and biodiversity values
- **Monte Carlo simulation** - 10,000+ simulations for uncertainty analysis

### 📊 Outputs
- **CSV exports** - All results in tabular format
- **Visualizations** - Distribution plots, boxplots, bar charts
- **GeoJSON** - Spatial outputs for GIS integration
- **Reports** - Summary statistics and confidence intervals

---

## 🚀 Installation

### Requirements
- Python 3.8+
- Google Earth Engine account (free)

### Option 1: Install from [PyPI](https://pypi.org/project/pinecone-fire/0.1.0/)
```bash
pip install pinecone-fire==0.1.0
```

### Option 2: Install from GitHub
```bash
pip install git+https://github.com/NASA-EarthRISE/PINECONe.git
```

### Option 3: Local Development
```bash
git clone https://github.com/NASA-EarthRISE/PINECONe.git
cd PINECONe
pip install -e .
```

### Earth Engine Authentication
```python
import ee
ee.Authenticate()  # First time only
ee.Initialize()
```

---

## ⚡ Quick Start

### Google Colab (Easiest!)
Click the badge above to open our interactive demo notebook in Google Colab. No installation required!

### Python Script

```python
import ee
from pinecone.data.biomass import BiomassData
from pinecone.data.focal_species import FocalSpeciesLayer
from pinecone.carbon.biomass_stats import BiomassStatsCalculator

# Initialize Earth Engine
ee.Initialize()

# Load your study area
aoi = ee.FeatureCollection('path/to/your/study/area')

# Configure focal species (e.g., Longleaf Pine)
focal_species = FocalSpeciesLayer(
    layer_source="path/to/species/layer",
    name="Longleaf Pine"
)

# Load biomass data
biomass = BiomassData(
    product='esa_cci_agb',
    year=2019
)

# Calculate statistics
calculator = BiomassStatsCalculator(biomass)
species_vectors = focal_species.vectorize(aoi, scale=30)
stats = calculator.calculate_stats(species_vectors, "MyStudyArea")

print(stats.getInfo())
```

---

## 📖 Complete Workflow

### Script 1: Biomass Statistics

Calculate above-ground biomass for focal species vs. non-focal species areas:

```python
from pinecone.data.biomass import BiomassData
from pinecone.data.focal_species import FocalSpeciesLayer
from pinecone.carbon.biomass_stats import BiomassStatsCalculator

# 1. Define your study areas
aois = {
    'Study_Area_1': ee.FeatureCollection('projects/your/area1'),
    'Study_Area_2': ee.FeatureCollection('projects/your/area2')
}

# 2. Configure focal species
focal_species = FocalSpeciesLayer(
    layer_source="projects/your/species/layer",
    binary_threshold=0,
    name="Longleaf Pine"
)

# 3. Load biomass data (choose product and year)
biomass = BiomassData(
    product='esa_cci_agb',  # or 'gedi_l4b', 'whrc'
    year=2019
)

# 4. Calculate statistics
calculator = BiomassStatsCalculator(biomass)

for aoi_name, aoi_fc in aois.items():
    # Vectorize species areas
    species_vectors = focal_species.vectorize(aoi_fc, scale=30)
    
    # Calculate biomass stats
    stats = calculator.calculate_stats(species_vectors, aoi_name)
    
    print(f"{aoi_name}: {stats.getInfo()}")
```

**Outputs:**
- Mean biomass per acre (tons/acre) ± standard deviation
- Total biomass (tons)
- Area statistics (acres)
- Separate statistics for focal species vs. non-focal species areas

---

### Script 2: Carbon Emissions

Estimate CO₂ emissions from biomass changes:

```python
from pinecone.carbon.biomass_change import BiomassChangeCalculator

# Initialize calculator
emissions_calculator = BiomassChangeCalculator(
    biomass_data=None,
    focal_species=focal_species,
    carbon_fraction=0.51,  # Carbon content of biomass
    credit_price_per_ton=1.0
)

# Calculate emissions for multiple AOIs
emissions_fc = emissions_calculator.calculate_for_multiple_aois(
    aois=aois,
    pre_date_start='2018-01-01',   # Before fire
    pre_date_end='2018-12-31',
    post_date_start='2019-01-01',  # After fire
    post_date_end='2019-12-31',
    resolution=100,
    apply_quality_filter=True  # Use ESA-CCI quality flags
)

# Get results
results = emissions_fc.getInfo()
for feature in results['features']:
    props = feature['properties']
    print(f"{props['AOI_Zone']}: {props['CO2_mean_tons_per_acre']:.2f} tons CO2/acre")
```

**Outputs:**
- CO₂ emissions per acre (tons/acre) ± standard deviation
- Total CO₂ emissions (tons)
- Negative values = carbon sequestration (biomass gain)
- Positive values = carbon emissions (biomass loss)

---

### Script 3: Economic Valuation

Calculate Total Economic Value with uncertainty:

```python
from pinecone.economics.tev_calculator import TEVCalculator, DEFAULT_ECONOMIC_PARAMS

# Initialize TEV calculator
tev_calculator = TEVCalculator(random_seed=42)

# Create input data from previous scripts
input_df = tev_calculator.create_input_dataframe(
    biomass_stats=biomass_stats_dict,      # From Script 1
    emissions_stats=emissions_stats_dict,  # From Script 2
    carbon_credit_price=10.0  # $/ton CO2e
)

# Define economic parameters for each area
case_acres = {
    'Study_Area_1_LLP': 650.0,
    'Study_Area_2_LLP': 900.0
}

# Run Monte Carlo simulation (10,000 iterations)
tev_results = tev_calculator.run_monte_carlo(
    input_df=input_df,
    base_cases=DEFAULT_ECONOMIC_PARAMS,  # Timber prices, ecosystem services, etc.
    case_acres=case_acres,
    num_simulations=10000
)

# Display results
print(tev_results)

# Generate visualizations
tev_calculator.plot_distributions()
tev_calculator.plot_boxplots()

# Export to CSV
tev_calculator.export_results('tev_results.csv')
```

**TEV Components:**
1. **Timber Value** = Stumpage price × Volume - Regeneration cost
2. **Carbon Credits** = CO₂ change × Credit price
3. **Ecosystem Services** = Water quality + Endangered species WTP
4. **Land Value** = NPV of hunting/lease revenues

**Outputs:**
- Mean TEV ($) ± standard deviation
- Median, 25th percentile, 75th percentile
- Distribution plots
- Sensitivity analysis

---

## 📁 Package Structure

```
PINECONe/
├── src/
│   └── pinecone/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── biomass.py           # BiomassData class
│       │   └── focal_species.py     # FocalSpeciesLayer class
│       ├── carbon/
│       │   ├── __init__.py
│       │   ├── biomass_stats.py     # BiomassStatsCalculator
│       │   └── biomass_change.py    # BiomassChangeCalculator
|       ├── ecosystem/
│       │   ├── __init__.py
│       │   ├── water_yield.py       # Calaculates Water value
│       └── economics/
│           ├── __init__.py
│           └── tev_calculator.py    # TEVCalculator
├── tests/
│   ├── test_biomass.py
│   ├── test_emissions.py
│   └── test_tev.py
├── examples/
│   ├── PINECONe_Demo.ipynb          # Interactive Colab demo
│   └── example_workflow.py          # Complete Python example
├── docs/
│   └── [documentation files]
├── README.md
├── LICENSE
├── setup.py
└── pyproject.toml
```

---

## 📚 Documentation

### Core Classes

#### `BiomassData`
Loads and manages biomass data from multiple sources.

**Parameters:**
- `product` (str): Biomass product ('esa_cci_agb', 'gedi_l4b', 'whrc', 'custom')
- `year` (int): Year to load data for
- `custom_image` (ee.Image): Optional custom biomass layer

**Methods:**
- `get_biomass_per_pixel()`: Returns biomass in tons per pixel
- `list_available_products()`: Lists supported products

---

#### `FocalSpeciesLayer`
Manages focal species layers and vectorization.

**Parameters:**
- `layer_source` (str or ee.Image): Path to species layer
- `binary_threshold` (float): Threshold for binary mask (default: 0)
- `name` (str): Display name for species

**Methods:**
- `vectorize(aoi, scale)`: Convert raster to vector polygons
- `get_non_species_areas(aoi, species_geom)`: Extract non-species areas

---

#### `BiomassStatsCalculator`
Calculate biomass statistics for polygons.

**Parameters:**
- `biomass_data` (BiomassData): Biomass data source

**Methods:**
- `calculate_stats(feature_collection, zone_name, scale)`: Calculate per-polygon stats
- `calculate_zone_summary(stats_fc)`: Aggregate statistics by zone

---

#### `BiomassChangeCalculator`
Calculate emissions from biomass changes.

**Parameters:**
- `focal_species` (FocalSpeciesLayer): Optional species layer
- `carbon_fraction` (float): Carbon content (default: 0.51)
- `credit_price_per_ton` (float): Credit price ($/ton)

**Methods:**
- `calculate_change(pre_dates, post_dates, aoi, resolution)`: Calculate change
- `calculate_for_multiple_aois(aois, ...)`: Batch processing

---

#### `TEVCalculator`
Calculate Total Economic Value with Monte Carlo simulation.

**Parameters:**
- `random_seed` (int): Seed for reproducibility

**Methods:**
- `calculate_tev(case_params, total_acres, num_simulations)`: Run simulation
- `run_monte_carlo(input_df, base_cases, case_acres, num_simulations)`: Batch analysis
- `plot_distributions()`: Visualize TEV distributions
- `plot_boxplots()`: Compare scenarios
- `export_results(filepath)`: Save to CSV

---

## 💡 Examples

### Example 1: Single Study Area Analysis

```python
# Full workflow for one area
from pinecone import *

# 1. Setup
aoi = ee.FeatureCollection('projects/my/study/area')
focal_species = FocalSpeciesLayer("projects/my/llp/layer", name="LLP")
biomass = BiomassData('esa_cci_agb', year=2019)

# 2. Biomass stats
calculator = BiomassStatsCalculator(biomass)
species_vectors = focal_species.vectorize(aoi)
stats = calculator.calculate_stats(species_vectors, "MyArea")

# 3. Emissions
emissions_calc = BiomassChangeCalculator(focal_species=focal_species)
emissions = emissions_calc.calculate_change(
    '2018-01-01', '2018-12-31',
    '2019-01-01', '2019-12-31',
    aoi
)

# 4. TEV
tev_calc = TEVCalculator()
# ... continue with TEV calculation
```

### Example 2: Multi-Year Analysis

```python
# Compare multiple years
years = [2017, 2018, 2019, 2020]
results = {}

for year in years:
    biomass = BiomassData('esa_cci_agb', year=year)
    # ... calculate stats
    results[year] = stats

# Analyze trends
import pandas as pd
df = pd.DataFrame(results).T
df.plot()
```

### Example 3: Custom Economic Parameters

```python
# Define your own economic scenarios
custom_params = {
    "MyStudyArea_LLP": {
        'E_Pt': (25.0, 5),      # Stumpage price ($/ton) ± std
        'g': (200, 30),          # Regeneration cost ($/acre) ± std
        'water_quality_value': (150.0, 10.0),
        'endangered_species_WTP': (20.0, 5.0),
        'R_t_lease': [500, 500, 500, 500, 500],  # Annual lease revenues
        'T_lease': 5,
        'r_lease': 0.05
    }
}

tev_results = tev_calculator.run_monte_carlo(
    input_df=input_df,
    base_cases=custom_params,
    case_acres={'MyStudyArea_LLP': 1000.0},
    num_simulations=10000
)
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/NASA-EarthRISE/PINECONe.git
cd PINECONe

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/
```

### Reporting Issues

Found a bug or have a feature request? Please [open an issue](https://github.com/NASA-EarthRISE/PINECONe/issues).

---

## 📄 Citation

If you use PINECONe in your research, please cite:

```bibtex
@software{pinecone2025,
  title = {PINECONe: Pine Ecosystem Carbon and Economics RX Fire Analysis},
  author = {{NASA EarthRISE Team}},
  year = {2025},
  url = {https://github.com/NASA-EarthRISE/PINECONe},
  version = {0.1.0}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **NASA EarthRISE** - 
- **Google Earth Engine** - Cloud computing platform
- **ESA Climate Change Initiative** - Biomass data products
- **GEDI Mission** - Lidar biomass estimates

---

## 📞 Contact

- **GitHub Issues**: https://github.com/NASA-EarthRISE/PINECONe/issues
- **Email**: [timothy.j.mayer@nasa.gov]

---

## 🗺️ Roadmap

- [ ] Publish to PyPI
- [ ] Add ReadTheDocs documentation
- [ ] Support for additional biomass products
- [ ] Web-based dashboard
- [ ] Integration with carbon registries
- [ ] Machine learning for fire detection
- [ ] Multi-year time series analysis

---

**Made with 🌲 by NASA EarthRISE**
