"""
Total Economic Value (TEV) calculator for prescribed fire analysis.
Integrates biomass stats, carbon emissions, and water yield to calculate economic value.
Version 2.0 - Dynamic parameterization with calculated ecosystem services.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from typing import Dict, Optional, Tuple, Union


class TEVCalculator:
    """
    Calculate Total Economic Value (TEV) for prescribed fire scenarios.
    
    Components:
    - Timber Value (PV) - from biomass data + user-specified prices
    - Carbon Benefits (PVC) - from emissions data + user-specified credit price
    - Ecosystem Services (PE) - from water yield + user-specified species values
    - Land Value for Hunting/Leases (L) - user-specified lease revenues
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize TEV Calculator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)
        self.results = None
        self.simulation_results = None
    
    def calculate_tev(
        self,
        case_params: Dict,
        total_acres: float,
        num_simulations: int = 1
    ) -> np.ndarray:
        """
        Calculate Total Economic Value with Monte Carlo simulation.
        
        Args:
            case_params: Dictionary of parameters (can be single values or (mean, std) tuples)
            total_acres: Total acreage (if 0, returns per-acre values)
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            Array of TEV values
        """
        simulated_tevs_per_acre = np.zeros(num_simulations)
        
        def get_value(param):
            """Get value: either directly or from normal distribution."""
            if isinstance(param, tuple):  # (mean, std_dev)
                return np.random.normal(param[0], param[1])
            return param
        
        for i in range(num_simulations):
            # 1. Timber Value (PV) per acre
            E_Pt = get_value(case_params['E_Pt'])
            epsilon_t = get_value(case_params.get('epsilon_t', 0))
            P_actual_t = E_Pt + epsilon_t
            
            V_t = get_value(case_params['V_t'])
            g = get_value(case_params['g'])
            pv_timber_per_acre = P_actual_t * V_t - g
            
            # 2. Carbon Benefits (PVC) per acre
            pvc_per_acre_val = get_value(case_params['pvc_per_acre'])
            
            # 3. Ecosystem Services (PE) per acre
            water_quality_value = get_value(case_params['water_quality_value'])
            endangered_species_WTP = get_value(case_params['endangered_species_WTP'])
            pe_per_acre = water_quality_value + endangered_species_WTP
            
            # 4. Land Value for Hunting/Leases (L) per acre
            R_t_lease = case_params['R_t_lease']
            T_lease = case_params['T_lease']
            r_lease = get_value(case_params['r_lease'])
            
            l_per_acre = 0
            for t_val in range(1, T_lease + 1):
                current_R = R_t_lease[t_val-1] if t_val-1 < len(R_t_lease) else 0
                l_per_acre += current_R / ((1 + r_lease)**t_val)
            
            # Total Economic Value per acre
            simulated_tevs_per_acre[i] = (
                pv_timber_per_acre + 
                pvc_per_acre_val + 
                pe_per_acre + 
                l_per_acre
            )
        
        # Scale by total acres if provided
        if total_acres is not None and total_acres > 0:
            return simulated_tevs_per_acre * total_acres
        else:
            return simulated_tevs_per_acre
    
    def create_economic_parameters(
        self,
        biomass_stats: Dict,
        emissions_stats: Dict,
        water_yield_stats: Optional[Dict] = None,
        user_params: Optional[Dict[str, Dict]] = None,
        carbon_credit_price: float = 10.0
    ) -> Dict[str, Dict]:
        """
        Create complete economic parameters by merging calculated and user-specified values.
        
        Args:
            biomass_stats: Results from Script 1 (biomass statistics)
            emissions_stats: Results from Script 2 (carbon emissions)
            water_yield_stats: Results from Script 4 (water yield) - optional
            user_params: User-specified economic parameters per zone
            carbon_credit_price: Price per ton CO2e
            
        Returns:
            Dictionary of complete parameters for each zone
        """
        economic_params = {}
        
        for zone_name in biomass_stats.keys():
            # Get calculated values
            biomass_data = biomass_stats[zone_name]
            emissions_data = emissions_stats.get(zone_name, {})
            
            # Extract biomass (V_t)
            agb_per_acre = biomass_data.get('AGB_per_acre_tons', 0)
            std_agb = biomass_data.get('AGB_StdDev_per_acre_tons', 0)
            
            # Extract emissions (pvc_per_acre)
            pvc_mean = emissions_data.get('CO2_mean_tons_per_acre', 0)
            pvc_std = emissions_data.get('CO2_std_tons_per_acre', 0)
            pvc_mean_adj = pvc_mean * carbon_credit_price
            pvc_std_adj = pvc_std * carbon_credit_price
            
            # Extract water yield (water_quality_value) if available
            if water_yield_stats and zone_name.replace('_LLP', '') in water_yield_stats:
                water_data = water_yield_stats[zone_name.replace('_LLP', '')]
                # Check if water_data is not None and has the required keys
                if water_data and isinstance(water_data, dict):
                    water_value = water_data.get('water_yield_per_acre_usd', 0)
                    water_std = water_data.get('water_yield_std_per_acre_usd', water_value * 0.1)
                else:
                    # Water yield calculation failed, use user-provided or default
                    water_value = 100.0
                    water_std = 2.0
            else:
                # Use user-provided or default
                water_value = 100.0
                water_std = 2.0
            
            # Get user-specified parameters for this zone
            if user_params and zone_name in user_params:
                zone_user_params = user_params[zone_name]
            else:
                # Use defaults
                zone_user_params = get_default_params_for_zone(zone_name)
            
            # Merge calculated and user-specified parameters
            zone_params = {
                # Calculated values (from scripts)
                'V_t': (agb_per_acre, std_agb),
                'pvc_per_acre': (pvc_mean_adj, pvc_std_adj),
                'water_quality_value': (water_value, water_std),
                
                # User-specified values
                'E_Pt': zone_user_params.get('E_Pt', (25.0, 3.0)),
                'g': zone_user_params.get('g', (200.0, 30.0)),
                'endangered_species_WTP': zone_user_params.get('endangered_species_WTP', (13.37, 2.0)),
                'R_t_lease': zone_user_params.get('R_t_lease', [200, 150, 100, 50, 25]),
                'T_lease': zone_user_params.get('T_lease', 5),
                'r_lease': zone_user_params.get('r_lease', 0.05),
                'epsilon_t': zone_user_params.get('epsilon_t', 0)
            }
            
            economic_params[zone_name] = zone_params
        
        return economic_params
    
    def run_monte_carlo(
        self,
        biomass_stats: Dict,
        emissions_stats: Dict,
        case_acres: Dict,
        water_yield_stats: Optional[Dict] = None,
        user_params: Optional[Dict[str, Dict]] = None,
        carbon_credit_price: float = 10.0,
        num_simulations: int = 10000
    ) -> pd.DataFrame:
        """
        Run Monte Carlo simulations for all cases.
        
        Args:
            biomass_stats: Results from Script 1
            emissions_stats: Results from Script 2
            case_acres: Dictionary of acres for each case
            water_yield_stats: Results from Script 4 (optional)
            user_params: User-specified economic parameters per zone
            carbon_credit_price: Price per ton CO2e
            num_simulations: Number of simulations
            
        Returns:
            DataFrame with summary statistics
        """
        # Create complete economic parameters
        economic_params = self.create_economic_parameters(
            biomass_stats=biomass_stats,
            emissions_stats=emissions_stats,
            water_yield_stats=water_yield_stats,
            user_params=user_params,
            carbon_credit_price=carbon_credit_price
        )
        
        all_results = []
        self.simulation_results = []
        
        for zone_name, zone_params in economic_params.items():
            acres = case_acres.get(zone_name, 0)
            
            if acres == 0:
                print(f"Warning: No acreage specified for {zone_name}, skipping")
                continue
            
            print(f"  Running {num_simulations:,} simulations for {zone_name}...")
            
            # Run simulation
            results = self.calculate_tev(zone_params, acres, num_simulations)
            
            # Store summary statistics
            all_results.append({
                'Case': zone_name,
                'Acres': acres,
                'Mean_TEV': np.mean(results),
                'Std_TEV': np.std(results),
                'Median_TEV': np.median(results),
                'Q25_TEV': np.percentile(results, 25),
                'Q75_TEV': np.percentile(results, 75),
                'Min_TEV': np.min(results),
                'Max_TEV': np.max(results)
            })
            
            # Store full results for plotting
            self.simulation_results.append({
                'Case': zone_name,
                'TEV': results
            })
        
        self.results = pd.DataFrame(all_results)
        return self.results
    
    def plot_distributions(
        self,
        title: str = 'Distribution of Total Economic Value (TEV)',
        figsize: Tuple[int, int] = (14, 7)
    ):
        """Plot TEV distributions as histograms."""
        if self.simulation_results is None:
            print("No simulation results available. Run Monte Carlo first.")
            return
        
        plt.figure(figsize=figsize)
        
        for entry in self.simulation_results:
            sns.histplot(
                entry['TEV'],
                kde=True,
                label=entry['Case'],
                stat='density',
                alpha=0.5
            )
        
        plt.title(title)
        plt.xlabel('Total Economic Value ($)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis in thousands
        formatter = ticker.FuncFormatter(lambda x, pos: f'${x/1000:,.0f}k')
        plt.gca().xaxis.set_major_formatter(formatter)
        
        plt.tight_layout()
        plt.show()
    
    def plot_boxplots(
        self,
        title: str = 'Comparison of Total Economic Value (TEV)',
        figsize: Tuple[int, int] = (12, 7)
    ):
        """Plot TEV comparison as boxplots."""
        if self.simulation_results is None:
            print("No simulation results available. Run Monte Carlo first.")
            return
        
        # Convert to long DataFrame
        plot_data = []
        for entry in self.simulation_results:
            for val in entry['TEV']:
                plot_data.append({
                    'Case': entry['Case'],
                    'TEV': val
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        plt.figure(figsize=figsize)
        sns.boxplot(data=plot_df, x='Case', y='TEV')
        
        plt.title(title)
        plt.xlabel('Scenario (Case)')
        plt.ylabel('Total Economic Value ($)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Format y-axis in thousands
        formatter = ticker.FuncFormatter(lambda x, pos: f'${x/1000:,.0f}k')
        plt.gca().yaxis.set_major_formatter(formatter)
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, filepath: str):
        """Export results to CSV."""
        if self.results is not None:
            self.results.to_csv(filepath, index=False)
            print(f"Results exported to {filepath}")
        else:
            print("No results to export. Run Monte Carlo first.")


def get_default_params_for_zone(zone_name: str) -> Dict:
    """
    Get default economic parameters for a zone.
    
    Args:
        zone_name: Name of the zone
        
    Returns:
        Dictionary of default parameters
    """
    # Default parameters based on zone characteristics
    if 'CS1' in zone_name:
        # Severe wildfire scenario
        return {
            'E_Pt': (7.50, 1),      # Low stumpage price
            'g': (375, 50),          # High regeneration cost
            'endangered_species_WTP': (13.37 * 0.1, 1),  # Low species value
            'R_t_lease': [50, 20, 0, 0, 0],  # Declining lease revenues
            'T_lease': 5,
            'r_lease': 0.06
        }
    elif 'CS2' in zone_name:
        # Mitigated wildfire scenario
        return {
            'E_Pt': (21, 3),
            'g': (200, 30),
            'endangered_species_WTP': (13.37 * 0.5, 2),
            'R_t_lease': [200, 100, 50, 20, 10],
            'T_lease': 5,
            'r_lease': 0.055
        }
    elif 'CS3' in zone_name:
        # Healthy/prescribed burn scenario
        return {
            'E_Pt': (36, 5),
            'g': (50, 10),
            'endangered_species_WTP': (13.37 * 1, 3),
            'R_t_lease': [700, 700, 700, 700, 700],
            'T_lease': 5,
            'r_lease': 0.05
        }
    else:
        # Generic defaults
        return {
            'E_Pt': (25, 3),
            'g': (200, 30),
            'endangered_species_WTP': (13.37, 2),
            'R_t_lease': [200, 200, 200, 200, 200],
            'T_lease': 5,
            'r_lease': 0.05
        }


# Helper function to create user parameter template
def create_user_params_template(zone_names: list) -> Dict[str, Dict]:
    """
    Create a template for user parameters.
    
    Args:
        zone_names: List of zone names
        
    Returns:
        Dictionary template that users can fill in
    """
    template = {}
    
    for zone_name in zone_names:
        template[zone_name] = {
            'E_Pt': (25.0, 3.0),      # Stumpage price ($/ton) ± std
            'g': (200.0, 30.0),        # Regeneration cost ($/acre) ± std
            'endangered_species_WTP': (13.37, 2.0),  # Willingness-to-pay ($/acre) ± std
            'R_t_lease': [200, 200, 200, 200, 200],  # Annual lease revenues ($/acre/year)
            'T_lease': 5,              # Lease period (years)
            'r_lease': 0.05,           # Discount rate
            'epsilon_t': 0             # Price shock (optional)
        }
    
    return template