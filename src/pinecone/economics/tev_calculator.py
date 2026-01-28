"""
Total Economic Value (TEV) calculator for prescribed fire analysis.
Integrates biomass stats and carbon emissions to calculate economic value.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from typing import Dict, Optional, Tuple


class TEVCalculator:
    """
    Calculate Total Economic Value (TEV) for prescribed fire scenarios.
    
    Components:
    - Timber Value (PV)
    - Carbon Benefits (PVC)
    - Ecosystem Services (PE)
    - Land Value for Hunting/Leases (L)
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
    
    def create_input_dataframe(
        self,
        biomass_stats: Dict,
        emissions_stats: Dict,
        carbon_credit_price: float = 10.0
    ) -> pd.DataFrame:
        """
        Create input DataFrame from Scripts 1 & 2 outputs.
        
        Args:
            biomass_stats: Results from Script 1 (biomass statistics)
            emissions_stats: Results from Script 2 (carbon emissions)
            carbon_credit_price: Price per ton CO2e
            
        Returns:
            DataFrame ready for TEV calculation
        """
        data = []
        
        # Extract data from both scripts
        for zone_name in biomass_stats.keys():
            biomass_data = biomass_stats[zone_name]
            emissions_data = emissions_stats.get(zone_name, {})
            
            # Get values with safe defaults
            agb_per_acre = biomass_data.get('AGB_per_acre_tons', 0)
            std_agb = biomass_data.get('AGB_StdDev_per_acre_tons', 0)
            
            pvc_mean = emissions_data.get('CO2_mean_tons_per_acre', 0)
            pvc_std = emissions_data.get('CO2_std_tons_per_acre', 0)
            
            # Apply carbon credit multiplier
            pvc_mean_adj = pvc_mean * carbon_credit_price
            pvc_std_adj = pvc_std * carbon_credit_price
            
            data.append({
                'Source': zone_name,
                'agb_per_acre': agb_per_acre,
                'std_agb_per_acre': std_agb,
                'pvc_per_acre_mean': pvc_mean_adj,
                'pvc_per_acre_std': pvc_std_adj,
                'Method': 'ESA'  # Can be parameterized
            })
        
        return pd.DataFrame(data)
    
    def run_monte_carlo(
        self,
        input_df: pd.DataFrame,
        base_cases: Dict,
        case_acres: Dict,
        num_simulations: int = 10000
    ) -> pd.DataFrame:
        """
        Run Monte Carlo simulations for all cases.
        
        Args:
            input_df: DataFrame with biomass and emissions data
            base_cases: Dictionary of base parameters for each case
            case_acres: Dictionary of acres for each case
            num_simulations: Number of simulations
            
        Returns:
            DataFrame with summary statistics
        """
        all_results = []
        self.simulation_results = []
        
        for _, row in input_df.iterrows():
            case_name = row['Source']
            method = row['Method']
            
            # Get base parameters for this case
            if case_name not in base_cases:
                print(f"Warning: No base parameters for {case_name}, skipping")
                continue
            
            # Copy base parameters and update with biomass/emissions data
            case_params = base_cases[case_name].copy()
            case_params['V_t'] = (row['agb_per_acre'], row['std_agb_per_acre'])
            case_params['pvc_per_acre'] = (row['pvc_per_acre_mean'], row['pvc_per_acre_std'])
            
            acres = case_acres.get(case_name, 0)
            
            # Run simulation
            results = self.calculate_tev(case_params, acres, num_simulations)
            
            # Store summary statistics
            all_results.append({
                'Case': case_name,
                'Method': method,
                'Mean_TEV': np.mean(results),
                'Std_TEV': np.std(results),
                'Median_TEV': np.median(results),
                'Q25_TEV': np.percentile(results, 25),
                'Q75_TEV': np.percentile(results, 75)
            })
            
            # Store full results for plotting
            self.simulation_results.append({
                'Case': case_name,
                'Method': method,
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
        
        colors = {'OBIWAN': 'blue', 'ESA': 'green'}
        
        for entry in self.simulation_results:
            color = colors.get(entry['Method'], 'gray')
            sns.histplot(
                entry['TEV'],
                kde=True,
                color=color,
                label=f"{entry['Case']} - {entry['Method']}",
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
                    'Method': entry['Method'],
                    'TEV': val
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        plt.figure(figsize=figsize)
        
        colors = {'OBIWAN': 'blue', 'ESA': 'green'}
        sns.boxplot(data=plot_df, x='Case', y='TEV', hue='Method', palette=colors)
        
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


# Default economic parameters for different scenarios
DEFAULT_ECONOMIC_PARAMS = {
    "EIA_CS1_LLP": {
        'E_Pt': (7.50, 1),      # Stumpage price ($/ton)
        'g': (375, 50),          # Regeneration cost ($/acre)
        'water_quality_value': (110.56, 2.04),
        'endangered_species_WTP': (13.37 * 0.1, 1),
        'R_t_lease': [50, 20, 0, 0, 0],
        'T_lease': 5,
        'r_carbon': 0.05,
        'r_ecosystem': 0.05,
        'r_lease': 0.06
    },
    "EIA_CS2_LLP": {
        'E_Pt': (21, 3),
        'g': (200, 30),
        'water_quality_value': (100.16, 1.38),
        'endangered_species_WTP': (13.37 * 0.5, 2),
        'R_t_lease': [200, 100, 50, 20, 10],
        'T_lease': 5,
        'r_carbon': 0.05,
        'r_ecosystem': 0.05,
        'r_lease': 0.055
    },
    "EIA_CS3_LLP": {
        'E_Pt': (36, 5),
        'g': (50, 10),
        'water_quality_value': (120.01, 0.76),
        'endangered_species_WTP': (13.37 * 1, 3),
        'R_t_lease': [700, 700, 700, 700, 700],
        'T_lease': 5,
        'r_carbon': 0.05,
        'r_ecosystem': 0.05,
        'r_lease': 0.05
    }
}