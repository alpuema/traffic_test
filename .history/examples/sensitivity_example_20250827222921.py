#!/usr/bin/env python3
"""
Sensitivity Analysis Example

This example demonstrates how to use the sensitivity analysis wrapper
to easily test different parameter configurations and identify optimal
settings for the traffic light optimization.

EASY CONFIGURATION: Modify the settings below to customize your analysis!

🔧 NEW: ACO PARAMETERS ARE NOW CONFIGURABLE!
You can now specify baseline ACO algorithm parameters (n_ants, n_iterations, etc.) 
in BASE_CONFIG and test different values in sensitivity analysis.

Available ACO Parameters:
• n_ants: Number of ants per iteration (5-50, more = better exploration but slower)
• n_iterations: Number of optimization iterations (3-100, more = better convergence but slower)  
• evaporation_rate: Pheromone decay rate (0.01-0.2, lower = longer memory)
• exploration_rate: Random exploration probability (0.1-0.4, higher = more exploration)
• alpha: Stop penalty weight (1-100, higher = penalize individual long stops more)

Usage:
    python examples/sensitivity_example.py

Author: Traffic Optimization System
Date: August 2025
"""

import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# 🎯 EASY CONFIGURATION SECTION - MODIFY THESE SETTINGS!
# ============================================================================

# ----------------------------------------
# ANALYSIS TYPE SELECTION
# ----------------------------------------
ANALYSIS_TYPE = "single_parameter"  # Options: "quick_demo", "single_parameter", "multi_parameter"

# ----------------------------------------
# BASE SCENARIO CONFIGURATION
# ----------------------------------------
BASE_CONFIG = {
    # Traffic scenario parameters
    'grid_size': 3,                    # Network size: 2, 3, or 4 (bigger = more complex)
    'n_vehicles': 25,                  # Number of vehicles: 10-100 (more = realistic but slower)
    'simulation_time': 1200,           # Simulation duration in seconds: 1200+ ensures car completion
    'traffic_pattern': 'commuter',     # Pattern: 'commuter', 'industrial', 'random'
    
    # ACO algorithm parameters (baseline values for sensitivity analysis)
    'n_ants': 15,                      # Number of ants per iteration (5-50, more = better exploration)
    'n_iterations': 12,                # Number of optimization iterations (3-100, more = better convergence)
    'evaporation_rate': 0.05,          # Pheromone evaporation (0.01-0.2, lower = more memory)
    'exploration_rate': 0.20,          # Random exploration probability (0.1-0.4, higher = more exploration)
    'alpha': 25.0                      # Stop penalty weight in cost function (1-100, higher = penalize long stops more)
}

# ----------------------------------------
# QUICK DEMO SETTINGS
# ----------------------------------------
QUICK_DEMO_CONFIG = {
    'parameter': 'n_iterations',       # Parameter to test: 'n_ants', 'n_iterations', 'evaporation_rate', 'alpha', 'n_vehicles'
    'values': [8, 12, 16, 20],         # Values to test for the parameter
    'replications': 2                  # Number of runs per value (more = better statistics)
}

# ----------------------------------------
# SINGLE PARAMETER ANALYSIS SETTINGS  
# ----------------------------------------
SINGLE_PARAM_CONFIG = {
    'parameter': 'n_ants',             # Parameter to analyze
    'values': [10, 50, 20, 25],        # Values to test
    'replications': 3                  # Replications per value
}

# ----------------------------------------
# MULTI-PARAMETER ANALYSIS SETTINGS
# ----------------------------------------
MULTI_PARAM_CONFIG = {
    'parameter_ranges': {
        # ACO Algorithm Parameters
        'n_ants': [10, 15, 20, 25],        # Different numbers of ants to test
        'n_iterations': [8, 12, 16],       # Different iteration counts
        # 'evaporation_rate': [0.03, 0.05, 0.1],  # Pheromone evaporation rates
        # 'exploration_rate': [0.15, 0.20, 0.25], # Random exploration rates
        # 'alpha': [15.0, 25.0, 35.0],            # Stop penalty weights
        
        # Traffic Scenario Parameters  
        # 'n_vehicles': [20, 25, 30],        # Different vehicle counts
        # 'grid_size': [2, 3, 4],            # Different grid sizes
        # 'simulation_time': [900, 1200, 1500], # Different simulation durations
    },
    'replications': 2,                 # Replications per combination
    'parallel': True,                  # Run in parallel for speed
    'max_workers': 4                   # Max parallel processes (adjust based on your CPU)
}

# ----------------------------------------
# DISPLAY OPTIONS
# ----------------------------------------
DISPLAY_CONFIG = {
    'show_progress': True,             # Show detailed progress during analysis
    'show_individual_plots': False,   # Show plots for each optimization run (disabled for sensitivity analysis)
    'show_summary_plots': True,       # Generate and display final summary plots
    'save_results': True,              # Save detailed results to files
    'output_dir': None                 # Output directory (None = auto-generate)
}

# ============================================================================
# 🔧 ANALYSIS FUNCTIONS (Usually no need to modify below this line)
# ============================================================================

def run_quick_sensitivity_demo():
    """Run a quick sensitivity analysis demonstration."""
    
    print("🔬 Quick Sensitivity Analysis Demo")
    print("=" * 70)
    print("Welcome to sensitivity analysis! This tool helps you find optimal settings.")
    print()
    print("📊 What is Sensitivity Analysis?")
    print("   • Tests different parameter values systematically")
    print("   • Finds which settings give the best performance")
    print("   • Shows how sensitive results are to parameter changes")
    print("   • Generates statistics and visualizations")
    print()
    
    # Display current configuration
    print("🎯 Current Configuration:")
    print(f"   Parameter to test: {QUICK_DEMO_CONFIG['parameter']}")
    print(f"   Values to test: {QUICK_DEMO_CONFIG['values']}")
    print(f"   Replications per value: {QUICK_DEMO_CONFIG['replications']}")
    print(f"   Total runs: {len(QUICK_DEMO_CONFIG['values']) * QUICK_DEMO_CONFIG['replications']}")
    print()
    
    # Ask user if they want to continue
    proceed = input("Ready to run sensitivity analysis? (y/n) [default: y]: ").strip().lower()
    if proceed == 'n':
        print("👋 No problem! Run again when you want to explore optimal settings.")
        print("💡 Tip: Edit the configuration at the top of this file to customize!")
        return True
    
    print("\n⚙️  BASE SCENARIO CONFIGURATION")
    print("-" * 40)
    
    print("📋 Base Scenario (what stays constant):")
    for key, value in BASE_CONFIG.items():
        if key == 'grid_size':
            description = f"{value}x{value} grid ({value**2} intersections)"
        elif key == 'n_vehicles':
            traffic_level = 'light' if value < 25 else 'moderate' if value < 50 else 'heavy'
            description = f"{value} vehicles ({traffic_level} traffic)"
        elif key == 'simulation_time':
            description = f"{value} seconds ({value/60:.1f} minutes)"
        elif key == 'traffic_pattern':
            description = f"{value} pattern (realistic traffic flow)"
        else:
            description = str(value)
        
        print(f"   {key.replace('_', ' ').title()}: {description}")
    
    print(f"\n🔬 Parameter Analysis:")
    print(f"   • Testing parameter: {QUICK_DEMO_CONFIG['parameter']}")
    print(f"   • Testing values: {QUICK_DEMO_CONFIG['values']}")
    print(f"   • Replications each: {QUICK_DEMO_CONFIG['replications']}")
    
    total_runs = len(QUICK_DEMO_CONFIG['values']) * QUICK_DEMO_CONFIG['replications']
    estimated_time = total_runs * BASE_CONFIG['simulation_time'] / 60 / 20  # Rough estimate
    print(f"   • Total runs: {total_runs}")
    print(f"   • Estimated time: ~{estimated_time:.1f} minutes")
    print()
    
    try:
        # Import the sensitivity analysis module
        from src.sensitivity_analysis import run_simple_parameter_sweep
        
        print("✅ Successfully imported sensitivity analysis module")
        
        print(f"\n🧪 Running Parameter Analysis...")
        
        # Single parameter sweep using configuration
        results = run_simple_parameter_sweep(
            parameter_name=QUICK_DEMO_CONFIG['parameter'],
            parameter_values=QUICK_DEMO_CONFIG['values'],
            base_config=BASE_CONFIG,
            n_replications=QUICK_DEMO_CONFIG['replications'],
            show_individual_plots=DISPLAY_CONFIG['show_individual_plots'],
            show_final_plot=DISPLAY_CONFIG['show_summary_plots']
        )
        
        print(f"✅ Parameter analysis completed!")
        print(f"   Best configuration found:")
        best_config = results['summary']['best_configuration']
        print(f"   Parameters: {best_config['parameters']}")
        print(f"   Average cost: {best_config['mean_cost']:.1f} seconds")
        print(f"   Average improvement: {best_config['mean_improvement']:.1f}%")
        
        print(f"\n📊 Results saved to: {results['output_dir']}")
        print(f"📈 Visualization plots: {len(results['plot_files'])} files")
        
        # Show example of multi-parameter analysis (without running it)
        print(f"\n💡 For Multi-Parameter Analysis, you could run:")
        print(f"")
        print(f"   param_ranges = {{")
        print(f"       'n_ants': [10, 20, 30],")
        print(f"       'n_iterations': [5, 10, 15],")
        print(f"       'n_vehicles': [20, 30, 40]")
        print(f"   }}")
        print(f"   ")
        print(f"   results = run_sensitivity_analysis(")
        print(f"       parameter_ranges=param_ranges,")
        print(f"       base_config=base_config,")
        
        # Show configuration for other analysis types
        print(f"\n💡 Want to try other analysis types?")
        print(f"   Edit the configuration at the top of this file:")
        print(f"   • ANALYSIS_TYPE = 'single_parameter' for detailed single parameter")
        print(f"   • ANALYSIS_TYPE = 'multi_parameter' for multiple parameters")
        print(f"   • Modify BASE_CONFIG for different scenarios")
        print(f"   • Adjust parameter ranges and replications")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running from the project root directory")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_single_parameter_analysis():
    """Run detailed single parameter sensitivity analysis."""
    
    print("🔬 Single Parameter Sensitivity Analysis")
    print("=" * 60)
    print("Detailed analysis of one parameter with more values and replications.")
    print()
    
    try:
        from src.sensitivity_analysis import run_simple_parameter_sweep
        
        print("📋 Configuration:")
        print(f"   Parameter: {SINGLE_PARAM_CONFIG['parameter']}")
        print(f"   Values: {SINGLE_PARAM_CONFIG['values']}")
        print(f"   Replications: {SINGLE_PARAM_CONFIG['replications']}")
        
        total_runs = len(SINGLE_PARAM_CONFIG['values']) * SINGLE_PARAM_CONFIG['replications']
        print(f"   Total runs: {total_runs}")
        
        proceed = input(f"\nThis will take ~{total_runs * 2} minutes. Continue? (y/n) [default: y]: ").strip().lower()
        if proceed == 'n':
            print("Analysis cancelled.")
            return True
        
        print(f"\n🧪 Running detailed parameter analysis...")
        
        results = run_simple_parameter_sweep(
            parameter_name=SINGLE_PARAM_CONFIG['parameter'],
            parameter_values=SINGLE_PARAM_CONFIG['values'],
            base_config=BASE_CONFIG,
            n_replications=SINGLE_PARAM_CONFIG['replications'],
            show_individual_plots=DISPLAY_CONFIG['show_individual_plots'],
            show_final_plot=DISPLAY_CONFIG['show_summary_plots']
        )
        
        print(f"✅ Detailed analysis completed!")
        print(f"   Best configuration: {results['summary']['best_configuration']['parameters']}")
        print(f"   Results directory: {results['output_dir']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False


def run_multi_parameter_analysis():
    """Run comprehensive multi-parameter sensitivity analysis."""
    
    print("🔬 Multi-Parameter Sensitivity Analysis")
    print("=" * 60)
    print("Comprehensive analysis testing combinations of multiple parameters.")
    print()
    
    try:
        from src.sensitivity_analysis import run_sensitivity_analysis
        
        print("📋 Configuration:")
        print(f"   Parameters: {list(MULTI_PARAM_CONFIG['parameter_ranges'].keys())}")
        for param, values in MULTI_PARAM_CONFIG['parameter_ranges'].items():
            print(f"     {param}: {values}")
        print(f"   Replications per combination: {MULTI_PARAM_CONFIG['replications']}")
        
        total_combinations = 1
        for values in MULTI_PARAM_CONFIG['parameter_ranges'].values():
            total_combinations *= len(values)
        
        total_runs = total_combinations * MULTI_PARAM_CONFIG['replications']
        estimated_time = total_runs * BASE_CONFIG['simulation_time'] / 60 / 10  # Rough estimate
        
        print(f"   Total combinations: {total_combinations}")
        print(f"   Total runs: {total_runs}")
        print(f"   Estimated time: ~{estimated_time:.0f} minutes")
        print(f"   Parallel processing: {'Yes' if MULTI_PARAM_CONFIG['parallel'] else 'No'}")
        
        proceed = input(f"\n⚠️  This is a comprehensive analysis. Continue? (y/n) [default: n]: ").strip().lower()
        if proceed != 'y':
            print("Analysis cancelled. Consider starting with single parameter analysis.")
            return True
        
        print(f"\n🧪 Running multi-parameter analysis...")
        
        results = run_sensitivity_analysis(
            parameter_ranges=MULTI_PARAM_CONFIG['parameter_ranges'],
            base_config=BASE_CONFIG,
            n_replications=MULTI_PARAM_CONFIG['replications'],
            parallel=MULTI_PARAM_CONFIG['parallel'],
            max_workers=MULTI_PARAM_CONFIG.get('max_workers'),
            output_dir=DISPLAY_CONFIG['output_dir'],
            show_individual_plots=DISPLAY_CONFIG['show_individual_plots'],
            show_final_plot=DISPLAY_CONFIG['show_summary_plots']
        )
        
        print(f"✅ Multi-parameter analysis completed!")
        print(f"   Best configuration: {results['summary']['best_configuration']['parameters']}")
        print(f"   Results directory: {results['output_dir']}")
        print(f"   Generated plots: {len(results['plot_files'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False


def run_comprehensive_analysis_example():
    """Show example of comprehensive multi-parameter sensitivity analysis (informational only)."""
    
    print("\n🔬 Multi-Parameter Analysis Information")
    print("=" * 60)
    print("This shows what a comprehensive analysis would involve.")
    print("⚠️  Set ANALYSIS_TYPE = 'multi_parameter' to actually run this.")
    print()
    
    # Show example configuration with ACO parameters
    example_config = {
        'parameter_ranges': {
            'n_ants': [10, 15, 20, 25],            # ACO: Number of ants
            'n_iterations': [8, 12, 16],           # ACO: Optimization iterations  
            'evaporation_rate': [0.03, 0.05, 0.1], # ACO: Pheromone decay
            'alpha': [15.0, 25.0, 35.0],           # ACO: Stop penalty weight
            'n_vehicles': [20, 25, 30]             # Scenario: Traffic volume
        },
        'replications': 2
    }
    
    print("📋 Example Configuration (ACO + Scenario Parameters):")
    aco_params = ['n_ants', 'n_iterations', 'evaporation_rate', 'alpha']
    scenario_params = ['n_vehicles', 'grid_size', 'simulation_time']
    
    print("   🐜 ACO Algorithm Parameters:")
    for param, values in example_config['parameter_ranges'].items():
        if param in aco_params:
            print(f"      {param}: {values}")
    
    print("   🚦 Traffic Scenario Parameters:")  
    for param, values in example_config['parameter_ranges'].items():
        if param in scenario_params:
            print(f"      {param}: {values}")
    
    print(f"   🔄 Replications per combination: {example_config['replications']}")
    
    total_combinations = 1
    for values in example_config['parameter_ranges'].values():
        total_combinations *= len(values)
    
    total_runs = total_combinations * example_config['replications']
    
    print(f"   📊 Total combinations: {total_combinations}")
    print(f"   🎯 Total optimization runs: {total_runs}")
    print(f"   ⏱️  Estimated time: {total_runs * 3:.0f} minutes (~{total_runs * 3 / 60:.1f} hours)")
    
    print(f"\n💻 To run comprehensive ACO parameter analysis:")
    print(f"   1. Edit ANALYSIS_TYPE = 'multi_parameter' at top of file")
    print(f"   2. Uncomment desired parameters in MULTI_PARAM_CONFIG")
    print(f"   3. Adjust BASE_CONFIG for your baseline ACO settings")
    print(f"   4. Consider starting with fewer parameter values")
    print(f"   5. Enable parallel processing for speed")
    
    return True
    print(f"   Base config: {base_config}")
    print(f"   Parameter ranges: {parameter_ranges}")
    
    total_combinations = 1
    for param, values in parameter_ranges.items():
        total_combinations *= len(values)
    
    print(f"   Total parameter combinations: {total_combinations}")
    print(f"   With 3 replications each: {total_combinations * 3} total runs")
    print(f"   Estimated time (5 min/run): {total_combinations * 3 * 5:.0f} minutes")
    
    print("\n💻 To run this analysis:")
    print("   from src.sensitivity_analysis import run_sensitivity_analysis")
    print("   results = run_sensitivity_analysis(parameter_ranges, base_config)")
    
    return True


def show_sensitivity_analysis_features():
    """Show all available features of the sensitivity analysis module."""
    
    print("\n🎯 Sensitivity Analysis Features")
    print("=" * 60)
    
    features = [
        ("🔄 Single Parameter Sweep", "Test one parameter across multiple values"),
        ("🌐 Multi-Parameter Analysis", "Test combinations of multiple parameters"),
        ("📊 Statistical Analysis", "Mean, std dev, min/max across replications"),
        ("📈 Automatic Visualization", "Generate plots showing parameter effects"),
        ("⚡ Parallel Processing", "Run multiple configurations simultaneously"),
        ("💾 Result Storage", "Save detailed results in JSON format"),
        ("🎯 Best Configuration", "Automatically identify optimal settings"),
        ("📋 Summary Reports", "Comprehensive analysis summaries")
    ]
    
    for feature, description in features:
        print(f"   {feature:<25} {description}")
    
    print(f"\n📚 Key Functions:")
    print(f"   • run_simple_parameter_sweep()  - Single parameter analysis")
    print(f"   • run_sensitivity_analysis()    - Multi-parameter analysis") 
    print(f"   • Automatic plot generation      - Visual analysis results")
    print(f"   • Statistical summaries          - Compare configurations")
    
    print(f"\n🔧 Supported Parameters:")
    supported_params = [
        'n_ants', 'n_iterations', 'grid_size', 'n_vehicles', 
        'simulation_time', 'traffic_pattern'
    ]
    for param in supported_params:
        print(f"   • {param}")
    
    return True


if __name__ == "__main__":
    print("🚀 Sensitivity Analysis - Easy Configuration")
    print("=" * 60)
    print(f"Current analysis type: {ANALYSIS_TYPE}")
    print("💡 Edit the configuration at the top of this file to customize!")
    print()
    
    # Run analysis based on configuration
    if ANALYSIS_TYPE == "quick_demo":
        print("Running Quick Demo Analysis...")
        success = run_quick_sensitivity_demo()
        
    elif ANALYSIS_TYPE == "single_parameter":
        print("Running Single Parameter Analysis...")
        success = run_single_parameter_analysis()
        
    elif ANALYSIS_TYPE == "multi_parameter":
        print("Running Multi-Parameter Analysis...")
        success = run_multi_parameter_analysis()
        
    else:
        print(f"❌ Unknown analysis type: {ANALYSIS_TYPE}")
        print("Valid options: 'quick_demo', 'single_parameter', 'multi_parameter'")
        print("\n📚 Available Analysis Types:")
        print("   • quick_demo: Fast demonstration with preset values")
        print("   • single_parameter: Detailed analysis of one parameter")  
        print("   • multi_parameter: Comprehensive analysis of multiple parameters")
        print("\n💡 Edit ANALYSIS_TYPE in the configuration section at the top of this file")
        success = False
    
    if success:
        # Show information about other analysis types
        if ANALYSIS_TYPE == "quick_demo":
            show_sensitivity_analysis_features()
            run_comprehensive_analysis_example()
        
        print(f"\n🎉 Analysis Complete!")
        print(f"\n💡 Next Steps:")
        print(f"   • Edit the configuration section to try different settings")
        print(f"   • Change ANALYSIS_TYPE to explore other analysis modes")
        print(f"   • Modify BASE_CONFIG for different traffic scenarios")
        print(f"   • Adjust parameter ranges and replications as needed")
        print(f"\n� Configuration sections in this file:")
        print(f"   🎯 ANALYSIS_TYPE - Choose analysis mode")
        print(f"   📋 BASE_CONFIG - Scenario settings")
        print(f"   🔬 QUICK_DEMO_CONFIG - Quick demo parameters")
        print(f"   📊 SINGLE_PARAM_CONFIG - Detailed single parameter")
        print(f"   🌐 MULTI_PARAM_CONFIG - Multi-parameter analysis")
        print(f"   📈 DISPLAY_CONFIG - Output and visualization options")
        
    else:
        print(f"\n❌ Analysis failed. Please check your configuration and setup.")
        sys.exit(1)
