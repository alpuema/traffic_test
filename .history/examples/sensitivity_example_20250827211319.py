#!/usr/bin/env python3
"""
Sensitivity Analysis Example

This example demonstrates how to use the sensitivity analysis wrapper
to easily test different parameter configurations and identify optimal
settings for the traffic light optimization.

Usage:
    python examples/sensitivity_example.py

Author: Traffic Optimization System
Date: August 2025
"""

import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_quick_sensitivity_demo():
    """Run a quick sensitivity analysis demonstration."""
    
    print("🔬 Quick Sensitivity Analysis Demo")
    print("=" * 60)
    print("This demo shows how to easily test different parameter values")
    print("to find optimal settings for your traffic scenarios.")
    print()
    
    try:
        # Import the sensitivity analysis module
        from src.sensitivity_analysis import run_simple_parameter_sweep, run_sensitivity_analysis
        
        print("✅ Successfully imported sensitivity analysis module")
        
        # Define base configuration
        base_config = {
            'grid_size': 3,
            'n_vehicles': 25,  # Smaller for quick demo
            'simulation_time': 300,  # Shorter simulation for speed
            'traffic_pattern': 'commuter'
        }
        
        print(f"\n📋 Base Configuration:")
        for key, value in base_config.items():
            print(f"   {key}: {value}")
        
        print(f"\n🧪 Running Single-Parameter Analysis: n_ants")
        print("Testing different numbers of ants: [5, 10, 15, 20]")
        
        # Single parameter sweep
        results = run_simple_parameter_sweep(
            parameter_name='n_ants',
            parameter_values=[5, 10, 15, 20],
            base_config=base_config,
            n_replications=2  # Few replications for quick demo
        )
        
        print(f"✅ Single-parameter analysis completed!")
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
        print(f"       n_replications=3")
        print(f"   )")
        
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


def run_comprehensive_analysis_example():
    """Example of comprehensive multi-parameter sensitivity analysis."""
    
    print("\n🔬 Comprehensive Multi-Parameter Analysis Example")
    print("=" * 60)
    print("This would test multiple parameters simultaneously.")
    print("⚠️  This is computationally intensive - not run in this demo.")
    print()
    
    # Example configuration for comprehensive analysis
    base_config = {
        'grid_size': 3,
        'n_vehicles': 50,
        'simulation_time': 600,
        'traffic_pattern': 'commuter'
    }
    
    parameter_ranges = {
        'n_ants': [10, 20, 30, 40],
        'n_iterations': [5, 10, 15, 20],
        'n_vehicles': [30, 40, 50, 60]
    }
    
    print("📋 Configuration:")
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
    print("🚀 Sensitivity Analysis Examples")
    print()
    
    # Run quick demo
    success = run_quick_sensitivity_demo()
    
    if success:
        # Show comprehensive example
        run_comprehensive_analysis_example()
        
        # Show all features
        show_sensitivity_analysis_features()
        
        print(f"\n🎉 Sensitivity Analysis Demo Complete!")
        print(f"💡 Use these tools to optimize your traffic light configurations!")
    else:
        print(f"\n❌ Demo failed. Please check your setup.")
        sys.exit(1)
