#!/usr/bin/env python3
"""
Quick Start Example for Traffic Light Optimization

This example demonstrates how to use the traffic light optimization system
with the new organized structure. It shows the most common usage patterns
and serves as a template for your own optimization runs.

Usage:
    python examples/quick_start.py

Author: Traffic Optimization System
Date: August 2025
"""

import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Run a quick start example demonstrating key features."""
    
    print("🚀 Traffic Light Optimization - Quick Start")
    print("=" * 60)
    print("This example demonstrates the key features of the system:")
    print("1. Traffic scenario generation")
    print("2. ACO optimization")
    print("3. Results visualization")
    print()
    
    try:
        # Import from src directory
        from src.simplified_traffic import generate_network_and_routes, list_available_patterns
        from src.optimization.simple_aco import run_simplified_aco_optimization
        
        print("✅ Successfully imported optimization modules")
        
        # Show available traffic patterns
        print("\n📋 Available Traffic Patterns:")
        patterns = list_available_patterns()
        for i, pattern in enumerate(patterns, 1):
            print(f"   {i}. {pattern}")
        
        # Configuration for a quick demo
        config = {
            'grid_size': 3,
            'n_vehicles': 25,
            'simulation_time': 400,
            'traffic_pattern': 'commuter',  # realistic pattern
            'seed': 42  # for reproducible results
        }
        
        print(f"\n⚙️  Configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        # Step 1: Generate traffic scenario
        print(f"\n🏗️  Step 1: Generating traffic scenario...")
        
        scenario_result = generate_network_and_routes(
            grid_size=config['grid_size'],
            n_vehicles=config['n_vehicles'],
            sim_time=config['simulation_time'],
            pattern=config['traffic_pattern'],
            seed=config['seed']
        )
        
        if not scenario_result['success']:
            print(f"❌ Failed to generate scenario: {scenario_result['error']}")
            return False
            
        print("✅ Traffic scenario generated successfully")
        print(f"   Network: {scenario_result['files']['network']}")
        print(f"   Routes: {scenario_result['files']['routes']}")
        
        # Step 2: Run ACO optimization
        print(f"\n🐜 Step 2: Running ACO optimization...")
        
        aco_config = {
            'grid_size': config['grid_size'],
            'n_vehicles': config['n_vehicles'],
            'simulation_time': config['simulation_time'],
            'n_ants': 15,  # smaller for quick demo
            'n_iterations': 5  # fewer iterations for quick demo
        }
        
        optimization_result = run_simplified_aco_optimization(aco_config)
        
        if not optimization_result['success']:
            print(f"❌ Optimization failed: {optimization_result['error']}")
            return False
            
        print("✅ Optimization completed successfully!")
        print(f"   Best cost: {optimization_result['best_cost']:.1f} seconds")
        print(f"   Improvement: {optimization_result.get('improvement_pct', 'N/A')}%")
        
        # Step 3: Display results
        print(f"\n📊 Step 3: Results Summary:")
        print(f"   Initial average delay: {optimization_result.get('initial_cost', 'N/A')} seconds")
        print(f"   Optimized average delay: {optimization_result['best_cost']:.1f} seconds")
        
        if 'plot_file' in optimization_result:
            print(f"   Progress plot: {optimization_result['plot_file']}")
            
        print(f"\n🎯 Quick Start Complete!")
        print(f"   ✓ Traffic scenario generated")
        print(f"   ✓ ACO optimization completed")  
        print(f"   ✓ Results saved to 'results/' directory")
        print(f"\n💡 Next Steps:")
        print(f"   • Try different traffic patterns: {', '.join(patterns[:3])}...")
        print(f"   • Experiment with grid sizes: 2x2, 3x3, 4x4")
        print(f"   • Run longer optimizations with more ants/iterations")
        print(f"   • Use 'python main.py' for interactive configuration")
        
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

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
