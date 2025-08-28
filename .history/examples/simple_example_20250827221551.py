#!/usr/bin/env python3
"""
Simple ACO Optimization Example - Clean and Direct

This example demonstrates the Simple ACO system for traffic light optimization.
It runs a quick optimization to show the system in action.

Author: Traffic Optimization System
Date: August 2025
"""

import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_example():
    """Run a simple ACO optimization example."""
    
    print("🚀 Simple ACO Optimization Example")
    print("=" * 50)
    
    print("🎯 OBJECTIVE FUNCTION EXPLAINED:")
    print("   The ACO optimizes traffic light timing by minimizing:")
    print("   Cost = Average Travel Time + Stop Penalty")
    print("   • Average Travel Time: Total time ÷ vehicles (primary factor)")
    print("   • Stop Penalty: 25.0 × max individual stop time (fairness factor)")
    print("   • Goal: Fast overall flow + prevent individual vehicles from waiting too long")
    print("   • Implementation: calculate_cost() in src/optimization/simple_aco.py")
    print()
    
    # Configuration for quick demo
    config = {
        'grid_size': 3,                 # 3x3 grid 
        'n_vehicles': 20,               # 20 vehicles for quick test
        'simulation_time': 400,         # 400 seconds
        'n_ants': 10,                   # 10 ants for speed
        'n_iterations': 5               # 5 iterations for speed
    }
    
    print(f"📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    try:
        # Import the optimization system
        from src.simplified_traffic import generate_network_and_routes
        from src.optimization.simple_aco import run_simplified_aco_optimization
        
        print(f"\n🏗️  Step 1: Generating Scenario")
        
        # Generate traffic scenario
        scenario_result = generate_network_and_routes(
            grid_size=config['grid_size'],
            n_vehicles=config['n_vehicles'],
            sim_time=config['simulation_time'], 
            pattern='balanced',  # Balanced traffic
            seed=42  # Reproducible results
        )
        
        if not scenario_result['success']:
            print(f"❌ Scenario generation failed: {scenario_result.get('error', 'Unknown error')}")
            return False
        
        print(f"✅ Scenario generated successfully")
        
        print(f"\n🐜 Step 2: Running ACO Optimization")
        
        # Run optimization
        optimization_result = run_simplified_aco_optimization(config)
        
        if optimization_result['success']:
            print(f"\n🎉 Optimization Completed!")
            print(f"   Best Cost: {optimization_result['best_cost']:.1f}")
            print(f"   Duration: {optimization_result['duration']:.1f} seconds")
            print(f"   Iterations: {len(optimization_result['cost_history'])}")
            
            # Show improvement over iterations
            cost_history = optimization_result['cost_history']
            if len(cost_history) > 1:
                initial_cost = cost_history[0]
                final_cost = cost_history[-1]
                improvement = ((initial_cost - final_cost) / initial_cost) * 100
                print(f"   Improvement: {improvement:.1f}%")
            
            print(f"\n📊 Key Benefits of Simple ACO:")
            print(f"   ✅ Direct range sampling (20-100s green, 3-6s yellow)")
            print(f"   ✅ No complex bins arrays or mapping logic")
            print(f"   ✅ Stable iteration performance")
            print(f"   ✅ Easy to understand and modify")
            print(f"   ✅ Automatic plotting saved to results/")
            
            return True
            
        else:
            print(f"❌ Optimization failed: {optimization_result.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_example()
    if success:
        print("\n✅ Example completed successfully!")
        print("💡 For more features, use: python optimize.py --help")
    else:
        print("\n❌ Example failed. Check error messages above.")
