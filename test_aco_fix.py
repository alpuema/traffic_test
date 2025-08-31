#!/usr/bin/env python3
"""
Test ACO Vehicle Completion Fix

This script tests if the ACO optimization now correctly ensures all vehicles complete.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.optimize import ACOTrafficOptimizer
from src.simplified_traffic import create_traffic_scenario

def test_aco_vehicle_completion():
    """Test ACO optimization with focus on vehicle completion."""
    
    print("🧪 TESTING ACO VEHICLE COMPLETION FIX")
    print("=" * 60)
    
    # Create scenario
    scenario = create_traffic_scenario(
        grid_size=5,
        n_vehicles=20,
        simulation_time=2400,  # Our increased time
        pattern='industrial',
        seed=42
    )
    
    if not scenario:
        print("❌ Failed to create scenario")
        return False
        
    print(f"✅ Created scenario with {scenario.get('n_vehicles', 20)} vehicles")
    
    # Set up optimizer with minimal iterations for quick test
    optimizer = ACOTrafficOptimizer(
        sumo_config=scenario['config_file'],
        n_ants=5,  # Fewer ants for faster test
        n_iterations=3,  # Fewer iterations for faster test
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        verbose=True,
        scenario_vehicles=20,
        simulation_time=2400,  # Our increased time
        show_plots=False,  # Skip plots for testing
        show_sumo_gui=False,
        compare_baseline=True
    )
    
    print(f"🔧 Running quick ACO test (5 ants, 3 iterations)...")
    
    try:
        best_solution, best_cost, optimization_data, baseline_comparison = optimizer.optimize()
        
        print(f"✅ Optimization completed!")
        print(f"📊 Best cost: {best_cost:.2f}")
        
        # Check if we have baseline comparison data
        if baseline_comparison:
            print(f"🔍 Baseline comparison structure: {baseline_comparison.keys()}")
            
            optimized_metrics = baseline_comparison.get('optimized', {}).get('metrics', {})
            baseline_metrics = baseline_comparison.get('baseline', {}).get('metrics', {})
            
            optimized_vehicles = optimized_metrics.get('vehicles', 0)
            baseline_vehicles = baseline_metrics.get('vehicles', 0)
            
            print(f"🚗 Baseline vehicles completed: {baseline_vehicles}/20")
            print(f"🚗 Optimized vehicles completed: {optimized_vehicles}/20")
            
            if optimized_vehicles >= 20:
                print("🎉 SUCCESS! All vehicles completed in optimized solution!")
                return True
            else:
                print(f"❌ STILL FAILING: Only {optimized_vehicles}/20 vehicles completed")
                return False
        else:
            print("⚠️  No baseline comparison data available")
            return False
            
    except Exception as e:
        print(f"❌ ACO optimization failed: {e}")
        return False

if __name__ == "__main__":
    success = test_aco_vehicle_completion()
    if success:
        print("\n✅ FIX VERIFIED: Vehicle completion issue resolved!")
    else:
        print("\n❌ FIX FAILED: Vehicle completion issue persists!")
        sys.exit(1)
