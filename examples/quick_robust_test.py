#!/usr/bin/env python3
"""
Quick Test: Robust ACO vs Regular ACO for Commuter Pattern Only

Fast comparison to validate the robust vs regular approach.
"""

import sys
import os
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simplified_traffic import create_traffic_scenario
from src.optimization.robust_aco import RobustACOTrafficOptimizer
from src.optimize import ACOTrafficOptimizer

def quick_test():
    """Quick test of robust vs regular ACO on commuter pattern."""
    print("🚀 QUICK ROBUST vs REGULAR ACO TEST")
    print("=" * 50)
    print("Testing: Commuter pattern only, small parameters")
    
    # Test parameters
    grid_size = 3
    n_vehicles = 15
    simulation_time = 1500
    pattern = 'commuter'
    
    print(f"📋 Config: {grid_size}x{grid_size} grid, {n_vehicles} vehicles, {simulation_time}s")
    
    # Generate base scenario
    print("\n🏗️ Generating test scenario...")
    scenario = create_traffic_scenario(
        grid_size=grid_size,
        n_vehicles=n_vehicles,
        simulation_time=simulation_time,
        pattern=pattern
    )
    
    if not scenario:
        print("❌ Scenario generation failed")
        return
    
    print("✅ Scenario generated successfully")
    
    # Test 1: Robust ACO
    print("\n🌱 Testing Robust ACO...")
    print("  Training on 3 seeds, 8 ants × 3 iterations")
    
    try:
        robust_optimizer = RobustACOTrafficOptimizer(
            sumo_config=scenario['config_file'],
            n_ants=8,
            n_iterations=3,
            scenario_vehicles=n_vehicles,
            simulation_time=simulation_time,
            show_plots=False,
            show_sumo_gui=False,
            compare_baseline=True,
            training_seeds=3,
            exploration_rate=0.25,
            validate_solution=True
        )
        
        start_time = time.time()
        robust_solution, robust_cost, robust_data, robust_baseline = robust_optimizer.optimize()
        robust_time = time.time() - start_time
        
        robust_improvement = 0
        if robust_baseline and isinstance(robust_baseline, dict):
            baseline_cost = robust_baseline.get('baseline_cost', 0)
            optimized_cost = robust_baseline.get('optimized_cost', robust_cost)
            if baseline_cost > 0:
                robust_improvement = ((baseline_cost - optimized_cost) / baseline_cost) * 100
        
        print(f"  ✅ Robust ACO completed: Cost {robust_cost:.1f}, "
              f"Improvement {robust_improvement:.1f}%, Time {robust_time:.1f}s")
    
    except Exception as e:
        print(f"  ❌ Robust ACO failed: {e}")
        return
    
    # Test 2: Regular ACO
    print("\n🐜 Testing Regular ACO...")
    print("  Single seed, 12 ants × 5 iterations")
    
    try:
        regular_optimizer = ACOTrafficOptimizer(
            sumo_config=scenario['config_file'],
            n_ants=12,
            n_iterations=5,
            scenario_vehicles=n_vehicles,
            simulation_time=simulation_time,
            show_plots=False,
            show_sumo_gui=False,
            compare_baseline=True
        )
        
        start_time = time.time()
        regular_solution, regular_cost, regular_data, regular_baseline = regular_optimizer.optimize()
        regular_time = time.time() - start_time
        
        regular_improvement = 0
        if regular_baseline and isinstance(regular_baseline, dict):
            baseline_cost = regular_baseline.get('baseline_cost', 0)
            optimized_cost = regular_baseline.get('optimized_cost', regular_cost)
            if baseline_cost > 0:
                regular_improvement = ((baseline_cost - optimized_cost) / baseline_cost) * 100
        
        print(f"  ✅ Regular ACO completed: Cost {regular_cost:.1f}, "
              f"Improvement {regular_improvement:.1f}%, Time {regular_time:.1f}s")
    
    except Exception as e:
        print(f"  ❌ Regular ACO failed: {e}")
        return
    
    # Comparison
    print("\n📊 COMPARISON RESULTS:")
    print(f"   Robust ACO:  Cost {robust_cost:.1f}s, Improvement {robust_improvement:.1f}%, Time {robust_time:.1f}s")
    print(f"   Regular ACO: Cost {regular_cost:.1f}s, Improvement {regular_improvement:.1f}%, Time {regular_time:.1f}s")
    
    if robust_cost < regular_cost:
        advantage = ((regular_cost - robust_cost) / regular_cost) * 100
        print(f"   🌟 Robust ACO is {advantage:.1f}% better (lower cost)")
    elif regular_cost < robust_cost:
        advantage = ((robust_cost - regular_cost) / robust_cost) * 100
        print(f"   🚀 Regular ACO is {advantage:.1f}% better (lower cost)")
    else:
        print(f"   🤝 Both approaches achieved similar results")
    
    print(f"\n💡 Robust ACO trades {robust_time - regular_time:.1f}s extra time for multi-seed robustness")
    print("✅ Quick test completed!")

if __name__ == "__main__":
    quick_test()
