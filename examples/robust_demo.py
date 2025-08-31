#!/usr/bin/env python3
"""
Quick Demo: Robust Traffic Pattern Comparison

A quick demonstration version of the robust traffic pattern comparison
with smaller parameters for faster testing.

This shows the same functionality as the full version but with:
- Smaller grid (2x2)
- Fewer vehicles (10)
- Shorter simulation time
- Fewer iterations
- Only 2 patterns for speed

Perfect for testing the robust ACO functionality without waiting long.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def quick_robust_demo():
    """Run a quick demo of robust traffic pattern comparison."""
    
    print("🚀 QUICK ROBUST ACO DEMO")
    print("=" * 40)
    print("Testing robust multi-seed ACO with small parameters...")
    
    try:
        from src.simplified_traffic import create_traffic_scenario
        from src.optimization.robust_aco import RobustACOTrafficOptimizer
        
        # Quick demo parameters
        GRID_SIZE = 2
        N_VEHICLES = 10
        SIMULATION_TIME = 2600
        TRAINING_SEEDS = 3
        N_ANTS = 5
        N_ITERATIONS = 3
        PATTERNS = ['commuter', 'random']  # Just 2 patterns for speed
        
        print(f"📋 Demo Config: {GRID_SIZE}x{GRID_SIZE} grid, {N_VEHICLES} vehicles, {TRAINING_SEEDS} training seeds")
        print(f"   ACO: {N_ANTS} ants × {N_ITERATIONS} iterations per pattern")
        print()
        
        results = []
        
        for pattern in PATTERNS:
            print(f"🌱 Testing {pattern} pattern with robust ACO...")
            
            # Create scenario
            scenario = create_traffic_scenario(
                grid_size=GRID_SIZE,
                n_vehicles=N_VEHICLES,
                simulation_time=SIMULATION_TIME,
                pattern=pattern,
                seed=42
            )
            
            if scenario:
                print(f"  ✅ Scenario created: {scenario['config_file']}")
                
                # Create robust optimizer
                optimizer = RobustACOTrafficOptimizer(
                    sumo_config=scenario['config_file'],
                    n_ants=N_ANTS,
                    n_iterations=N_ITERATIONS,
                    training_seeds=TRAINING_SEEDS,
                    scenario_vehicles=N_VEHICLES,
                    simulation_time=SIMULATION_TIME,
                    verbose=False,  # Reduce output for demo
                    show_plots=False,
                    show_sumo_gui=False,
                    compare_baseline=True
                )
                
                print(f"  🐜 Running robust optimization...")
                best_solution, best_cost, opt_data, baseline_comp = optimizer.optimize()
                
                print(f"  ✅ {pattern} completed! Cost: {best_cost:.1f}")
                
                improvement = 0
                if baseline_comp and isinstance(baseline_comp, dict):
                    improvement = baseline_comp.get('improvement', {}).get('percent', 0)
                    print(f"     Multi-seed improvement: {improvement:.1f}%")
                
                results.append({
                    'pattern': pattern,
                    'cost': best_cost,
                    'improvement': improvement
                })
                
            else:
                print(f"  ❌ Failed to create {pattern} scenario")
                results.append({'pattern': pattern, 'cost': None, 'improvement': 0})
        
        print()
        print("📊 DEMO RESULTS SUMMARY:")
        print("-" * 30)
        
        for result in results:
            if result['cost'] is not None:
                print(f"{result['pattern'].capitalize():10} Cost: {result['cost']:6.1f}, Improvement: {result['improvement']:+5.1f}%")
            else:
                print(f"{result['pattern'].capitalize():10} Failed")
        
        print()
        print("🎉 Quick demo completed successfully!")
        print("💡 The robust ACO trains on multiple seeds to prevent overfitting")
        print("📈 Run the full version for comprehensive analysis across all patterns")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

if __name__ == "__main__":
    success = quick_robust_demo()
    if success:
        print("\n✅ Robust ACO is working correctly!")
        print("🚀 Ready to run full analysis: python examples/robust_traffic_pattern_comparison.py")
    else:
        print("\n💥 Demo failed - check the implementation")
