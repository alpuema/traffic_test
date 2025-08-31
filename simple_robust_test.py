#!/usr/bin/env python3
"""
Simple test for robust ACO to isolate any issues.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def simple_test():
    """Simple test with minimal configuration."""
    
    try:
        from src.simplified_traffic import create_traffic_scenario
        from src.optimization.robust_aco import RobustACOTrafficOptimizer
        
        print("🧪 Simple Robust ACO Test")
        print("=" * 30)
        
        # Create scenario
        scenario = create_traffic_scenario(
            grid_size=2,
            n_vehicles=8,  # Even smaller for test
            simulation_time=400,
            pattern='random',
            seed=42
        )
        
        if scenario:
            print(f"✅ Scenario: {scenario['config_file']}")
            
            # Create optimizer with minimal settings
            optimizer = RobustACOTrafficOptimizer(
                sumo_config=scenario['config_file'],
                n_ants=3,
                n_iterations=2,
                training_seeds=2,  # Just 2 seeds
                scenario_vehicles=8,
                simulation_time=400,
                verbose=True,
                show_plots=False,
                show_sumo_gui=False,
                compare_baseline=True  # Enable baseline comparison to test the error
            )
            
            print("🚀 Running optimization...")
            best_solution, best_cost, opt_data, baseline_comp = optimizer.optimize()
            
            print(f"✅ Success! Best cost: {best_cost:.1f}")
            print(f"📊 Solution type: {type(best_solution)}")
            print(f"📊 Data type: {type(opt_data)}")
            print(f"📊 Baseline type: {type(baseline_comp)}")
            
            return True
            
        else:
            print("❌ Failed to create scenario")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    simple_test()
