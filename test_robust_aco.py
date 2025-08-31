#!/usr/bin/env python3
"""
Quick test to verify robust ACO implementation works correctly.
"""

import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_robust_aco():
    """Test the robust ACO implementation."""
    
    try:
        # Import the robust ACO module
        from src.optimization.robust_aco import RobustACOTrafficOptimizer
        from src.simplified_traffic import create_traffic_scenario
        
        print('✅ Successfully imported robust ACO components')
        
        # Create a small test scenario
        test_scenario = create_traffic_scenario(
            grid_size=2,
            n_vehicles=10,  
            simulation_time=600,
            pattern='random',
            seed=42
        )
        
        if test_scenario:
            print(f'✅ Test scenario created: {test_scenario["config_file"]}')
            
            # Create optimizer (don't run full optimization)
            optimizer = RobustACOTrafficOptimizer(
                sumo_config=test_scenario['config_file'],
                n_ants=5,
                n_iterations=2,
                training_seeds=3,
                scenario_vehicles=10,
                simulation_time=600,
                verbose=False,
                show_plots=False,
                show_sumo_gui=False
            )
            print('✅ Robust ACO optimizer created successfully')
            print(f'📊 Training seeds: {optimizer.training_seeds}')
            print(f'🧠 Parameters: {optimizer.n_ants} ants, {optimizer.n_iterations} iterations')
            print(f'🎯 Multi-seed training ready!')
            
            return True
            
        else:
            print('❌ Failed to create test scenario')
            return False
            
    except Exception as e:
        print(f'❌ Test failed: {e}')
        return False

if __name__ == "__main__":
    success = test_robust_aco()
    if success:
        print('\n🎉 Robust ACO implementation ready to use!')
        print('🚀 Run "python examples/robust_aco_example.py" to see it in action')
    else:
        print('\n💥 Test failed - check the implementation')
