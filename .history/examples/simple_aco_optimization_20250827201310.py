#!/usr/bin/env python3
"""
Interactive Simple ACO Optimization Example

This example provides an easy-to-use interface for traffic light optimization
with configurable options like traffic patterns, GUI settings, and more.

Author: Traffic Optimization System
Date: August 2025
"""

import sys
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_user_configuration():
    """Get configuration from user input with sensible defaults."""
    
    print("🚀 Interactive ACO Traffic Light Optimization")
    print("=" * 60)
    print("Configure your optimization run (press Enter for defaults)")
    print()
    
    # Get grid size
    while True:
        grid_input = input("Grid size (2, 3, or 4) [default: 3]: ").strip()
        if not grid_input:
            grid_size = 3
            break
        try:
            grid_size = int(grid_input)
            if grid_size in [2, 3, 4]:
                break
            else:
                print("❌ Please enter 2, 3, or 4")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Get number of vehicles
    while True:
        vehicles_input = input(f"Number of vehicles (10-100) [default: 30]: ").strip()
        if not vehicles_input:
            n_vehicles = 30
            break
        try:
            n_vehicles = int(vehicles_input)
            if 10 <= n_vehicles <= 100:
                break
            else:
                print("❌ Please enter a number between 10 and 100")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Get simulation time
    while True:
        time_input = input("Simulation time in seconds (300-3600) [default: 600]: ").strip()
        if not time_input:
            sim_time = 600
            break
        try:
            sim_time = int(time_input)
            if 300 <= sim_time <= 3600:
                break
            else:
                print("❌ Please enter a time between 300 and 3600 seconds")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Get traffic pattern
    patterns = {
        '1': 'balanced',
        '2': 'random', 
        '3': 'commuter',
        '4': 'commercial',
        '5': 'industrial'
    }
    
    print("\nTraffic Patterns:")
    print("1. Balanced - Realistic urban traffic (recommended)")
    print("2. Random - Completely random origins/destinations")  
    print("3. Commuter - Rush hour, suburbs to downtown")
    print("4. Commercial - Shopping district pattern")
    print("5. Industrial - Industrial zone corridors")
    
    while True:
        pattern_input = input("Choose traffic pattern (1-5) [default: 1]: ").strip()
        if not pattern_input:
            pattern = 'balanced'
            break
        if pattern_input in patterns:
            pattern = patterns[pattern_input]
            break
        else:
            print("❌ Please enter 1, 2, 3, 4, or 5")
    
    # Get ACO parameters
    while True:
        ants_input = input("Number of ants per iteration (5-50) [default: 20]: ").strip()
        if not ants_input:
            n_ants = 20
            break
        try:
            n_ants = int(ants_input)
            if 5 <= n_ants <= 50:
                break
            else:
                print("❌ Please enter a number between 5 and 50")
        except ValueError:
            print("❌ Please enter a valid number")
    
    while True:
        iterations_input = input("Number of iterations (3-20) [default: 10]: ").strip()
        if not iterations_input:
            n_iterations = 10
            break
        try:
            n_iterations = int(iterations_input)
            if 3 <= n_iterations <= 20:
                break
            else:
                print("❌ Please enter a number between 3 and 20")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Get display options
    show_plots = input("Show optimization plots? (y/n) [default: y]: ").strip().lower()
    show_plots = show_plots != 'n'
    
    launch_gui = input("Launch SUMO-GUI with results? (y/n) [default: n]: ").strip().lower()
    launch_gui = launch_gui == 'y'
    
    verbose = input("Show detailed progress? (y/n) [default: y]: ").strip().lower()
    verbose = verbose != 'n'
    
    return {
        'grid_size': grid_size,
        'n_vehicles': n_vehicles,
        'simulation_time': sim_time,
        'pattern': pattern,
        'n_ants': n_ants,
        'n_iterations': n_iterations,
        'show_plots': show_plots,
        'launch_gui': launch_gui,
        'verbose': verbose,
        'seed': 42  # Fixed for reproducibility
    }

def run_optimization_with_config(config):
    """Run ACO optimization with user configuration."""
    
    print("\n" + "=" * 60)
    print("� STARTING TRAFFIC LIGHT OPTIMIZATION")
    print("=" * 60)
    
    if config['verbose']:
        print(f"📋 Final Configuration:")
        print(f"   Grid: {config['grid_size']}x{config['grid_size']}")
        print(f"   Vehicles: {config['n_vehicles']}")
        print(f"   Simulation time: {config['simulation_time']}s")
        print(f"   Traffic pattern: {config['pattern']}")
        print(f"   ACO: {config['n_ants']} ants × {config['n_iterations']} iterations")
        print(f"   Show plots: {'Yes' if config['show_plots'] else 'No'}")
        print(f"   Launch GUI: {'Yes' if config['launch_gui'] else 'No'}")
    
    try:
        # Import the optimization system
        from src.simplified_traffic import generate_network_and_routes
        from src.optimization.simple_aco import run_simplified_aco_optimization
        
        if config['verbose']:
            print(f"\n🏗️  Step 1: Generating Traffic Scenario")
        
        # Generate traffic scenario
        scenario_result = generate_network_and_routes(
            grid_size=config['grid_size'],
            n_vehicles=config['n_vehicles'],
            sim_time=config['simulation_time'], 
            pattern=config['pattern'],
            seed=config['seed']
        )
        
        if not scenario_result['success']:
            print(f"❌ Scenario generation failed: {scenario_result.get('error', 'Unknown error')}")
            return False
        
        if config['verbose']:
            print(f"✅ Scenario generated successfully")
            print(f"\n🐜 Step 2: Running ACO Optimization")
        else:
            print("🐜 Running optimization...")
        
        # Update simple ACO configuration with user preferences
        aco_config = {
            'grid_size': config['grid_size'],
            'n_vehicles': config['n_vehicles'],
            'simulation_time': config['simulation_time'],
            'n_ants': config['n_ants'],
            'n_iterations': config['n_iterations']
        }
        
        # Temporarily modify the simple_aco module settings
        import src.optimization.simple_aco as aco_module
        original_show_plots = aco_module.SHOW_PLOTS
        original_show_progress = aco_module.SHOW_PROGRESS
        
        aco_module.SHOW_PLOTS = config['show_plots']
        aco_module.SHOW_PROGRESS = config['verbose']
        
        try:
            # Run optimization
            optimization_result = run_simplified_aco_optimization(aco_config)
        finally:
            # Restore original settings
            aco_module.SHOW_PLOTS = original_show_plots
            aco_module.SHOW_PROGRESS = original_show_progress
        
        if optimization_result['success']:
            print(f"\n🎉 OPTIMIZATION COMPLETED!")
            print(f"   Best Cost: {optimization_result['best_cost']:.1f}")
            print(f"   Duration: {optimization_result['duration']:.1f} seconds")
            print(f"   Total Iterations: {len(optimization_result['cost_history'])}")
            
            # Show improvement over iterations
            cost_history = optimization_result['cost_history']
            if len(cost_history) > 1:
                initial_cost = cost_history[0]
                final_cost = cost_history[-1]
                improvement = ((initial_cost - final_cost) / initial_cost) * 100
                print(f"   Improvement: {improvement:.1f}%")
            
            # Launch GUI if requested
            if config['launch_gui']:
                print(f"\n�️  Launching SUMO-GUI...")
                try:
                    import subprocess
                    import sumolib
                    sumo_gui = sumolib.checkBinary('sumo-gui')
                    
                    # Get the generated config file
                    config_file = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'sumo_data', f'grid_{config["grid_size"]}x{config["grid_size"]}.sumocfg'
                    )
                    
                    if os.path.exists(config_file):
                        print(f"   Opening: {config_file}")
                        subprocess.Popen([sumo_gui, '-c', config_file])
                    else:
                        print(f"   ❌ Config file not found: {config_file}")
                        
                except Exception as e:
                    print(f"   ❌ Could not launch SUMO-GUI: {e}")
                    print(f"   💡 Try running: sumo-gui -c sumo_data/grid_{config['grid_size']}x{config['grid_size']}.sumocfg")
            
            if config['show_plots']:
                print(f"\n�📊 Optimization plot saved to: results/aco_optimization_progress.png")
            
            print(f"\n✨ Key Benefits of Simple ACO:")
            print(f"   ✅ Direct range sampling (20-100s green, 3-6s yellow)")
            print(f"   ✅ No complex bins arrays or mapping logic")
            print(f"   ✅ Stable iteration performance")
            print(f"   ✅ Easy to understand and modify")
            
            return True
            
        else:
            print(f"❌ Optimization failed: {optimization_result.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if config['verbose']:
            import traceback
            traceback.print_exc()
        return False

def main():
    """Main function with options for different usage modes."""
    
    print("🚦 Simple ACO Traffic Light Optimization")
    print("=" * 50)
    print("Choose how to run:")
    print("1. Interactive mode - Configure all options")
    print("2. Quick demo - Use sensible defaults")
    print("3. Show available traffic patterns")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-3): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                return True
                
            elif choice == "1":
                print("\n🔧 Interactive Configuration Mode")
                config = get_user_configuration()
                success = run_optimization_with_config(config)
                break
                
            elif choice == "2":
                print("\n🚀 Quick Demo Mode")
                config = {
                    'grid_size': 3,
                    'n_vehicles': 30,
                    'simulation_time': 600,
                    'pattern': 'balanced',
                    'n_ants': 15,
                    'n_iterations': 8,
                    'show_plots': True,
                    'launch_gui': False,
                    'verbose': True,
                    'seed': 42
                }
                print("Using default configuration for quick demo...")
                success = run_optimization_with_config(config)
                break
                
            elif choice == "3":
                show_traffic_patterns()
                continue
                
            else:
                print("❌ Invalid choice. Please enter 0, 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            return False
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False
    
    return success

def show_traffic_patterns():
    """Display available traffic patterns and their descriptions."""
    
    print("\n📋 Available Traffic Patterns:")
    print("-" * 50)
    
    patterns = {
        'balanced': 'Balanced urban traffic with realistic distribution',
        'random': 'Completely random vehicle origins and destinations',  
        'commuter': 'Rush hour pattern - suburbs to downtown',
        'commercial': 'Shopping district - distributed to concentrated areas',
        'industrial': 'Industrial zone pattern - specific high-traffic corridors'
    }
    
    for name, desc in patterns.items():
        print(f"  {name:12} - {desc}")
    
    print("\n💡 Tip: 'balanced' is recommended for general optimization testing")

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Program completed successfully!")
        print("💡 For advanced features, use: python optimize.py --help")
    else:
        print("\n❌ Program ended with errors.")
