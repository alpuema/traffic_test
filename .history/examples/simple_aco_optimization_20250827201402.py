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
    """Display available traffic patterns with optional heatmap visualization."""
    
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
    
    # Offer heatmap visualization
    print("\n🗺️  Would you like to see heatmaps for these patterns?")
    show_heatmaps = input("Generate pattern heatmaps? (y/n) [default: n]: ").strip().lower()
    
    if show_heatmaps == 'y':
        generate_traffic_pattern_heatmaps()

def generate_traffic_pattern_heatmaps():
    """Generate and display heatmaps for all traffic patterns."""
    
    print("\n🎨 Generating Traffic Pattern Heatmaps...")
    print("This will create temporary scenarios to analyze patterns.")
    
    try:
        from src.simplified_traffic import generate_network_and_routes
        
        patterns = ['balanced', 'random', 'commuter', 'commercial', 'industrial']
        grid_size = 3  # Use 3x3 grid for visualization
        n_vehicles = 100  # More vehicles for better heatmap data
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('Traffic Pattern Heatmaps - Origins (Top) and Destinations (Bottom)', fontsize=16, fontweight='bold')
        
        for idx, pattern in enumerate(patterns):
            print(f"   Analyzing {pattern} pattern...")
            
            # Generate scenario for this pattern
            result = generate_network_and_routes(
                grid_size=grid_size,
                n_vehicles=n_vehicles,
                sim_time=600,
                pattern=pattern,
                seed=42  # Fixed seed for consistent comparison
            )
            
            if result['success']:
                # Analyze the generated trips
                origins, destinations, grid_coords = analyze_trips_for_heatmap(
                    result['files']['trips'], 
                    result['files']['network'],
                    grid_size
                )
                
                # Create heatmaps
                create_pattern_heatmap(axes[0, idx], origins, grid_coords, f"{pattern}\n(Origins)", 'Reds')
                create_pattern_heatmap(axes[1, idx], destinations, grid_coords, f"{pattern}\n(Destinations)", 'Blues')
            
            else:
                print(f"   ❌ Failed to generate {pattern} pattern")
                # Create empty plots for failed patterns
                axes[0, idx].text(0.5, 0.5, f'{pattern}\nFailed', ha='center', va='center', transform=axes[0, idx].transAxes)
                axes[1, idx].text(0.5, 0.5, f'{pattern}\nFailed', ha='center', va='center', transform=axes[1, idx].transAxes)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save heatmap
        heatmap_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'results', 'traffic_pattern_heatmaps.png'
        )
        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        
        print(f"\n📊 Heatmaps saved to: {heatmap_path}")
        print("🖼️  Displaying heatmaps...")
        plt.show()
        
        # Cleanup temporary files
        cleanup_temp_scenario_files(grid_size)
        
    except Exception as e:
        print(f"❌ Error generating heatmaps: {e}")
        import traceback
        traceback.print_exc()

def analyze_trips_for_heatmap(trips_file, network_file, grid_size):
    """Analyze trips file to extract origin/destination coordinates."""
    
    # First, get edge coordinates from network file
    edge_coords = get_edge_coordinates(network_file, grid_size)
    
    # Parse trips file to get from/to edges
    origins = defaultdict(int)
    destinations = defaultdict(int)
    
    try:
        tree = ET.parse(trips_file)
        root = tree.getroot()
        
        for trip in root.findall('trip'):
            from_edge = trip.get('from')
            to_edge = trip.get('to')
            
            if from_edge in edge_coords:
                coord = edge_coords[from_edge]
                origins[coord] += 1
            
            if to_edge in edge_coords:
                coord = edge_coords[to_edge]
                destinations[coord] += 1
                
    except Exception as e:
        print(f"   ⚠️  Error parsing trips: {e}")
    
    return origins, destinations, edge_coords

def get_edge_coordinates(network_file, grid_size):
    """Extract edge coordinates from SUMO network file."""
    edge_coords = {}
    
    try:
        tree = ET.parse(network_file)
        root = tree.getroot()
        
        # For grid networks, we can map edge IDs to grid positions
        for edge in root.findall('edge'):
            edge_id = edge.get('id')
            if edge_id and not edge_id.startswith(':'):
                # Extract coordinates from edge geometry or infer from grid structure
                coord = infer_grid_position(edge_id, grid_size)
                if coord:
                    edge_coords[edge_id] = coord
                    
    except Exception as e:
        print(f"   ⚠️  Error parsing network: {e}")
    
    return edge_coords

def infer_grid_position(edge_id, grid_size):
    """Infer grid position from edge ID in SUMO grid networks."""
    try:
        # SUMO grid edges typically follow patterns like "A0A1", "B1B2", etc.
        # We'll use a heuristic mapping based on common SUMO grid naming
        
        # Extract junction names from edge ID
        if len(edge_id) >= 4:
            from_junction = edge_id[:2]  # e.g., "A0"
            to_junction = edge_id[2:4]   # e.g., "A1"
            
            # Map junction names to coordinates
            from_coord = junction_to_coord(from_junction, grid_size)
            to_coord = junction_to_coord(to_junction, grid_size)
            
            if from_coord and to_coord:
                # Use midpoint of edge as its coordinate
                return ((from_coord[0] + to_coord[0]) / 2, (from_coord[1] + to_coord[1]) / 2)
        
    except:
        pass
    
    # Fallback: random position within grid
    import random
    return (random.random() * grid_size, random.random() * grid_size)

def junction_to_coord(junction_name, grid_size):
    """Convert SUMO junction name to grid coordinate."""
    try:
        if len(junction_name) >= 2:
            row_letter = junction_name[0]  # A, B, C, ...
            col_number = int(junction_name[1])  # 0, 1, 2, ...
            
            # Convert to grid coordinates
            row = ord(row_letter) - ord('A')
            col = col_number
            
            if 0 <= row < grid_size and 0 <= col < grid_size:
                return (col, grid_size - 1 - row)  # Flip Y for proper display
                
    except:
        pass
    
    return None

def create_pattern_heatmap(ax, data_dict, edge_coords, title, colormap):
    """Create a heatmap for origin or destination data."""
    
    grid_size = 3  # Fixed for visualization
    
    # Create grid for heatmap
    heatmap_grid = np.zeros((grid_size, grid_size))
    
    # Fill grid with data
    for coord, count in data_dict.items():
        x, y = coord
        # Map continuous coordinates to grid cells
        grid_x = int(np.clip(x, 0, grid_size - 0.001))
        grid_y = int(np.clip(y, 0, grid_size - 0.001))
        heatmap_grid[grid_y, grid_x] += count
    
    # Create heatmap
    im = ax.imshow(heatmap_grid, cmap=colormap, aspect='equal', origin='upper')
    
    # Customize plot
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_xticklabels([f'Col {i}' for i in range(grid_size)])
    ax.set_yticklabels([f'Row {i}' for i in range(grid_size)])
    
    # Add grid lines
    for i in range(grid_size + 1):
        ax.axhline(i - 0.5, color='white', linewidth=1)
        ax.axvline(i - 0.5, color='white', linewidth=1)
    
    # Add value annotations
    for i in range(grid_size):
        for j in range(grid_size):
            value = int(heatmap_grid[i, j])
            if value > 0:
                ax.text(j, i, str(value), ha='center', va='center', 
                       color='white' if value > heatmap_grid.max() * 0.5 else 'black',
                       fontsize=8, fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def cleanup_temp_scenario_files(grid_size):
    """Clean up temporary scenario files created for heatmap generation."""
    try:
        import shutil
        
        # Remove temporary files
        temp_files = [
            f'sumo_data/grid_{grid_size}x{grid_size}.net.xml',
            f'sumo_data/grid_{grid_size}x{grid_size}.trips.xml', 
            f'sumo_data/grid_{grid_size}x{grid_size}.rou.xml',
            f'sumo_data/grid_{grid_size}x{grid_size}.sumocfg'
        ]
        
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for file_path in temp_files:
            full_path = os.path.join(project_dir, file_path)
            if os.path.exists(full_path):
                os.remove(full_path)
                
    except Exception as e:
        print(f"   ⚠️  Could not cleanup temp files: {e}")
        # Continue anyway - not critical

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Program completed successfully!")
        print("💡 For advanced features, use: python optimize.py --help")
    else:
        print("\n❌ Program ended with errors.")
