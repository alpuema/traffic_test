#!/usr/bin/env python3
"""
Simple ACO Traffic Light Optimization Example

A simplified example showing how to run ACO optimization for traffic lights.
Just modify the parameters below and run the script!

Author: Traffic Optimization System
Date: August 2025
"""

import sys
import os
import time
import json

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# USER PARAMETERS - MODIFY THESE TO CUSTOMIZE YOUR OPTIMIZATION
# ============================================================================

# Grid Configuration
GRID_SIZE = 5           # Options: 2, 3, or 4 (number of intersections will be GRID_SIZE²)
N_VEHICLES = 20         # Number of vehicles (10-100, more vehicles = more complex scenario)
SIMULATION_TIME = 2400   # Simulation duration in seconds (300-3600, longer = more accurate)
                        # Increased to ensure all vehicles complete - uses 80% for departures, 20% for completion

# Traffic Pattern
TRAFFIC_PATTERN = 'industrial'  # Options: 'commuter', 'industrial', 'random'
                             # commuter: Rush hour pattern (suburbs to downtown)
                             # industrial: Horizontal corridor traffic
                             # random: Completely random origins/destinations

# ACO Algorithm Parameters
N_ANTS = 50           # Number of ants (10-50, more ants = better exploration but slower)
N_ITERATIONS = 20     # Number of iterations (5-20, more iterations = better solutions but slower)
ALPHA = 1.0           # Pheromone importance (0.5-2.0, higher = more pheromone influence)
BETA = 2.0            # Heuristic importance (1.0-3.0, higher = more greedy behavior)
RHO = 0.1             # Evaporation rate (0.1-0.9, higher = faster pheromone decay)

# Display Options
SHOW_PLOTS = True     # Show optimization progress plots
SHOW_SUMO_GUI = False  # Launch SUMO GUI to visualize the optimized traffic scenario
VERBOSE = True        # Print detailed progress information
RANDOM_SEED = 42      # For reproducible results (None for random each time)
COMPARE_BASELINE = True  # Compare optimized solution against baseline (30s green, 4s yellow)

# ============================================================================
# MAIN SCRIPT - NO NEED TO MODIFY BELOW THIS LINE
# ============================================================================

def main():
    """Run the ACO optimization with the parameters specified above."""
    
    print("=" * 60)
    print("SIMPLE ACO TRAFFIC LIGHT OPTIMIZATION")
    print("=" * 60)
    print()
    
    # Display configuration
    print("Configuration:")
    print(f"  Grid Size: {GRID_SIZE}x{GRID_SIZE} ({GRID_SIZE**2} intersections)")
    print(f"  Vehicles: {N_VEHICLES}")
    print(f"  Simulation Time: {SIMULATION_TIME} seconds ({SIMULATION_TIME/60:.1f} minutes)")
    print(f"  Traffic Pattern: {TRAFFIC_PATTERN}")
    print(f"  ACO Ants: {N_ANTS}")
    print(f"  ACO Iterations: {N_ITERATIONS}")
    print(f"  Show Plots: {SHOW_PLOTS}")
    print(f"  Show SUMO GUI: {SHOW_SUMO_GUI}")
    print(f"  Compare Baseline: {COMPARE_BASELINE}")
    print()
    
    # Ask user to confirm
    input("Press Enter to start optimization (or Ctrl+C to cancel)...")
    print()
    
    try:
        # Import required modules
        from src.optimize import ACOTrafficOptimizer
        from src.simplified_traffic import create_traffic_scenario
        
        print("Setting up traffic scenario...")
        
        # Create traffic scenario
        scenario = create_traffic_scenario(
            grid_size=GRID_SIZE,
            n_vehicles=N_VEHICLES,
            simulation_time=SIMULATION_TIME,
            pattern=TRAFFIC_PATTERN,
            seed=RANDOM_SEED
        )
        
        print(f"Created {GRID_SIZE}x{GRID_SIZE} grid with {N_VEHICLES} vehicles")
        print(f"Traffic pattern: {TRAFFIC_PATTERN}")
        print()
        
        # Set up optimizer
        print("Initializing ACO optimizer...")
        
        optimizer = ACOTrafficOptimizer(
            sumo_config=scenario['config_file'],
            n_ants=N_ANTS,
            n_iterations=N_ITERATIONS,
            alpha=ALPHA,
            beta=BETA,
            rho=RHO,
            verbose=VERBOSE,
            scenario_vehicles=scenario.get('n_vehicles', N_VEHICLES),
            simulation_time=SIMULATION_TIME,
            show_plots=SHOW_PLOTS,
            show_sumo_gui=SHOW_SUMO_GUI,
            compare_baseline=COMPARE_BASELINE
        )
        
        print(f"ACO configured with {N_ANTS} ants, {N_ITERATIONS} iterations")
        print()
        
        # Run optimization
        print("Starting optimization...")
        print("This may take a few minutes depending on your parameters...")
        print("-" * 50)
        
        start_time = time.time()
        
        # Run the optimization
        best_solution, best_cost, optimization_data, baseline_comparison = optimizer.optimize()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print()
        print("-" * 50)
        print("OPTIMIZATION COMPLETE!")
        print(f"Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print()
        
        # Display results
        print("RESULTS:")
        print(f"  Best cost achieved: {best_cost:.2f}")
        print(f"  Number of traffic lights optimized: {len(best_solution)}")
        print()
        
        # Show traffic light timings
        print("Optimized Traffic Light Timings:")
        for tl_id, timings in best_solution.items():
            # Each phase is either green/red OR yellow, not both
            if 'yellow' in timings:
                print(f"  {tl_id}: Yellow phase = {timings['yellow']}s")
            elif 'green' in timings:
                print(f"  {tl_id}: Green/Red phase = {timings['green']}s")
            else:
                print(f"  {tl_id}: Unknown phase type")
        print()
        
        # Show baseline comparison if available
        if baseline_comparison and COMPARE_BASELINE:
            print("📊 BASELINE COMPARISON:")
            baseline_cost = baseline_comparison['baseline']['cost']
            optimized_cost = baseline_comparison['optimized']['cost']
            improvement = baseline_comparison['improvement']
            
            print(f"  Baseline (30s green, 4s yellow): {baseline_cost:.1f}")
            print(f"  Optimized solution: {optimized_cost:.1f}")
            
            if improvement['percent'] > 0:
                print(f"  ✅ Improvement: {improvement['percent']:.1f}% better ({improvement['absolute']:.1f} cost reduction)")
            elif improvement['percent'] < 0:
                print(f"  ❌ Degradation: {abs(improvement['percent']):.1f}% worse ({abs(improvement['absolute']):.1f} cost increase)")
            else:
                print(f"  ➖ No significant difference")
            print()
        
        # Show plots if requested
        if SHOW_PLOTS and optimization_data:
            try:
                import matplotlib.pyplot as plt
                
                print("Generating optimization progress plot...")
                
                # Plot optimization progress
                iterations = list(range(1, len(optimization_data) + 1))
                costs = [data['best_cost'] for data in optimization_data]
                
                plt.figure(figsize=(10, 6))
                plt.plot(iterations, costs, 'b-o', linewidth=2, markersize=6, label='ACO Optimization')
                
                # Add baseline cost as dashed horizontal line if available
                if baseline_comparison and COMPARE_BASELINE:
                    baseline_cost = baseline_comparison['baseline']['cost']
                    if baseline_cost != float('inf'):
                        plt.axhline(y=baseline_cost, color='r', linestyle='--', linewidth=2, 
                                   label=f'Baseline ({baseline_cost:.1f})', alpha=0.8)
                
                plt.xlabel('Iteration')
                plt.ylabel('Best Cost')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Set integer x-ticks
                plt.xticks(range(1, len(costs) + 1))
                
                plt.tight_layout()
                
                # Save plot
                results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
                os.makedirs(results_dir, exist_ok=True)
                plot_path = os.path.join(results_dir, 'aco_optimization_progress.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                
                print(f"Plot saved to: {plot_path}")
                plt.show()
                
            except ImportError:
                print("Matplotlib not available - skipping plots")
            except Exception as e:
                print(f"Error generating plots: {e}")
        
        # Save results
        try:
            results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
            os.makedirs(results_dir, exist_ok=True)
            
            # Save traffic light settings
            import csv
            csv_path = os.path.join(results_dir, 'aco_traffic_light_timings.csv')
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Traffic_Light_ID', 'Green_Time', 'Yellow_Time'])
                for tl_id, timings in best_solution.items():
                    writer.writerow([tl_id, timings.get('green', ''), timings.get('yellow', '')])
            
            print(f"Traffic light timings saved to: {csv_path}")
            
            # Save baseline comparison if available
            if baseline_comparison and COMPARE_BASELINE:
                comparison_path = os.path.join(results_dir, 'baseline_comparison.json')
                with open(comparison_path, 'w') as f:
                    json.dump(baseline_comparison, f, indent=2, default=str)
                print(f"Baseline comparison saved to: {comparison_path}")
            
        except Exception as e:
            print(f"Warning: Could not save results to file: {e}")
        
        print()
        print("=" * 60)
        print("SUCCESS! Optimization completed successfully.")
        
        if COMPARE_BASELINE:
            print()
            print("🚦 HOW TRAFFIC LIGHT OPTIMIZATION WORKS:")
            print("=" * 60)
            print("The optimization ONLY changes phase durations, NOT which directions get green.")
            print("Red timing is AUTOMATIC - when one direction is green, conflicts are red.")
            print("Example: 30s green North-South automatically means 30s red East-West.")
            print("This ensures safety while optimizing traffic flow efficiency!")
        
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"ERROR: Required module not found: {e}")
        print("Make sure you have installed all dependencies and the src modules are available.")
        return False
        
    except Exception as e:
        print(f"ERROR during optimization: {e}")
        import traceback
        if VERBOSE:
            traceback.print_exc()
        return False


def print_help():
    """Print help information about the parameters."""
    print("=" * 60)
    print("PARAMETER HELP")
    print("=" * 60)
    print()
    print("GRID_SIZE (2-4):")
    print("  2: 4 intersections - Quick testing")
    print("  3: 9 intersections - Balanced complexity (recommended)")
    print("  4: 16 intersections - Complex scenarios")
    print()
    print("N_VEHICLES (10-100):")
    print("  10-25: Light traffic")
    print("  25-50: Moderate traffic (recommended)")
    print("  50-100: Heavy traffic")
    print()
    print("SIMULATION_TIME (300-3600 seconds):")
    print("  300-600: Quick test (5-10 minutes)")
    print("  600-1200: Standard run (10-20 minutes, recommended)")
    print("  1200-3600: Detailed analysis (20-60 minutes)")
    print()
    print("TRAFFIC_PATTERN:")
    print("  'commuter': Rush hour pattern - suburbs to downtown")
    print("  'industrial': Industrial corridor - horizontal traffic flow")
    print("  'random': Completely random origins and destinations")
    print()
    print("ACO PARAMETERS:")
    print("  N_ANTS (10-50): More ants = better exploration but slower")
    print("  N_ITERATIONS (5-20): More iterations = better solutions but slower")
    print("  ALPHA (0.5-2.0): Pheromone importance")
    print("  BETA (1.0-3.0): Heuristic importance")
    print("  RHO (0.1-0.9): Evaporation rate")
    print()
    print("DISPLAY OPTIONS:")
    print("  SHOW_PLOTS: Show optimization progress plots")
    print("  SHOW_SUMO_GUI: Launch SUMO GUI with optimized results")
    print("  VERBOSE: Print detailed progress information")
    print()


if __name__ == "__main__":
    # Check if help was requested
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_help()
    else:
        success = main()
        if not success:
            print("\nFor parameter help, run: python simple_aco_optimization.py --help")
            sys.exit(1)
