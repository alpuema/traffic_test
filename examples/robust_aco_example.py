#!/usr/bin/env python3
"""
Robust ACO Optimization Example

This example demonstrates the robust multi-seed ACO algorithm that trains
across multiple traffic scenarios to reduce overfitting and find solutions
that generalize well to unseen traffic patterns.

Key Benefits:
• Trains on N different traffic seeds simultaneously  
• Prevents overfitting to a single traffic scenario
• Solutions work better on new/unseen traffic patterns
• Adaptive seed weighting focuses on challenging scenarios
• Validation on completely fresh seeds

Usage:
    python examples/robust_aco_example.py

Author: Traffic Optimization System
Date: August 2025
"""

import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# 🎯 CONFIGURATION - MODIFY THESE SETTINGS!
# ============================================================================

# Scenario Configuration
GRID_SIZE = 3                  # Grid size (3x3 = 9 intersections, faster for demo)
N_VEHICLES = 20               # Number of vehicles per scenario
SIMULATION_TIME = 1800        # Simulation duration (30 minutes)
TRAFFIC_PATTERN = 'commuter'  # Traffic pattern ('commuter', 'industrial', 'random')

# Robust ACO Parameters
N_TRAINING_SEEDS = 5          # Number of different traffic seeds to train on (3-10 recommended)
N_ANTS = 15                   # Ants per iteration (fewer for multi-seed since each evaluation is slower)
N_ITERATIONS = 8              # Optimization iterations (fewer since training is more robust)
EXPLORATION_RATE = 0.25       # Higher exploration for robustness (0.20-0.30)
EVAPORATION_RATE = 0.08       # Pheromone evaporation (slightly lower for stability)

# Validation and Display
VALIDATE_ON_NEW_SEEDS = True  # Test final solution on completely new seeds
SHOW_PLOTS = True             # Show optimization and robustness plots
SHOW_SUMO_GUI = False         # Launch SUMO GUI with results
COMPARE_BASELINE = True       # Compare against baseline uniform timing

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run robust ACO optimization example."""
    
    print("🌱 ROBUST MULTI-SEED ACO OPTIMIZATION")
    print("=" * 70)
    print("Welcome to robust traffic light optimization!")
    print()
    print("🧠 What makes this 'robust'?")
    print("   • Trains on multiple different traffic scenarios simultaneously")
    print("   • Prevents overfitting to a single traffic pattern")
    print("   • Solutions work better on new/unseen traffic")
    print("   • Adaptive learning focuses on challenging scenarios")
    print("   • Validation testing on completely fresh seeds")
    print()
    print(f"📋 Configuration:")
    print(f"   Grid: {GRID_SIZE}x{GRID_SIZE} ({GRID_SIZE**2} intersections)")
    print(f"   Vehicles per scenario: {N_VEHICLES}")
    print(f"   Training seeds: {N_TRAINING_SEEDS}")
    print(f"   ACO: {N_ANTS} ants × {N_ITERATIONS} iterations")
    print(f"   Exploration rate: {EXPLORATION_RATE:.2f} (higher for robustness)")
    print()
    
    # Ask user if they want to continue
    proceed = input("Ready to run robust optimization? (y/n) [default: y]: ").strip().lower()
    if proceed == 'n':
        print("👋 No problem! Edit the configuration above to customize settings.")
        return
    
    try:
        # Import the robust ACO module
        from src.optimization.robust_aco import RobustACOTrafficOptimizer
        from src.simplified_traffic import create_traffic_scenario
        
        print("\n🏗️  CREATING BASE SCENARIO")
        print("-" * 40)
        
        # Create a base scenario for the optimizer to use as template
        base_scenario = create_traffic_scenario(
            grid_size=GRID_SIZE,
            n_vehicles=N_VEHICLES,
            simulation_time=SIMULATION_TIME,
            pattern=TRAFFIC_PATTERN,
            seed=42  # Base seed for template
        )
        
        if not base_scenario:
            print("❌ Failed to create base scenario")
            return
        
        print(f"✅ Base scenario created: {base_scenario['config_file']}")
        
        print("\n🐜 ROBUST ACO OPTIMIZATION")
        print("-" * 40)
        
        # Create robust optimizer
        optimizer = RobustACOTrafficOptimizer(
            sumo_config=base_scenario['config_file'],
            n_ants=N_ANTS,
            n_iterations=N_ITERATIONS,
            alpha=1.0,  # Stop penalty weight
            beta=2.0,   # Heuristic weight
            rho=EVAPORATION_RATE,
            verbose=True,
            scenario_vehicles=N_VEHICLES,
            simulation_time=SIMULATION_TIME,
            show_plots=SHOW_PLOTS,
            show_sumo_gui=SHOW_SUMO_GUI,
            compare_baseline=COMPARE_BASELINE,
            # Robust-specific parameters
            training_seeds=N_TRAINING_SEEDS,
            exploration_rate=EXPLORATION_RATE,
            validate_solution=VALIDATE_ON_NEW_SEEDS
        )
        
        # Run optimization
        print(f"🚀 Starting robust optimization...")
        print(f"⏰ This will take longer than regular ACO due to multi-seed evaluation")
        print()
        
        best_solution, best_cost, optimization_data, baseline_comparison = optimizer.optimize()
        
        print(f"\n🎉 ROBUST OPTIMIZATION COMPLETED!")
        print("=" * 50)
        print(f"📊 Best robust cost: {best_cost:.2f}")
        
        # Display baseline comparison results
        if baseline_comparison:
            baseline_cost = baseline_comparison.get('baseline', {}).get('cost', 0)
            optimized_cost = baseline_comparison.get('optimized', {}).get('cost', 0)
            improvement = baseline_comparison.get('improvement', {}).get('percent', 0)
            
            baseline_vehicles = baseline_comparison.get('baseline', {}).get('metrics', {}).get('vehicles', 0)
            optimized_vehicles = baseline_comparison.get('optimized', {}).get('metrics', {}).get('vehicles', 0)
            
            print(f"\n📊 MULTI-SEED BASELINE COMPARISON:")
            print(f"   Baseline (30s/4s): {baseline_cost:.1f} cost, {baseline_vehicles:.1f}/{N_VEHICLES} vehicles")
            print(f"   Robust optimized: {optimized_cost:.1f} cost, {optimized_vehicles:.1f}/{N_VEHICLES} vehicles")
            
            if improvement > 0:
                print(f"   ✅ Robust improvement: {improvement:.1f}% better!")
            else:
                print(f"   ⚠️  Performance: {abs(improvement):.1f}% change")
            
            # Show robustness info
            robustness = baseline_comparison.get('robustness', {})
            if robustness:
                baseline_seeds = robustness.get('baseline_seeds_evaluated', 0)
                optimized_seeds = robustness.get('optimized_seeds_evaluated', 0)
                total_seeds = robustness.get('total_seeds', N_TRAINING_SEEDS)
                print(f"   🌱 Training robustness: {optimized_seeds}/{total_seeds} seeds successful")
            
            # Show validation results if available
            validation = baseline_comparison.get('validation', {})
            if validation.get('success'):
                val_improvement = validation.get('improvement_percent', 0)
                val_seeds = validation.get('seeds_evaluated', 0)
                val_total = validation.get('total_validation_seeds', 0)
                
                print(f"\n🔍 VALIDATION ON UNSEEN SEEDS:")
                print(f"   Validation improvement: {val_improvement:.1f}%")
                print(f"   Validation robustness: {val_seeds}/{val_total} seeds successful")
                
                if val_improvement > 0:
                    print(f"   ✅ Solution generalizes well to new scenarios!")
                else:
                    print(f"   ⚠️  Solution may be overfitting to training seeds")
        
        print(f"\n💡 SOLUTION SUMMARY:")
        print(f"   • Trained on {N_TRAINING_SEEDS} different traffic scenarios")
        print(f"   • Uses robust evaluation across multiple seeds")
        print(f"   • Higher exploration rate ({EXPLORATION_RATE:.2f}) for generalization")
        print(f"   • Adaptive weighting focuses on challenging scenarios")
        
        if VALIDATE_ON_NEW_SEEDS:
            print(f"   • Validated on completely new seeds for true robustness")
        
        print("\n🎯 Next steps:")
        print("   • Compare robust vs regular ACO performance")
        print("   • Test the solution on different traffic patterns")
        print("   • Adjust training_seeds parameter for your needs")
        
        return best_solution, best_cost, optimization_data, baseline_comparison
        
    except Exception as e:
        print(f"\n❌ Robust optimization failed: {e}")
        print("💡 Try reducing the number of training seeds or checking scenario generation")
        raise

def compare_robust_vs_regular_aco():
    """Compare robust ACO against regular ACO on the same scenario."""
    
    print("\n🔬 ROBUST vs REGULAR ACO COMPARISON")
    print("=" * 50)
    print("This compares the robust multi-seed ACO against the regular single-seed ACO")
    print("to demonstrate the benefits of robust training.")
    print()
    
    try:
        from src.optimization.robust_aco import RobustACOTrafficOptimizer
        from src.optimize import ACOTrafficOptimizer
        from src.simplified_traffic import create_traffic_scenario
        
        # Create test scenario
        test_scenario = create_traffic_scenario(
            grid_size=GRID_SIZE,
            n_vehicles=N_VEHICLES,
            simulation_time=SIMULATION_TIME,
            pattern=TRAFFIC_PATTERN,
            seed=123  # Fixed seed for fair comparison
        )
        
        if not test_scenario:
            print("❌ Failed to create test scenario")
            return
        
        print("📊 Running both optimizers on same test scenario...")
        
        # Test regular ACO
        print("\n1️⃣  REGULAR ACO (single seed)")
        regular_optimizer = ACOTrafficOptimizer(
            sumo_config=test_scenario['config_file'],
            n_ants=N_ANTS,
            n_iterations=N_ITERATIONS,
            scenario_vehicles=N_VEHICLES,
            simulation_time=SIMULATION_TIME,
            show_plots=False,
            show_sumo_gui=False,
            compare_baseline=True
        )
        
        _, regular_cost, _, regular_baseline = regular_optimizer.optimize()
        
        # Test robust ACO
        print("\n2️⃣  ROBUST ACO (multi-seed)")
        robust_optimizer = RobustACOTrafficOptimizer(
            sumo_config=test_scenario['config_file'],
            n_ants=N_ANTS,
            n_iterations=N_ITERATIONS,
            scenario_vehicles=N_VEHICLES,
            simulation_time=SIMULATION_TIME,
            training_seeds=N_TRAINING_SEEDS,
            show_plots=False,
            show_sumo_gui=False,
            compare_baseline=True,
            validate_solution=True
        )
        
        _, robust_cost, _, robust_baseline = robust_optimizer.optimize()
        
        # Compare results
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"   Regular ACO cost: {regular_cost:.2f}")
        print(f"   Robust ACO cost: {robust_cost:.2f}")
        
        if robust_cost < regular_cost:
            improvement = ((regular_cost - robust_cost) / regular_cost) * 100
            print(f"   ✅ Robust ACO better by {improvement:.1f}%")
        else:
            degradation = ((robust_cost - regular_cost) / regular_cost) * 100
            print(f"   ⚠️  Robust ACO worse by {degradation:.1f}% (but may generalize better)")
        
        # Compare generalization
        if robust_baseline and robust_baseline.get('validation'):
            val_improvement = robust_baseline['validation'].get('improvement_percent', 0)
            print(f"   🔍 Robust ACO validation improvement: {val_improvement:.1f}%")
            print(f"   💡 Positive validation improvement suggests better generalization")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")

if __name__ == "__main__":
    main()
    
    # Optionally run comparison
    print("\n" + "="*70)
    run_comparison = input("\nRun comparison with regular ACO? (y/n) [default: n]: ").strip().lower()
    if run_comparison == 'y':
        compare_robust_vs_regular_aco()
