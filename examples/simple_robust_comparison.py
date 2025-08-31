#!/usr/bin/env python3
"""
Simple Robust ACO vs Regular ACO Traffic Pattern Comparison

This script provides a direct comparison between:
- Robust ACO: Multi-seed training to prevent overfitting
- Regular ACO: Single-seed training (traditional approach)

Focus on core functionality without complex cross-seed validation.
Date: August 2025
"""

import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simplified_traffic import create_traffic_scenario
from src.optimization.robust_aco import RobustACOTrafficOptimizer
from src.optimize import ACOTrafficOptimizer

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

# Grid and simulation settings
GRID_SIZE = 4
N_VEHICLES = 20
SIMULATION_TIME = 2000
PATTERNS = ['commuter', 'industrial', 'random']

# Robust ACO parameters
ROBUST_N_ANTS = 15
ROBUST_N_ITERATIONS = 8
TRAINING_SEEDS = 5
EXPLORATION_RATE = 0.25

# Regular ACO parameters (for comparison)
REGULAR_N_ANTS = 20
REGULAR_N_ITERATIONS = 12

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_progress(msg, indent=0):
    """Print formatted progress messages."""
    prefix = "  " * indent
    timestamp = time.strftime("[%H:%M:%S]")
    print(f"{prefix}{timestamp} {msg}")

def save_results(results, filename):
    """Save results to JSON file."""
    results_file = os.path.join("results", filename)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📁 Results saved to: {results_file}")

# ============================================================================
# OPTIMIZATION FUNCTIONS
# ============================================================================

def run_robust_aco(pattern):
    """Run robust ACO optimization for a traffic pattern."""
    print_progress(f"🐜🌱 Running Robust ACO for '{pattern}' pattern...")
    print_progress(f"Training on {TRAINING_SEEDS} seeds to prevent overfitting", 1)
    
    try:
        # Generate base scenario
        scenario = create_traffic_scenario(
            grid_size=GRID_SIZE,
            n_vehicles=N_VEHICLES,
            simulation_time=SIMULATION_TIME,
            pattern=pattern
        )
        
        if not scenario:
            return {'success': False, 'error': 'Scenario generation failed'}
        
        print_progress(f"✅ Base scenario generated for {pattern}", 1)
        
        # Initialize robust optimizer
        optimizer = RobustACOTrafficOptimizer(
            sumo_config=scenario['config_file'],
            n_ants=ROBUST_N_ANTS,
            n_iterations=ROBUST_N_ITERATIONS,
            scenario_vehicles=N_VEHICLES,
            simulation_time=SIMULATION_TIME,
            show_plots=False,
            show_sumo_gui=False,
            compare_baseline=True,
            training_seeds=TRAINING_SEEDS,
            exploration_rate=EXPLORATION_RATE,
            validate_solution=True
        )
        
        # Run optimization
        start_time = time.time()
        best_solution, best_cost, optimization_data, baseline_comparison = optimizer.optimize()
        end_time = time.time()
        
        # Calculate improvement
        improvement = None
        if baseline_comparison and isinstance(baseline_comparison, dict):
            baseline_cost = baseline_comparison.get('baseline_cost', 0)
            optimized_cost = baseline_comparison.get('optimized_cost', best_cost)
            if baseline_cost > 0:
                improvement = ((baseline_cost - optimized_cost) / baseline_cost) * 100
        
        return {
            'success': True,
            'pattern': pattern,
            'algorithm': 'robust_aco',
            'best_solution': best_solution,
            'best_cost': best_cost,
            'improvement': improvement,
            'baseline_comparison': baseline_comparison,
            'optimization_time': end_time - start_time,
            'training_seeds': TRAINING_SEEDS,
            'n_ants': ROBUST_N_ANTS,
            'n_iterations': ROBUST_N_ITERATIONS
        }
    
    except Exception as e:
        print_progress(f"❌ Robust ACO failed for {pattern}: {e}", 1)
        return {'success': False, 'pattern': pattern, 'error': str(e)}

def run_regular_aco(pattern):
    """Run regular ACO optimization for a traffic pattern."""
    print_progress(f"🐜 Running Regular ACO for '{pattern}' pattern...")
    print_progress(f"Single-seed training (traditional approach)", 1)
    
    try:
        # Generate scenario
        scenario = create_traffic_scenario(
            grid_size=GRID_SIZE,
            n_vehicles=N_VEHICLES,
            simulation_time=SIMULATION_TIME,
            pattern=pattern
        )
        
        if not scenario:
            return {'success': False, 'error': 'Scenario generation failed'}
        
        print_progress(f"✅ Scenario generated for {pattern}", 1)
        
        # Initialize regular optimizer
        optimizer = ACOTrafficOptimizer(
            sumo_config=scenario['config_file'],
            n_ants=REGULAR_N_ANTS,
            n_iterations=REGULAR_N_ITERATIONS,
            scenario_vehicles=N_VEHICLES,
            simulation_time=SIMULATION_TIME,
            show_plots=False,
            show_sumo_gui=False,
            compare_baseline=True
        )
        
        # Run optimization
        start_time = time.time()
        best_solution, best_cost, optimization_data, baseline_comparison = optimizer.optimize()
        end_time = time.time()
        
        # Calculate improvement
        improvement = None
        if baseline_comparison and isinstance(baseline_comparison, dict):
            baseline_cost = baseline_comparison.get('baseline_cost', 0)
            optimized_cost = baseline_comparison.get('optimized_cost', best_cost)
            if baseline_cost > 0:
                improvement = ((baseline_cost - optimized_cost) / baseline_cost) * 100
        
        return {
            'success': True,
            'pattern': pattern,
            'algorithm': 'regular_aco',
            'best_solution': best_solution,
            'best_cost': best_cost,
            'improvement': improvement,
            'baseline_comparison': baseline_comparison,
            'optimization_time': end_time - start_time,
            'n_ants': REGULAR_N_ANTS,
            'n_iterations': REGULAR_N_ITERATIONS
        }
    
    except Exception as e:
        print_progress(f"❌ Regular ACO failed for {pattern}: {e}", 1)
        return {'success': False, 'pattern': pattern, 'error': str(e)}

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_comparison_plots(robust_results, regular_results):
    """Create comparison plots between robust and regular ACO."""
    
    # Extract data for plotting
    patterns = []
    robust_costs = []
    regular_costs = []
    robust_improvements = []
    regular_improvements = []
    robust_times = []
    regular_times = []
    
    for pattern in PATTERNS:
        # Find results for this pattern
        robust_result = next((r for r in robust_results if r.get('pattern') == pattern and r.get('success')), None)
        regular_result = next((r for r in regular_results if r.get('pattern') == pattern and r.get('success')), None)
        
        if robust_result and regular_result:
            patterns.append(pattern.capitalize())
            robust_costs.append(robust_result['best_cost'])
            regular_costs.append(regular_result['best_cost'])
            robust_improvements.append(robust_result.get('improvement', 0) or 0)
            regular_improvements.append(regular_result.get('improvement', 0) or 0)
            robust_times.append(robust_result['optimization_time'])
            regular_times.append(regular_result['optimization_time'])
    
    if not patterns:
        print("⚠️ No matching results found for plotting")
        return
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🌱 Robust ACO vs Regular ACO Comparison', fontsize=16, fontweight='bold')
    
    x = np.arange(len(patterns))
    width = 0.35
    
    # Plot 1: Best Costs
    ax1.bar(x - width/2, robust_costs, width, label='Robust ACO', color='darkgreen', alpha=0.8)
    ax1.bar(x + width/2, regular_costs, width, label='Regular ACO', color='darkred', alpha=0.8)
    ax1.set_xlabel('Traffic Pattern')
    ax1.set_ylabel('Best Cost (seconds)')
    ax1.set_title('Best Solution Cost')
    ax1.set_xticks(x)
    ax1.set_xticklabels(patterns)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (robust, regular) in enumerate(zip(robust_costs, regular_costs)):
        ax1.text(i - width/2, robust + max(robust_costs) * 0.01, f'{robust:.1f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax1.text(i + width/2, regular + max(regular_costs) * 0.01, f'{regular:.1f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 2: Improvements over Baseline
    ax2.bar(x - width/2, robust_improvements, width, label='Robust ACO', color='darkgreen', alpha=0.8)
    ax2.bar(x + width/2, regular_improvements, width, label='Regular ACO', color='darkred', alpha=0.8)
    ax2.set_xlabel('Traffic Pattern')
    ax2.set_ylabel('Improvement over Baseline (%)')
    ax2.set_title('Improvement over Baseline')
    ax2.set_xticks(x)
    ax2.set_xticklabels(patterns)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on bars
    for i, (robust, regular) in enumerate(zip(robust_improvements, regular_improvements)):
        ax2.text(i - width/2, robust + max(max(robust_improvements), max(regular_improvements)) * 0.05, 
                f'{robust:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax2.text(i + width/2, regular + max(max(robust_improvements), max(regular_improvements)) * 0.05, 
                f'{regular:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 3: Optimization Time
    ax3.bar(x - width/2, robust_times, width, label='Robust ACO', color='darkgreen', alpha=0.8)
    ax3.bar(x + width/2, regular_times, width, label='Regular ACO', color='darkred', alpha=0.8)
    ax3.set_xlabel('Traffic Pattern')
    ax3.set_ylabel('Optimization Time (seconds)')
    ax3.set_title('Optimization Time')
    ax3.set_xticks(x)
    ax3.set_xticklabels(patterns)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (robust, regular) in enumerate(zip(robust_times, regular_times)):
        ax3.text(i - width/2, robust + max(robust_times) * 0.01, f'{robust:.0f}s', 
                ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax3.text(i + width/2, regular + max(regular_times) * 0.01, f'{regular:.0f}s', 
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 4: Direct Performance Comparison
    better_robust = sum(1 for r, reg in zip(robust_costs, regular_costs) if r < reg)
    better_regular = sum(1 for r, reg in zip(robust_costs, regular_costs) if r > reg)
    tied = sum(1 for r, reg in zip(robust_costs, regular_costs) if abs(r - reg) < 1.0)
    
    labels = ['Robust ACO\nBetter', 'Regular ACO\nBetter', 'Tied']
    sizes = [better_robust, better_regular, tied]
    colors = ['darkgreen', 'darkred', 'gray']
    
    ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.0f', startangle=90, alpha=0.8)
    ax4.set_title('Performance Comparison\n(Lower Cost is Better)')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join("results", "simple_robust_comparison.png")
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plot saved to: {plot_file}")
    
    plt.show()

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def main():
    """Run the complete robust vs regular ACO comparison."""
    print("🌱🐜 SIMPLE ROBUST ACO vs REGULAR ACO COMPARISON")
    print("=" * 80)
    print("This analysis compares Robust ACO (multi-seed) vs Regular ACO (single-seed)")
    print("across different traffic patterns.\n")
    
    print("📋 Configuration:")
    print(f"   Grid: {GRID_SIZE}x{GRID_SIZE} ({GRID_SIZE * GRID_SIZE} intersections)")
    print(f"   Vehicles per scenario: {N_VEHICLES}")
    print(f"   Simulation time: {SIMULATION_TIME}s")
    print(f"   Patterns: {', '.join(PATTERNS)}")
    print(f"   Robust ACO: {ROBUST_N_ANTS} ants × {ROBUST_N_ITERATIONS} iterations, {TRAINING_SEEDS} training seeds")
    print(f"   Regular ACO: {REGULAR_N_ANTS} ants × {REGULAR_N_ITERATIONS} iterations")
    
    start_time = time.time()
    
    # ========================================================================
    # PHASE 1: ROBUST ACO OPTIMIZATION
    # ========================================================================
    print("\n🌱 PHASE 1: ROBUST ACO OPTIMIZATION")
    print("-" * 50)
    
    robust_results = []
    for pattern in PATTERNS:
        result = run_robust_aco(pattern)
        robust_results.append(result)
        if result['success']:
            improvement = result.get('improvement', 0) or 0
            print_progress(f"✅ {pattern} completed! Cost: {result['best_cost']:.1f}, "
                          f"Improvement: {improvement:.1f}%", 1)
        else:
            print_progress(f"❌ {pattern} failed: {result.get('error', 'Unknown error')}", 1)
        print()
    
    # ========================================================================
    # PHASE 2: REGULAR ACO OPTIMIZATION
    # ========================================================================
    print("🐜 PHASE 2: REGULAR ACO OPTIMIZATION")
    print("-" * 50)
    
    regular_results = []
    for pattern in PATTERNS:
        result = run_regular_aco(pattern)
        regular_results.append(result)
        if result['success']:
            improvement = result.get('improvement', 0) or 0
            print_progress(f"✅ {pattern} completed! Cost: {result['best_cost']:.1f}, "
                          f"Improvement: {improvement:.1f}%", 1)
        else:
            print_progress(f"❌ {pattern} failed: {result.get('error', 'Unknown error')}", 1)
        print()
    
    # ========================================================================
    # PHASE 3: ANALYSIS AND VISUALIZATION
    # ========================================================================
    print("📊 PHASE 3: ANALYSIS AND VISUALIZATION")
    print("-" * 50)
    
    # Create comparison plots
    create_comparison_plots(robust_results, regular_results)
    
    # Calculate summary statistics
    successful_robust = [r for r in robust_results if r.get('success')]
    successful_regular = [r for r in regular_results if r.get('success')]
    
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"   Successful runs: Robust {len(successful_robust)}/{len(PATTERNS)}, "
          f"Regular {len(successful_regular)}/{len(PATTERNS)}")
    
    if successful_robust and successful_regular:
        avg_robust_cost = np.mean([r['best_cost'] for r in successful_robust])
        avg_regular_cost = np.mean([r['best_cost'] for r in successful_regular])
        avg_robust_time = np.mean([r['optimization_time'] for r in successful_robust])
        avg_regular_time = np.mean([r['optimization_time'] for r in successful_regular])
        
        print(f"   Average costs: Robust {avg_robust_cost:.1f}s, Regular {avg_regular_cost:.1f}s")
        print(f"   Average times: Robust {avg_robust_time:.1f}s, Regular {avg_regular_time:.1f}s")
        
        if avg_robust_cost < avg_regular_cost:
            print(f"   🌟 Robust ACO is {((avg_regular_cost - avg_robust_cost) / avg_regular_cost) * 100:.1f}% better on average")
        elif avg_regular_cost < avg_robust_cost:
            print(f"   🚀 Regular ACO is {((avg_robust_cost - avg_regular_cost) / avg_robust_cost) * 100:.1f}% better on average")
        else:
            print("   🤝 Both approaches perform similarly on average")
    
    # Save results
    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': {
            'grid_size': GRID_SIZE,
            'n_vehicles': N_VEHICLES,
            'simulation_time': SIMULATION_TIME,
            'patterns': PATTERNS,
            'robust_aco': {
                'n_ants': ROBUST_N_ANTS,
                'n_iterations': ROBUST_N_ITERATIONS,
                'training_seeds': TRAINING_SEEDS,
                'exploration_rate': EXPLORATION_RATE
            },
            'regular_aco': {
                'n_ants': REGULAR_N_ANTS,
                'n_iterations': REGULAR_N_ITERATIONS
            }
        },
        'robust_results': robust_results,
        'regular_results': regular_results
    }
    
    save_results(all_results, 'simple_robust_comparison_results.json')
    
    end_time = time.time()
    print(f"\n🎉 Analysis completed in {end_time - start_time:.1f} seconds")
    print("💡 Robust ACO trains on multiple seeds to prevent overfitting and improve generalization")
    
    return all_results

if __name__ == "__main__":
    results = main()
