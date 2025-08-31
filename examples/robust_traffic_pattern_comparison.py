#!/usr/bin/env python3
"""
Robust Traffic Pattern Comparison Analysis

This script performs a comprehensive analysis of Robust Multi-Seed ACO optimization 
across different traffic patterns with the following components:

1. Trains Robust ACO on three different traffic patterns (commuter, industrial, random)
2. Each pattern is trained using multiple seeds to prevent overfitting  
3. Plots convergence comparison with baseline costs in a single figure
4. Evaluates robustness by testing each solution on 10 different seeds
5. Compares robust vs single-seed performance to demonstrate generalization benefits
6. Provides statistical comparison of robust optimized vs baseline performance

The key difference from regular traffic pattern comparison is that this uses
the RobustACOTrafficOptimizer which trains on multiple seeds simultaneously,
preventing overfitting and improving generalization to unseen traffic scenarios.

Author: Traffic Optimization System  
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
# ANALYSIS PARAMETERS - MODIFY THESE TO CUSTOMIZE THE COMPARISON
# ============================================================================

# Scenario Configuration (consistent across all patterns)
GRID_SIZE = 4           # Grid dimensions
N_VEHICLES = 20         # Number of vehicles per scenario
SIMULATION_TIME = 2000  # Simulation duration in seconds

# Robust ACO Parameters (optimized for multi-seed training)
N_ANTS = 15            # Number of ants per iteration (fewer since each evaluation is more expensive)
N_ITERATIONS = 8       # Number of optimization iterations (fewer due to robust training)
TRAINING_SEEDS = 5     # Number of seeds to train on simultaneously (3-10 recommended)
EXPLORATION_RATE = 0.25 # Higher exploration for robustness (0.20-0.30)
EVAPORATION_RATE = 0.08 # Slightly lower evaporation for stability
ALPHA = 1.0            # Stop penalty weight
BETA = 2.0             # Heuristic importance weight

# Analysis Configuration
BASE_TRAINING_SEED = 42 # Base seed for scenario generation (robust ACO will use multiple derived seeds)
TEST_SEEDS = [123, 456, 789, 999, 111, 222, 333, 444, 555, 666]  # 10 seeds for evaluation
TRAFFIC_PATTERNS = ['commuter', 'industrial', 'random']  # Three patterns to analyze

# Comparison Options
COMPARE_WITH_REGULAR_ACO = True  # Compare robust vs regular single-seed ACO
REGULAR_ACO_PARAMS = {           # Parameters for regular ACO comparison
    'n_ants': 20,
    'n_iterations': 12,
    'evaporation_rate': 0.1
}

# Display Options
SHOW_PLOTS = True      # Show comparison plots
SAVE_RESULTS = True    # Save detailed results to files
VERBOSE = True         # Print detailed progress

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_progress(message, indent=0):
    """Print progress message with optional indentation."""
    if VERBOSE:
        prefix = "  " * indent
        print(f"{prefix}{message}")

def run_robust_aco_for_pattern(pattern, base_seed):
    """
    Run Robust ACO optimization for a specific traffic pattern.
    
    Args:
        pattern: Traffic pattern name
        base_seed: Base random seed for training scenario generation
        
    Returns:
        Dictionary with optimization results and metadata
    """
    print_progress(f"🐜🌱 Starting Robust ACO optimization for '{pattern}' pattern...")
    print_progress(f"Training on {TRAINING_SEEDS} different seeds to prevent overfitting", 1)
    
    try:
        # Generate base training scenario
        base_scenario = create_traffic_scenario(
            grid_size=GRID_SIZE,
            n_vehicles=N_VEHICLES,
            simulation_time=SIMULATION_TIME,
            pattern=pattern,
            seed=base_seed
        )
        
        if not base_scenario:
            return {'success': False, 'error': f'Base scenario generation failed for {pattern}'}
        
        print_progress(f"✅ Base training scenario generated for {pattern}", 1)
        
        # Create robust ACO optimizer
        start_time = time.time()
        
        optimizer = RobustACOTrafficOptimizer(
            sumo_config=base_scenario['config_file'],
            n_ants=N_ANTS,
            n_iterations=N_ITERATIONS,
            alpha=ALPHA,
            beta=BETA,
            rho=EVAPORATION_RATE,
            verbose=True,
            scenario_vehicles=N_VEHICLES,
            simulation_time=SIMULATION_TIME,
            show_plots=False,  # We'll create our own plots
            show_sumo_gui=False,
            compare_baseline=True,
            # Robust-specific parameters
            training_seeds=TRAINING_SEEDS,
            exploration_rate=EXPLORATION_RATE,
            validate_solution=True
        )
        
        # Run robust optimization
        print_progress(f"🚀 Running robust optimization (this may take longer due to multi-seed training)...", 1)
        
        best_solution, best_cost, optimization_data, baseline_comparison = optimizer.optimize()
        
        training_time = time.time() - start_time
        
        print_progress(f"✅ {pattern} robust optimization completed in {training_time:.1f}s", 1)
        print_progress(f"   Best robust cost: {best_cost:.1f}", 1)
        
        if baseline_comparison:
            improvement = baseline_comparison.get('improvement', {}).get('percent', 0)
            print_progress(f"   Multi-seed improvement: {improvement:.1f}%", 1)
        
        return {
            'success': True,
            'pattern': pattern,
            'best_cost': best_cost,
            'best_solution': best_solution,
            'optimization_data': optimization_data,
            'baseline_comparison': baseline_comparison,
            'training_time': training_time,
            'training_seeds': TRAINING_SEEDS,
            'base_scenario': base_scenario
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def run_regular_aco_for_pattern(pattern, base_seed):
    """
    Run regular single-seed ACO for comparison with robust ACO.
    
    Args:
        pattern: Traffic pattern name  
        base_seed: Random seed for training
        
    Returns:
        Dictionary with optimization results
    """
    print_progress(f"🐜 Running regular ACO for '{pattern}' pattern (single seed)...")
    
    try:
        # Generate training scenario
        scenario = create_traffic_scenario(
            grid_size=GRID_SIZE,
            n_vehicles=N_VEHICLES,
            simulation_time=SIMULATION_TIME,
            pattern=pattern,
            seed=base_seed
        )
        
        if not scenario:
            return {'success': False, 'error': f'Scenario generation failed for {pattern}'}
        
        # Create regular ACO optimizer
        start_time = time.time()
        
        optimizer = ACOTrafficOptimizer(
            sumo_config=scenario['config_file'],
            n_ants=REGULAR_ACO_PARAMS['n_ants'],
            n_iterations=REGULAR_ACO_PARAMS['n_iterations'],
            scenario_vehicles=N_VEHICLES,
            simulation_time=SIMULATION_TIME,
            show_plots=False,
            show_sumo_gui=False,
            compare_baseline=True
        )
        
        best_solution, best_cost, optimization_data, baseline_comparison = optimizer.optimize()
        
        training_time = time.time() - start_time
        
        print_progress(f"✅ {pattern} regular ACO completed in {training_time:.1f}s", 1)
        print_progress(f"   Best cost: {best_cost:.1f}", 1)
        
        return {
            'success': True,
            'pattern': pattern,
            'best_cost': best_cost,
            'best_solution': best_solution,
            'optimization_data': optimization_data,
            'baseline_comparison': baseline_comparison,
            'training_time': training_time,
            'scenario': scenario
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def evaluate_cross_seed_performance(pattern_results, pattern, solution_type="robust"):
    """
    Evaluate solution across multiple seeds and compare with baseline.
    
    Args:
        pattern_results: Results from training phase
        pattern: Traffic pattern name
        solution_type: "robust" or "regular" for labeling
        
    Returns:
        Dictionary with cross-seed evaluation statistics
    """
    print_progress(f"🔍 Evaluating {pattern} {solution_type} solution robustness across {len(TEST_SEEDS)} seeds...")
    
    optimized_costs = []
    baseline_costs = []
    successful_evaluations = 0
    
    for i, test_seed in enumerate(TEST_SEEDS, 1):
        print_progress(f"Testing seed {test_seed} ({i}/{len(TEST_SEEDS)})", 1)
        
        try:
            # Generate test scenario
            test_scenario = create_traffic_scenario(
                grid_size=GRID_SIZE,
                n_vehicles=N_VEHICLES,
                simulation_time=SIMULATION_TIME,
                pattern=pattern,
                seed=test_seed
            )
            
            if not test_scenario:
                print_progress(f"⚠️ Scenario generation failed for seed {test_seed}", 2)
                continue
            
            # Evaluate both optimized and baseline solutions
            from src.optimization.simple_aco import evaluate_solution, calculate_cost, create_baseline_solution
            
            # Get phase types from optimization data
            phase_types = None
            if 'optimization_data' in pattern_results and pattern_results['optimization_data']:
                phase_types = pattern_results['optimization_data'].get('phase_types')
            
            if not phase_types:
                # Create dummy phase types for evaluation
                phase_types = {f"tls_{i}": ["G", "y", "r", "y"] for i in range(GRID_SIZE * GRID_SIZE)}
            
            # Create temporary directory for evaluation
            temp_dir = f"temp_eval_{pattern}_{solution_type}_{test_seed}"
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                # Evaluate optimized solution
                opt_metrics = evaluate_solution(
                    pattern_results['best_solution'],
                    test_scenario['files']['network'],
                    test_scenario['files']['routes'],
                    temp_dir
                )
                
                opt_cost = calculate_cost(opt_metrics)
                optimized_costs.append(opt_cost)
                
                # Evaluate baseline solution
                baseline_solution = create_baseline_solution(phase_types, green_duration=30, yellow_duration=4)
                base_metrics = evaluate_solution(
                    baseline_solution,
                    test_scenario['files']['network'],
                    test_scenario['files']['routes'],
                    temp_dir
                )
                
                base_cost = calculate_cost(base_metrics)
                baseline_costs.append(base_cost)
                
                successful_evaluations += 1
                
            finally:
                # Cleanup temp directory
                import shutil
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            
        except Exception as e:
            print_progress(f"⚠️ Evaluation failed for seed {test_seed}: {e}", 2)
            continue
    
    if successful_evaluations == 0:
        return {'success': False, 'error': 'No successful evaluations'}
    
    # Calculate statistics
    avg_optimized = np.mean(optimized_costs)
    avg_baseline = np.mean(baseline_costs)
    std_optimized = np.std(optimized_costs)
    std_baseline = np.std(baseline_costs)
    
    improvement = ((avg_baseline - avg_optimized) / avg_baseline) * 100
    
    print_progress(f"✅ Cross-seed evaluation completed for {pattern} {solution_type}", 1)
    print_progress(f"   Successful evaluations: {successful_evaluations}/{len(TEST_SEEDS)}", 1)
    print_progress(f"   Average optimized cost: {avg_optimized:.1f} ± {std_optimized:.1f}", 1)
    print_progress(f"   Average baseline cost: {avg_baseline:.1f} ± {std_baseline:.1f}", 1)
    print_progress(f"   Overall improvement: {improvement:.1f}%", 1)
    
    return {
        'success': True,
        'avg_optimized_cost': avg_optimized,
        'avg_baseline_cost': avg_baseline,
        'std_optimized_cost': std_optimized,
        'std_baseline_cost': std_baseline,
        'overall_improvement': improvement,
        'successful_evaluations': successful_evaluations,
        'individual_optimized_costs': optimized_costs,
        'individual_baseline_costs': baseline_costs
    }

def plot_robust_comparison(robust_results, regular_results=None):
    """
    Create comprehensive plots comparing robust ACO performance.
    
    Args:
        robust_results: Results from robust ACO training
        regular_results: Optional results from regular ACO for comparison
    """
    print_progress("📊 Creating robust ACO comparison plots...")
    
    # Create figure with subplots
    if regular_results:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    fig.suptitle('Robust Multi-Seed ACO: Traffic Pattern Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Training convergence comparison
    colors = ['#2E8B57', '#DC143C', '#4169E1']  # Green, Red, Blue
    patterns = ['commuter', 'industrial', 'random']
    
    for i, pattern in enumerate(patterns):
        if pattern in [r['pattern'] for r in robust_results if r['success']]:
            result = next(r for r in robust_results if r['success'] and r['pattern'] == pattern)
            
            # Get convergence data
            opt_data = result.get('optimization_data', {})
            cost_history = opt_data.get('cost_history', [])
            
            if cost_history:
                ax1.plot(cost_history, color=colors[i], linewidth=2.5, 
                        label=f'{pattern.capitalize()} (Robust)', marker='o', markersize=4)
                
                # Add baseline comparison if available
                baseline_comp = result.get('baseline_comparison', {})
                if baseline_comp and 'baseline' in baseline_comp:
                    baseline_cost = baseline_comp['baseline']['cost']
                    ax1.axhline(y=baseline_cost, color=colors[i], linestyle='--', alpha=0.7,
                               label=f'{pattern.capitalize()} Baseline')
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost')
    ax1.set_title('Robust ACO Convergence by Pattern')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training vs Cross-seed performance comparison
    patterns_with_results = [r['pattern'] for r in robust_results if r['success']]
    training_costs = [r['best_cost'] for r in robust_results if r['success']]
    
    # We'll need cross-seed results for this plot - let's create a placeholder
    ax2.bar(patterns_with_results, training_costs, color=colors[:len(patterns_with_results)], alpha=0.7)
    ax2.set_ylabel('Cost')  
    ax2.set_title('Robust ACO Training Performance')
    ax2.set_xticklabels(patterns_with_results, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # If we have regular ACO results, add comparison plots
    if regular_results:
        # Plot 3: Robust vs Regular training comparison
        robust_costs = [r['best_cost'] for r in robust_results if r['success']]
        regular_costs = [r['best_cost'] for r in regular_results if r['success']]
        
        x = np.arange(len(patterns_with_results))
        width = 0.35
        
        ax3.bar(x - width/2, robust_costs, width, label='Robust ACO', color='darkgreen', alpha=0.8)
        ax3.bar(x + width/2, regular_costs, width, label='Regular ACO', color='darkred', alpha=0.8)
        
        ax3.set_xlabel('Traffic Pattern')
        ax3.set_ylabel('Training Cost')
        ax3.set_title('Robust vs Regular ACO: Training Performance')
        ax3.set_xticks(x)
        ax3.set_xticklabels(patterns_with_results)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Training time comparison
        robust_times = [r['training_time'] for r in robust_results if r['success']]
        regular_times = [r['training_time'] for r in regular_results if r['success']]
        
        ax4.bar(x - width/2, robust_times, width, label='Robust ACO', color='darkblue', alpha=0.8)
        ax4.bar(x + width/2, regular_times, width, label='Regular ACO', color='orange', alpha=0.8)
        
        ax4.set_xlabel('Traffic Pattern')
        ax4.set_ylabel('Training Time (seconds)')
        ax4.set_title('Training Time Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(patterns_with_results)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if SAVE_RESULTS:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        plt.savefig(results_dir / "robust_traffic_pattern_comparison.png", dpi=300, bbox_inches='tight')
        print_progress(f"📊 Plots saved to: {results_dir / 'robust_traffic_pattern_comparison.png'}")
    
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

def save_analysis_results(robust_results, cross_seed_results, regular_results=None, regular_cross_results=None):
    """Save comprehensive analysis results to JSON file."""
    if not SAVE_RESULTS:
        return
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    analysis_data = {
        'analysis_type': 'robust_traffic_pattern_comparison',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'parameters': {
            'grid_size': GRID_SIZE,
            'n_vehicles': N_VEHICLES,
            'simulation_time': SIMULATION_TIME,
            'training_seeds': TRAINING_SEEDS,
            'n_ants': N_ANTS,
            'n_iterations': N_ITERATIONS,
            'exploration_rate': EXPLORATION_RATE,
            'evaporation_rate': EVAPORATION_RATE,
            'test_seeds': TEST_SEEDS,
            'traffic_patterns': TRAFFIC_PATTERNS
        },
        'robust_results': {
            'training': robust_results,
            'cross_seed_evaluation': cross_seed_results
        }
    }
    
    if regular_results:
        analysis_data['regular_aco_comparison'] = {
            'training': regular_results,
            'cross_seed_evaluation': regular_cross_results
        }
    
    filename = results_dir / "robust_traffic_pattern_analysis.json"
    with open(filename, 'w') as f:
        json.dump(analysis_data, f, indent=2, default=str)
    
    print_progress(f"💾 Analysis results saved to: {filename}")

def main():
    """Run comprehensive robust traffic pattern comparison."""
    
    print("🌱🐜 ROBUST MULTI-SEED ACO TRAFFIC PATTERN COMPARISON")
    print("=" * 80)
    print("This analysis compares Robust ACO performance across different traffic patterns.")
    print("Robust ACO trains on multiple seeds simultaneously to prevent overfitting.")
    print()
    print(f"📋 Configuration:")
    print(f"   Grid: {GRID_SIZE}x{GRID_SIZE} ({GRID_SIZE**2} intersections)")
    print(f"   Vehicles per scenario: {N_VEHICLES}")
    print(f"   Training seeds per pattern: {TRAINING_SEEDS}")
    print(f"   Robust ACO: {N_ANTS} ants × {N_ITERATIONS} iterations")
    print(f"   Cross-validation: {len(TEST_SEEDS)} test seeds")
    print(f"   Patterns: {', '.join(TRAFFIC_PATTERNS)}")
    
    if COMPARE_WITH_REGULAR_ACO:
        print(f"   Regular ACO comparison: {REGULAR_ACO_PARAMS['n_ants']} ants × {REGULAR_ACO_PARAMS['n_iterations']} iterations")
    print()
    
    # Phase 1: Train Robust ACO on all patterns
    print("🌱 PHASE 1: ROBUST ACO TRAINING")
    print("-" * 50)
    
    robust_results = []
    for pattern in TRAFFIC_PATTERNS:
        result = run_robust_aco_for_pattern(pattern, BASE_TRAINING_SEED)
        robust_results.append(result)
        
        if not result['success']:
            print_progress(f"❌ Robust ACO failed for {pattern}: {result['error']}")
    
    successful_robust = [r for r in robust_results if r['success']]
    print_progress(f"✅ Robust ACO training completed: {len(successful_robust)}/{len(TRAFFIC_PATTERNS)} patterns successful")
    
    # Phase 2: Optional Regular ACO comparison
    regular_results = []
    if COMPARE_WITH_REGULAR_ACO:
        print("\n🐜 PHASE 2: REGULAR ACO COMPARISON")
        print("-" * 50)
        
        for pattern in TRAFFIC_PATTERNS:
            result = run_regular_aco_for_pattern(pattern, BASE_TRAINING_SEED)
            regular_results.append(result)
            
            if not result['success']:
                print_progress(f"❌ Regular ACO failed for {pattern}: {result['error']}")
        
        successful_regular = [r for r in regular_results if r['success']]
        print_progress(f"✅ Regular ACO comparison completed: {len(successful_regular)}/{len(TRAFFIC_PATTERNS)} patterns successful")
    
    # Phase 3: Cross-seed evaluation
    print("\n🔍 PHASE 3: CROSS-SEED ROBUSTNESS EVALUATION")
    print("-" * 60)
    
    cross_seed_results = []
    for result in successful_robust:
        cross_result = evaluate_cross_seed_performance(result, result['pattern'], "robust")
        cross_seed_results.append(cross_result)
    
    regular_cross_results = []
    if COMPARE_WITH_REGULAR_ACO:
        print("\n🔍 PHASE 3b: REGULAR ACO CROSS-SEED EVALUATION")
        print("-" * 60)
        
        for result in [r for r in regular_results if r['success']]:
            cross_result = evaluate_cross_seed_performance(result, result['pattern'], "regular")
            regular_cross_results.append(cross_result)
    
    # Phase 4: Analysis and visualization
    print("\n📊 PHASE 4: ANALYSIS AND VISUALIZATION")
    print("-" * 50)
    
    # Create plots
    plot_robust_comparison(
        successful_robust, 
        [r for r in regular_results if r['success']] if COMPARE_WITH_REGULAR_ACO else None
    )
    
    # Save results
    save_analysis_results(
        robust_results, cross_seed_results, 
        regular_results if COMPARE_WITH_REGULAR_ACO else None,
        regular_cross_results if COMPARE_WITH_REGULAR_ACO else None
    )
    
    # Phase 5: Summary report
    print("\n📋 FINAL SUMMARY REPORT")
    print("=" * 60)
    
    print("Pattern Analysis Results:")
    print("-" * 40)
    print(f"{'Pattern':<12} {'Training':<15} {'Cross-Seed Avg':<20} {'Improvement':<12}")
    print(f"{'':12} {'Cost':<15} {'Cost (Robust)':<20} {'%':<12}")
    print("-" * 60)
    
    for i, pattern_result in enumerate(successful_robust):
        pattern = pattern_result['pattern']
        training_cost = pattern_result['best_cost']
        
        # Get cross-seed results
        cross_result = cross_seed_results[i] if i < len(cross_seed_results) and cross_seed_results[i]['success'] else None
        
        if cross_result:
            avg_cross = cross_result['avg_optimized_cost']
            improvement = cross_result['overall_improvement']
            print(f"{pattern.capitalize():<12} {training_cost:6.1f}         {avg_cross:8.1f}              {improvement:+6.1f}%")
        else:
            print(f"{pattern.capitalize():<12} {training_cost:6.1f}         {'Failed':<20}        {'N/A':<12}")
    
    print("-" * 60)
    
    # Robust vs Regular comparison
    if COMPARE_WITH_REGULAR_ACO:
        print("\nRobust vs Regular ACO Comparison:")
        print("-" * 40)
        
        for i, pattern in enumerate(TRAFFIC_PATTERNS):
            robust_result = next((r for r in successful_robust if r['pattern'] == pattern), None)
            regular_result = next((r for r in regular_results if r['success'] and r['pattern'] == pattern), None)
            
            if robust_result and regular_result:
                robust_cross = cross_seed_results[i] if i < len(cross_seed_results) and cross_seed_results[i]['success'] else None
                regular_cross = regular_cross_results[i] if i < len(regular_cross_results) and regular_cross_results[i]['success'] else None
                
                print(f"\n{pattern.capitalize()} Pattern:")
                print(f"  Training Cost - Robust: {robust_result['best_cost']:.1f}, Regular: {regular_result['best_cost']:.1f}")
                
                if robust_cross and regular_cross:
                    print(f"  Cross-Seed Avg - Robust: {robust_cross['avg_optimized_cost']:.1f}, Regular: {regular_cross['avg_optimized_cost']:.1f}")
                    print(f"  Generalization - Robust: {robust_cross['overall_improvement']:.1f}%, Regular: {regular_cross['overall_improvement']:.1f}%")
    
    print("\n🎉 ROBUST TRAFFIC PATTERN ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Key Benefits of Robust ACO:")
    print("• Trains on multiple seeds to prevent overfitting")
    print("• Better generalization to unseen traffic patterns")
    print("• More consistent performance across different scenarios")
    print("• Adaptive learning focuses on challenging cases")
    
    return {
        'robust_results': robust_results,
        'cross_seed_results': cross_seed_results,
        'regular_results': regular_results if COMPARE_WITH_REGULAR_ACO else None,
        'regular_cross_results': regular_cross_results if COMPARE_WITH_REGULAR_ACO else None,
        'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

if __name__ == "__main__":
    # Check for help request
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print("Robust Traffic Pattern Comparison Analysis")
        print("=" * 45)
        print("This script compares Robust Multi-Seed ACO optimization across different traffic patterns.")
        print("\nUsage: python robust_traffic_pattern_comparison.py")
        print("\nThe script will:")
        print("  1. Train Robust ACO on commuter, industrial, and random patterns")
        print("  2. Each training uses multiple seeds to prevent overfitting")
        print("  3. Optionally compare with regular single-seed ACO")
        print("  4. Test each solution on 10 different seeds for robustness")
        print("  5. Provide statistical performance analysis and plots")
        print("\nResults are saved to: results/robust_traffic_pattern_analysis.json")
        print("Plots are saved to: results/robust_traffic_pattern_comparison.png")
        print("\nKey Benefits:")
        print("  • Prevents overfitting to specific traffic scenarios")
        print("  • Better generalization to unseen traffic patterns")
        print("  • More robust and consistent performance")
    else:
        try:
            results = main()
            print("\n✅ Robust analysis completed successfully!")
        except KeyboardInterrupt:
            print("\n\n⏹️  Analysis interrupted by user")
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            if VERBOSE:
                import traceback
                traceback.print_exc()
            sys.exit(1)
