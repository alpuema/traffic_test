#!/usr/bin/env python3
"""
Traffic Pattern Comparison Analysis

This script performs a comprehensive analysis of ACO optimization across
different traffic patterns with the following components:

1. Trains ACO on three different traffic patterns (commuter, industrial, random)
2. Plots convergence comparison with baseline costs in a single figure
3. Evaluates robustness by testing each solution on 10 different seeds
4. Provides statistical comparison of optimized vs baseline performance

Author: Traffic Optimization System
Date: August 2025
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simplified_traffic import (
    generate_network_and_routes, 
    save_optimized_solution, 
    load_solution,
    evaluate_solution_with_new_seed
)
from src.optimize import ACOTrafficOptimizer, evaluate_existing_solution
from src.optimization.simple_aco import (
    create_baseline_solution, 
    evaluate_solution, 
    calculate_cost,
    run_traditional_aco_optimization
)

# ============================================================================
# ANALYSIS PARAMETERS - MODIFY THESE TO CUSTOMIZE THE COMPARISON
# ============================================================================

# Scenario Configuration (consistent across all patterns)
GRID_SIZE = 4           # Grid dimensions
N_VEHICLES = 20         # Number of vehicles per scenario
SIMULATION_TIME = 2000  # Simulation duration in seconds

# ACO Parameters (consistent across all patterns)
N_ANTS = 50            # Number of ants per iteration
N_ITERATIONS = 100      # Number of optimization iterations
ALPHA = 1.0            # Pheromone importance weight
BETA = 2.0             # Heuristic importance weight
EVAPORATION_RATE = 0.1 # Pheromone evaporation rate
EXPLORATION_RATE = 0.15 # Pure exploration probability

# Analysis Configuration
TRAINING_SEED = 42      # Seed for training each pattern
TEST_SEEDS = [123, 456, 789, 999, 111, 222, 333, 444, 555, 666]  # 10 seeds for evaluation
TRAFFIC_PATTERNS = ['commuter', 'industrial', 'random']  # Three patterns to analyze

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

def evaluate_baseline_on_seed(test_seed, pattern, grid_size, n_vehicles, sim_time, phase_types):
    """
    Evaluate baseline solution (30s green, 4s yellow) on a specific seed and pattern.
    
    Args:
        test_seed: Random seed for traffic generation
        pattern: Traffic pattern ('commuter', 'industrial', 'random')
        grid_size: Grid dimensions
        n_vehicles: Number of vehicles
        sim_time: Simulation time
        phase_types: Phase type information
        
    Returns:
        Dictionary with baseline evaluation results
    """
    try:
        # Generate scenario with test seed
        scenario = generate_network_and_routes(
            grid_size=grid_size,
            n_vehicles=n_vehicles,
            sim_time=sim_time,
            pattern=pattern,
            seed=test_seed
        )
        
        if not scenario['success']:
            return {'success': False, 'error': 'Scenario generation failed'}
        
        # Create baseline solution
        baseline_solution = create_baseline_solution(phase_types, green_duration=30, yellow_duration=4)
        
        # Evaluate baseline solution
        temp_dir = f"temp_baseline_{test_seed}"
        os.makedirs(temp_dir, exist_ok=True)
        
        metrics = evaluate_solution(
            baseline_solution, 
            scenario['files']['network'], 
            scenario['files']['routes'], 
            temp_dir
        )
        
        cost = calculate_cost(metrics)
        avg_time = metrics['total_time'] / metrics['vehicles'] if metrics['vehicles'] > 0 else 0
        
        # Cleanup
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        return {
            'success': True,
            'cost': cost,
            'avg_time': avg_time,
            'metrics': metrics
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def run_aco_for_pattern(pattern, training_seed):
    """
    Run ACO optimization for a specific traffic pattern.
    
    Args:
        pattern: Traffic pattern name
        training_seed: Random seed for training
        
    Returns:
        Dictionary with optimization results and metadata
    """
    print_progress(f"🐜 Starting ACO optimization for '{pattern}' pattern...")
    
    # Configure ACO parameters
    config = {
        'grid_size': GRID_SIZE,
        'n_vehicles': N_VEHICLES,
        'simulation_time': SIMULATION_TIME,
        'n_ants': N_ANTS,
        'n_iterations': N_ITERATIONS,
        'evaporation_rate': EVAPORATION_RATE,
        'exploration_rate': EXPLORATION_RATE,
        'pheromone_weight': ALPHA,
        'heuristic_weight': BETA
    }
    
    try:
        # Generate training scenario
        scenario = generate_network_and_routes(
            grid_size=GRID_SIZE,
            n_vehicles=N_VEHICLES,
            sim_time=SIMULATION_TIME,
            pattern=pattern,
            seed=training_seed
        )
        
        if not scenario['success']:
            return {'success': False, 'error': f'Scenario generation failed for {pattern}'}
        
        print_progress(f"✅ Training scenario generated for {pattern}", 1)
        
        # Run ACO optimization using the traditional ACO function
        start_time = time.time()
        
        # Temporarily modify global parameters for this run
        import src.optimization.simple_aco as aco_module
        original_grid = aco_module.GRID_SIZE
        original_vehicles = aco_module.N_VEHICLES
        original_sim_time = aco_module.SIMULATION_TIME
        original_ants = aco_module.N_ANTS
        original_iterations = aco_module.N_ITERATIONS
        original_evap = aco_module.EVAPORATION_RATE
        original_explore = aco_module.EXPLORATION_RATE
        original_alpha = aco_module.ALPHA
        original_beta = aco_module.BETA
        original_show_progress = aco_module.SHOW_PROGRESS
        original_show_plots = aco_module.SHOW_PLOTS
        
        # Apply configuration
        aco_module.GRID_SIZE = GRID_SIZE
        aco_module.N_VEHICLES = N_VEHICLES
        aco_module.SIMULATION_TIME = SIMULATION_TIME
        aco_module.N_ANTS = N_ANTS
        aco_module.N_ITERATIONS = N_ITERATIONS
        aco_module.EVAPORATION_RATE = EVAPORATION_RATE
        aco_module.EXPLORATION_RATE = EXPLORATION_RATE
        aco_module.ALPHA = ALPHA
        aco_module.BETA = BETA
        aco_module.SHOW_PROGRESS = True  # Reduce noise
        aco_module.SHOW_PLOTS = False 
        
        # Run optimization
        results = run_traditional_aco_optimization(
            config=config,
            show_plots_override=False,
            show_gui_override=False,
            compare_baseline=True
        )
        
        # Restore original parameters
        aco_module.GRID_SIZE = original_grid
        aco_module.N_VEHICLES = original_vehicles
        aco_module.SIMULATION_TIME = original_sim_time
        aco_module.N_ANTS = original_ants
        aco_module.N_ITERATIONS = original_iterations
        aco_module.EVAPORATION_RATE = original_evap
        aco_module.EXPLORATION_RATE = original_explore
        aco_module.ALPHA = original_alpha
        aco_module.BETA = original_beta
        aco_module.SHOW_PROGRESS = original_show_progress
        aco_module.SHOW_PLOTS = original_show_plots
        
        training_time = time.time() - start_time
        
        if results['success']:
            print_progress(f"✅ {pattern} optimization completed in {training_time:.1f}s", 1)
            print_progress(f"   Best cost: {results['best_cost']:.1f}", 1)
            
            # Extract convergence data
            cost_history = results['cost_history']
            baseline_comparison = results.get('baseline_comparison')
            
            return {
                'success': True,
                'pattern': pattern,
                'best_cost': results['best_cost'],
                'best_solution': results['best_solution'],
                'cost_history': cost_history,
                'baseline_comparison': baseline_comparison,
                'phase_types': results['phase_types'],
                'training_time': training_time,
                'scenario_files': scenario['files']
            }
        else:
            return {'success': False, 'error': results.get('error', 'Unknown optimization error')}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

def evaluate_cross_seed_performance(pattern_results, pattern):
    """
    Evaluate optimized solution across multiple seeds and compare with baseline.
    
    Args:
        pattern_results: Results from training phase
        pattern: Traffic pattern name
        
    Returns:
        Dictionary with cross-seed evaluation statistics
    """
    print_progress(f"🔍 Evaluating {pattern} solution robustness across {len(TEST_SEEDS)} seeds...")
    
    optimized_costs = []
    baseline_costs = []
    successful_evaluations = 0
    
    for i, test_seed in enumerate(TEST_SEEDS, 1):
        print_progress(f"Testing seed {test_seed} ({i}/{len(TEST_SEEDS)})", 1)
        
        try:
            # Generate test scenario
            scenario = generate_network_and_routes(
                grid_size=GRID_SIZE,
                n_vehicles=N_VEHICLES,
                sim_time=SIMULATION_TIME,
                pattern=pattern,
                seed=test_seed
            )
            
            if not scenario['success']:
                print_progress(f"⚠️ Scenario generation failed for seed {test_seed}", 2)
                continue
            
            # Evaluate optimized solution
            temp_dir = f"temp_opt_{pattern}_{test_seed}"
            os.makedirs(temp_dir, exist_ok=True)
            
            opt_metrics = evaluate_solution(
                pattern_results['best_solution'],
                scenario['files']['network'],
                scenario['files']['routes'],
                temp_dir
            )
            opt_cost = calculate_cost(opt_metrics)
            
            # Evaluate baseline solution
            baseline_result = evaluate_baseline_on_seed(
                test_seed, pattern, GRID_SIZE, N_VEHICLES, SIMULATION_TIME, 
                pattern_results['phase_types']
            )
            
            # Cleanup
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            if baseline_result['success'] and opt_cost != float('inf'):
                optimized_costs.append(opt_cost)
                baseline_costs.append(baseline_result['cost'])
                successful_evaluations += 1
                
                improvement = ((baseline_result['cost'] - opt_cost) / baseline_result['cost']) * 100
                print_progress(f"✅ Opt: {opt_cost:.1f}, Base: {baseline_result['cost']:.1f}, Improvement: {improvement:+.1f}%", 2)
            else:
                error_msg = baseline_result.get('error', 'Invalid optimized cost')
                print_progress(f"❌ Evaluation failed: {error_msg}", 2)
                
        except Exception as e:
            print_progress(f"❌ Error with seed {test_seed}: {e}", 2)
    
    # Calculate statistics
    if successful_evaluations > 0:
        avg_opt_cost = np.mean(optimized_costs)
        avg_base_cost = np.mean(baseline_costs)
        std_opt_cost = np.std(optimized_costs)
        std_base_cost = np.std(baseline_costs)
        
        overall_improvement = ((avg_base_cost - avg_opt_cost) / avg_base_cost) * 100
        
        print_progress(f"📊 {pattern} Cross-Seed Results ({successful_evaluations}/{len(TEST_SEEDS)} successful):", 1)
        print_progress(f"   Optimized: μ={avg_opt_cost:.1f}, σ={std_opt_cost:.1f}", 2)
        print_progress(f"   Baseline:  μ={avg_base_cost:.1f}, σ={std_base_cost:.1f}", 2)
        print_progress(f"   Overall improvement: {overall_improvement:+.1f}%", 2)
        
        return {
            'success': True,
            'pattern': pattern,
            'successful_evaluations': successful_evaluations,
            'optimized_costs': optimized_costs,
            'baseline_costs': baseline_costs,
            'avg_optimized_cost': avg_opt_cost,
            'avg_baseline_cost': avg_base_cost,
            'std_optimized_cost': std_opt_cost,
            'std_baseline_cost': std_base_cost,
            'overall_improvement': overall_improvement
        }
    else:
        print_progress(f"❌ No successful evaluations for {pattern}", 1)
        return {'success': False, 'error': 'No successful cross-seed evaluations'}

def create_comparison_plot(all_results, cross_seed_results):
    """
    Create a comprehensive comparison plot showing:
    1. Convergence curves for all three patterns
    2. Baseline costs as dashed lines
    3. Final average costs from cross-seed evaluation
    """
    try:
        import matplotlib.pyplot as plt
        
        print_progress("📊 Creating comprehensive comparison plot...")
        
        # Set up the plot with landscape aspect ratio
        plt.figure(figsize=(16, 6))
        
        # Set larger font sizes
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12
        })
        
        # Color scheme for patterns
        colors = {
            'commuter': 'blue',
            'industrial': 'green', 
            'random': 'orange'
        }
        
        # Plot convergence curves for each pattern
        max_iterations = 0
        for pattern_result in all_results:
            if pattern_result['success']:
                pattern = pattern_result['pattern']
                cost_history = pattern_result['cost_history']
                iterations = range(len(cost_history))
                max_iterations = max(max_iterations, len(cost_history))
                
                color = colors.get(pattern, 'black')
                plt.plot(iterations, cost_history, 
                        color=color, linestyle='-', marker='o', 
                        linewidth=2.5, markersize=5, 
                        label=f'{pattern.capitalize()} ACO')
                
                # Add baseline as dashed line
                baseline_comp = pattern_result.get('baseline_comparison')
                if baseline_comp and 'baseline' in baseline_comp:
                    baseline_cost = baseline_comp['baseline']['cost']
                    if baseline_cost != float('inf'):
                        plt.axhline(y=baseline_cost, 
                                   color=color, linestyle='--', linewidth=2, 
                                   alpha=0.7, label=f'{pattern.capitalize()} Baseline')
        
        # Formatting
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Cost', fontsize=14)
        plt.title('ACO Convergence Comparison Across Traffic Patterns', fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        # Set x-ticks every 5 iterations to avoid clutter
        if max_iterations > 0:
            tick_positions = list(range(0, max_iterations, 5))
            if max_iterations - 1 not in tick_positions:  # Add final iteration
                tick_positions.append(max_iterations - 1)
            plt.xticks(tick_positions)
        
        plt.tight_layout()
        
        # Save plot
        if SAVE_RESULTS:
            results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, 'traffic_pattern_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print_progress(f"📊 Comparison plot saved: {plot_path}")
        
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        print_progress(f"❌ Error creating plot: {e}")

def save_analysis_results(all_results, cross_seed_results):
    """Save comprehensive analysis results to JSON files."""
    if not SAVE_RESULTS:
        return
        
    try:
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Prepare data for JSON serialization
        analysis_summary = {
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {
                'grid_size': GRID_SIZE,
                'n_vehicles': N_VEHICLES,
                'simulation_time': SIMULATION_TIME,
                'n_ants': N_ANTS,
                'n_iterations': N_ITERATIONS,
                'training_seed': TRAINING_SEED,
                'test_seeds': TEST_SEEDS
            },
            'patterns': {}
        }
        
        # Add results for each pattern
        for pattern_result, cross_result in zip(all_results, cross_seed_results):
            if pattern_result['success']:
                pattern = pattern_result['pattern']
                
                # Training results
                training_data = {
                    'training_cost': pattern_result['best_cost'],
                    'training_time': pattern_result['training_time'],
                    'convergence_history': pattern_result['cost_history']
                }
                
                # Cross-seed results
                if cross_result['success']:
                    cross_seed_data = {
                        'successful_evaluations': cross_result['successful_evaluations'],
                        'avg_optimized_cost': cross_result['avg_optimized_cost'],
                        'avg_baseline_cost': cross_result['avg_baseline_cost'],
                        'std_optimized_cost': cross_result['std_optimized_cost'],
                        'std_baseline_cost': cross_result['std_baseline_cost'],
                        'overall_improvement': cross_result['overall_improvement'],
                        'optimized_costs': cross_result['optimized_costs'],
                        'baseline_costs': cross_result['baseline_costs']
                    }
                else:
                    cross_seed_data = {'error': cross_result.get('error', 'Unknown error')}
                
                # Baseline comparison from training
                baseline_data = {}
                baseline_comp = pattern_result.get('baseline_comparison')
                if baseline_comp:
                    baseline_data = {
                        'training_baseline_cost': baseline_comp['baseline']['cost'],
                        'training_improvement': baseline_comp['improvement']['percent']
                    }
                
                analysis_summary['patterns'][pattern] = {
                    'training': training_data,
                    'cross_seed_evaluation': cross_seed_data,
                    'baseline_comparison': baseline_data
                }
        
        # Save summary
        summary_path = os.path.join(results_dir, 'traffic_pattern_analysis.json')
        with open(summary_path, 'w') as f:
            json.dump(analysis_summary, f, indent=2, default=str)
        
        print_progress(f"💾 Analysis results saved: {summary_path}")
        
    except Exception as e:
        print_progress(f"❌ Error saving results: {e}")

def main():
    """Run the comprehensive traffic pattern comparison analysis."""
    
    print("🚦 TRAFFIC PATTERN COMPARISON ANALYSIS")
    print("=" * 60)
    print()
    print("This analysis will:")
    print("  1. Train ACO on three traffic patterns (commuter, industrial, random)")
    print("  2. Plot convergence comparison with baselines")
    print("  3. Test robustness across 10 different traffic seeds")
    print("  4. Provide statistical performance comparison")
    print()
    
    # Display configuration
    print("📋 Analysis Configuration:")
    print(f"   Scenario: {GRID_SIZE}x{GRID_SIZE} grid, {N_VEHICLES} vehicles, {SIMULATION_TIME}s")
    print(f"   ACO: {N_ANTS} ants × {N_ITERATIONS} iterations")
    print(f"   Training seed: {TRAINING_SEED}")
    print(f"   Test seeds: {len(TEST_SEEDS)} different seeds")
    print(f"   Patterns: {', '.join(TRAFFIC_PATTERNS)}")
    print()
    
    input("Press Enter to start analysis (this may take several minutes)...")
    print()
    
    # ========================================================================
    # PHASE 1: TRAIN ACO FOR EACH TRAFFIC PATTERN
    # ========================================================================
    
    print("🎓 PHASE 1: Training ACO for Each Pattern")
    print("-" * 50)
    
    all_results = []
    total_training_time = 0
    
    for i, pattern in enumerate(TRAFFIC_PATTERNS, 1):
        print(f"\n📍 Pattern {i}/{len(TRAFFIC_PATTERNS)}: {pattern.upper()}")
        print("-" * 30)
        
        result = run_aco_for_pattern(pattern, TRAINING_SEED)
        all_results.append(result)
        
        if result['success']:
            total_training_time += result['training_time']
            baseline_comp = result.get('baseline_comparison')
            if baseline_comp:
                baseline_cost = baseline_comp['baseline']['cost']
                training_improvement = baseline_comp['improvement']['percent']
                print_progress(f"✅ Training complete: Cost {result['best_cost']:.1f} vs Baseline {baseline_cost:.1f} ({training_improvement:+.1f}%)")
            else:
                print_progress(f"✅ Training complete: Cost {result['best_cost']:.1f}")
        else:
            print_progress(f"❌ Training failed: {result['error']}")
    
    print(f"\n✅ All training completed in {total_training_time:.1f}s total")
    
    # ========================================================================
    # PHASE 2: CROSS-SEED ROBUSTNESS EVALUATION
    # ========================================================================
    
    print(f"\n🔍 PHASE 2: Cross-Seed Robustness Evaluation")
    print("-" * 50)
    
    cross_seed_results = []
    
    for pattern_result in all_results:
        if pattern_result['success']:
            cross_result = evaluate_cross_seed_performance(pattern_result, pattern_result['pattern'])
            cross_seed_results.append(cross_result)
        else:
            cross_seed_results.append({'success': False, 'pattern': pattern_result.get('pattern', 'unknown')})
    
    # ========================================================================
    # PHASE 3: VISUALIZATION AND ANALYSIS
    # ========================================================================
    
    print(f"\n📊 PHASE 3: Results Analysis & Visualization")
    print("-" * 50)
    
    # Create comparison plot
    create_comparison_plot(all_results, cross_seed_results)
    
    # ========================================================================
    # PHASE 4: COMPREHENSIVE RESULTS SUMMARY
    # ========================================================================
    
    print(f"\n📋 PHASE 4: Comprehensive Results Summary")
    print("=" * 60)
    
    print("\n🏆 TRAINING PERFORMANCE COMPARISON:")
    print("-" * 40)
    training_costs = []
    for result in all_results:
        if result['success']:
            pattern = result['pattern']
            cost = result['best_cost']
            time_taken = result['training_time']
            training_costs.append((pattern, cost))
            
            baseline_comp = result.get('baseline_comparison')
            if baseline_comp:
                baseline_cost = baseline_comp['baseline']['cost']
                improvement = baseline_comp['improvement']['percent']
                print(f"   {pattern.capitalize():12}: Cost {cost:7.1f} vs Baseline {baseline_cost:7.1f} ({improvement:+5.1f}%) [{time_taken:4.1f}s]")
            else:
                print(f"   {pattern.capitalize():12}: Cost {cost:7.1f} (no baseline) [{time_taken:4.1f}s]")
    
    # Find best training pattern
    if training_costs:
        best_pattern, best_cost = min(training_costs, key=lambda x: x[1])
        print(f"\n   🥇 Best training performance: {best_pattern.capitalize()} (Cost: {best_cost:.1f})")
    
    print("\n🎯 CROSS-SEED ROBUSTNESS COMPARISON:")
    print("-" * 40)
    robustness_summary = []
    for cross_result in cross_seed_results:
        if cross_result['success']:
            pattern = cross_result['pattern']
            avg_improvement = cross_result['overall_improvement']
            std_opt = cross_result['std_optimized_cost']
            successful = cross_result['successful_evaluations']
            
            robustness_summary.append((pattern, avg_improvement, std_opt))
            
            status = "🎯" if avg_improvement > 0 else "⚠️"
            print(f"   {pattern.capitalize():12}: {status} {avg_improvement:+5.1f}% avg improvement, σ={std_opt:5.1f} ({successful}/{len(TEST_SEEDS)} seeds)")
    
    # Find most robust pattern
    if robustness_summary:
        # Sort by improvement percentage
        robust_ranking = sorted(robustness_summary, key=lambda x: x[1], reverse=True)
        best_robust_pattern, best_improvement, _ = robust_ranking[0]
        print(f"\n   🏆 Most robust pattern: {best_robust_pattern.capitalize()} ({best_improvement:+.1f}% avg improvement)")
        
        # Sort by consistency (lowest std deviation)
        consistent_ranking = sorted(robustness_summary, key=lambda x: x[2])
        most_consistent_pattern, _, lowest_std = consistent_ranking[0]
        print(f"   🎯 Most consistent pattern: {most_consistent_pattern.capitalize()} (σ={lowest_std:.1f})")
    
    print("\n💡 KEY INSIGHTS:")
    print("-" * 40)
    
    successful_patterns = [r for r in cross_seed_results if r['success']]
    if successful_patterns:
        # Overall performance insights
        improvements = [r['overall_improvement'] for r in successful_patterns]
        avg_overall_improvement = np.mean(improvements)
        
        if avg_overall_improvement > 5:
            print("   ✅ EXCELLENT: ACO consistently outperforms baseline across patterns")
        elif avg_overall_improvement > 0:
            print("   ✅ GOOD: ACO generally outperforms baseline")
        elif avg_overall_improvement > -5:
            print("   ⚠️  MIXED: ACO performance comparable to baseline")
        else:
            print("   ❌ POOR: ACO underperforms compared to simple baseline")
        
        # Pattern-specific insights
        best_improvements = [r for r in successful_patterns if r['overall_improvement'] > 0]
        if len(best_improvements) == len(successful_patterns):
            print("   🎯 All patterns benefit from ACO optimization")
        elif len(best_improvements) >= len(successful_patterns) * 0.67:
            print("   👍 Most patterns benefit from ACO optimization")
        else:
            print("   🤔 Mixed results - some patterns may need different approaches")
            
        # Consistency insights
        std_devs = [r['std_optimized_cost'] for r in successful_patterns]
        avg_std = np.mean(std_devs)
        if avg_std < 50:
            print("   📊 Solutions are highly consistent across seeds")
        elif avg_std < 100:
            print("   📊 Solutions show moderate consistency across seeds")
        else:
            print("   📊 Solutions show high variability across seeds")
    
    # Save detailed results
    if SAVE_RESULTS:
        save_analysis_results(all_results, cross_seed_results)
    
    # ========================================================================
    # FINAL SUMMARY TABLE
    # ========================================================================
    
    print("\n📋 FINAL RESULTS SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Pattern':<12} {'Training':<20} {'Evaluation (10 seeds)':<35} {'Improvement':<12}")
    print(f"{'':12} {'Baseline → Optimized':<20} {'Baseline Avg → Optimized Avg':<35} {'Over Baseline':<12}")
    print("-" * 80)
    
    for pattern_result, cross_result in zip(all_results, cross_seed_results):
        if pattern_result['success']:
            pattern = pattern_result['pattern']
            
            # Training results
            training_cost = pattern_result['best_cost']
            baseline_comp = pattern_result.get('baseline_comparison')
            training_baseline = baseline_comp['baseline']['cost'] if baseline_comp else 0
            
            # Cross-seed results
            if cross_result['success']:
                avg_opt = cross_result['avg_optimized_cost']
                avg_base = cross_result['avg_baseline_cost']
                improvement = cross_result['overall_improvement']
                
                print(f"{pattern.capitalize():<12} {training_baseline:6.1f} → {training_cost:6.1f}     "
                      f"{avg_base:8.1f} → {avg_opt:8.1f}               {improvement:+6.1f}%")
            else:
                print(f"{pattern.capitalize():<12} {training_baseline:6.1f} → {training_cost:6.1f}     "
                      f"{'Failed evaluation':<27}        {'N/A':<12}")
    
    print("-" * 80)
    print()
    
    print("\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
    print("=" * 60)
    
    return {
        'training_results': all_results,
        'cross_seed_results': cross_seed_results,
        'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

if __name__ == "__main__":
    # Check for help request
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print("Traffic Pattern Comparison Analysis")
        print("=" * 40)
        print("This script compares ACO optimization across different traffic patterns.")
        print("\nUsage: python traffic_pattern_comparison.py")
        print("\nThe script will:")
        print("  1. Train ACO on commuter, industrial, and random patterns")
        print("  2. Show convergence comparison with baselines in one plot")
        print("  3. Test each solution on 10 different seeds")
        print("  4. Provide statistical performance analysis")
        print("\nResults are saved to: results/traffic_pattern_analysis.json")
        print("Plots are saved to: results/traffic_pattern_comparison.png")
    else:
        try:
            results = main()
            print("\n✅ Analysis completed successfully!")
        except KeyboardInterrupt:
            print("\n\n⏹️  Analysis interrupted by user")
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            if VERBOSE:
                import traceback
                traceback.print_exc()
            sys.exit(1)
