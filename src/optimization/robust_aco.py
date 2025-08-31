"""
Robust Multi-Seed ACO for Traffic Light Optimization

This implementation reduces overfitting by training the ACO algorithm across
multiple traffic seeds, ensuring solutions generalize well to different
traffic scenarios.

Key features:
- Trains on N different traffic seeds simultaneously
- Evaluates solutions across all seeds to prevent overfitting
- Uses ensemble evaluation for robust cost calculation
- Maintains original ACO pheromone-based learning
- Compatible with existing interface

Author: Traffic Optimization System
Date: August 2025
"""

import xml.etree.ElementTree as ET
import subprocess
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import time
import random
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Import functions from the original ACO
from .simple_aco import (
    print_progress, get_project_paths, analyze_traffic_light_phases,
    apply_solution_to_network, create_sumo_config, parse_tripinfo_file,
    calculate_cost, create_baseline_solution, extract_files_from_sumo_config
)

# ============================================================================
# ROBUST ACO CONFIGURATION
# ============================================================================

# Default parameters (can be overridden by config)
DEFAULT_TRAINING_SEEDS = 5     # Number of different traffic seeds to train on
DEFAULT_VALIDATION_SEEDS = 3   # Number of seeds for final validation
SEED_WEIGHT_STRATEGY = 'equal' # 'equal', 'performance_weighted', 'adaptive'

# ============================================================================
# MULTI-SEED SCENARIO MANAGEMENT
# ============================================================================

def generate_multi_seed_scenarios(base_config, training_seeds):
    """
    Generate multiple traffic scenarios with different seeds for robust training.
    
    Args:
        base_config: Base scenario configuration
        training_seeds: List of seeds to generate scenarios for
    
    Returns:
        List of scenario dictionaries with file paths
    """
    from ..simplified_traffic import generate_network_and_routes
    
    scenarios = []
    
    print_progress(f"🌱 Generating {len(training_seeds)} training scenarios...")
    
    for i, seed in enumerate(training_seeds):
        print_progress(f"   Seed {i+1}/{len(training_seeds)}: {seed}")
        
        # Generate scenario with this seed
        scenario = generate_network_and_routes(
            grid_size=base_config['grid_size'],
            n_vehicles=base_config['n_vehicles'],
            sim_time=base_config['simulation_time'],
            pattern=base_config['traffic_pattern'],
            seed=seed,
            output_dir=os.path.join(get_project_paths()['temp'], f'seed_{seed}')
        )
        
        if scenario['success']:
            scenarios.append({
                'seed': seed,
                'files': scenario['files'],
                'weight': 1.0  # Initial equal weighting
            })
            print_progress(f"   ✅ Scenario {i+1} ready")
        else:
            print_progress(f"   ❌ Failed to generate scenario for seed {seed}")
    
    return scenarios

def cleanup_scenario_files(scenarios):
    """Clean up temporary scenario files."""
    for scenario in scenarios:
        try:
            seed_dir = os.path.dirname(scenario['files']['network'])
            if os.path.exists(seed_dir):
                shutil.rmtree(seed_dir)
        except Exception as e:
            print_progress(f"⚠️  Warning: Could not clean up scenario files: {e}")

# ============================================================================
# ROBUST EVALUATION FUNCTIONS
# ============================================================================

def evaluate_solution_multi_seed(solution, scenarios, temp_dir):
    """
    Evaluate a solution across multiple traffic seeds for robust assessment.
    
    Args:
        solution: Traffic light phase durations
        scenarios: List of scenario dictionaries
        temp_dir: Temporary directory for evaluation files
    
    Returns:
        Dictionary with aggregated metrics across all seeds
    """
    all_metrics = []
    valid_evaluations = 0
    
    for scenario in scenarios:
        seed = scenario['seed']
        net_file = scenario['files']['network']
        route_file = scenario['files']['routes']
        weight = scenario['weight']
        
        # Evaluate on this seed
        try:
            # Create temporary files for this seed evaluation
            temp_net_file = os.path.join(temp_dir, f"seed_{seed}_temp_{random.randint(1000,9999)}.net.xml")
            temp_cfg_file = temp_net_file.replace('.net.xml', '.sumocfg')
            temp_tripinfo_file = temp_net_file.replace('.net.xml', '_tripinfo.xml')
            
            # Copy and modify network file
            shutil.copy2(net_file, temp_net_file)
            apply_solution_to_network(temp_net_file, solution)
            
            # Create SUMO configuration with extended timeout for robust evaluation
            create_sumo_config(temp_cfg_file, temp_net_file, route_file, temp_tripinfo_file, None)
            
            # Run SUMO simulation
            result = subprocess.run([
                'sumo', '-c', temp_cfg_file,
                '--no-warnings', '--no-step-log',
                '--time-to-teleport', '600'  # More generous timeout for multi-seed
            ], capture_output=True, text=True, timeout=400)
            
            # Parse results
            if os.path.exists(temp_tripinfo_file):
                metrics = parse_tripinfo_file(temp_tripinfo_file)
                metrics['seed'] = seed
                metrics['weight'] = weight
                all_metrics.append(metrics)
                valid_evaluations += 1
            
            # Cleanup
            for temp_file in [temp_net_file, temp_cfg_file, temp_tripinfo_file]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
        except Exception as e:
            print_progress(f"   ⚠️  Evaluation failed for seed {seed}: {e}")
    
    # Aggregate results across seeds
    if valid_evaluations == 0:
        return {'total_time': float('inf'), 'max_stop': 0, 'vehicles': 0, 'seeds_evaluated': 0}
    
    # Calculate weighted averages
    total_weight = sum(m['weight'] for m in all_metrics)
    
    aggregated = {
        'total_time': sum(m['total_time'] * m['weight'] for m in all_metrics) / total_weight,
        'max_stop': sum(m['max_stop'] * m['weight'] for m in all_metrics) / total_weight,
        'avg_wait': sum(m.get('avg_wait', 0) * m['weight'] for m in all_metrics) / total_weight,
        'wait_p95': sum(m.get('wait_p95', 0) * m['weight'] for m in all_metrics) / total_weight,
        'vehicles': sum(m['vehicles'] * m['weight'] for m in all_metrics) / total_weight,
        'seeds_evaluated': valid_evaluations,
        'seed_details': all_metrics
    }
    
    return aggregated

def calculate_robust_cost(metrics):
    """Calculate cost from multi-seed aggregated metrics."""
    total_time = metrics.get('total_time', float('inf'))
    wait_p95 = metrics.get('wait_p95', 0)
    vehicles = metrics.get('vehicles', 1)
    seeds_evaluated = metrics.get('seeds_evaluated', 0)
    
    # Penalize solutions that fail on some seeds
    seed_penalty = max(0, len(metrics.get('seed_details', [])) - seeds_evaluated) * 100
    
    if total_time == float('inf') or vehicles == 0 or seeds_evaluated == 0:
        return float('inf')
    
    # Robust cost = average travel time + waiting penalty + seed failure penalty
    avg_time = total_time / vehicles
    wait_component = min(wait_p95, 60.0)  # Cap outliers
    waiting_penalty = 2.0  # Use same penalty weight as original
    
    return avg_time + waiting_penalty * wait_component + seed_penalty

# ============================================================================
# ADAPTIVE SEED WEIGHTING
# ============================================================================

def update_seed_weights(scenarios, solution_performance_history):
    """
    Adaptively update seed weights based on solution performance consistency.
    Seeds where solutions show high variance get higher weight in training.
    """
    if len(solution_performance_history) < 3:  # Need history to adapt
        return
    
    for scenario in scenarios:
        seed = scenario['seed']
        
        # Extract performance on this seed across iterations
        seed_performances = []
        for iteration_data in solution_performance_history[-5:]:  # Last 5 iterations
            seed_details = iteration_data.get('seed_details', [])
            for detail in seed_details:
                if detail.get('seed') == seed and detail.get('vehicles', 0) > 0:
                    cost = calculate_cost(detail)
                    if cost != float('inf'):
                        seed_performances.append(cost)
        
        # Increase weight for seeds with high variance (harder to optimize)
        if len(seed_performances) >= 3:
            variance = np.var(seed_performances)
            mean_performance = np.mean(seed_performances)
            
            # Higher variance = more challenging seed = higher weight
            if mean_performance > 0:
                cv = np.sqrt(variance) / mean_performance  # Coefficient of variation
                scenario['weight'] = 1.0 + min(cv, 2.0)  # Cap at 3x weight
            
    # Normalize weights
    total_weight = sum(s['weight'] for s in scenarios)
    if total_weight > 0:
        for scenario in scenarios:
            scenario['weight'] = scenario['weight'] / total_weight * len(scenarios)

# ============================================================================
# ROBUST ACO ALGORITHM
# ============================================================================

def initialize_robust_pheromone_matrix(n_phases, phase_types):
    """Initialize pheromone matrix for robust ACO."""
    # Same as original but with more conservative initial values
    from .simple_aco import GREEN_MIN_DURATION, GREEN_MAX_DURATION, YELLOW_MIN_DURATION, YELLOW_MAX_DURATION
    
    pheromone_matrix = {}
    
    for phase_i in range(n_phases):
        pheromone_matrix[phase_i] = {}
        
        if phase_i < len(phase_types) and phase_types[phase_i]:
            duration_range = range(GREEN_MIN_DURATION, GREEN_MAX_DURATION + 1)
        else:
            duration_range = range(YELLOW_MIN_DURATION, YELLOW_MAX_DURATION + 1)
        
        # More conservative initial pheromone for robustness
        for duration in duration_range:
            pheromone_matrix[phase_i][duration] = 0.05
    
    return pheromone_matrix

def generate_robust_ant_solution(n_phases, phase_types, pheromone_matrix, exploration_rate=0.2):
    """Generate solution with slightly higher exploration for robustness."""
    from .simple_aco import GREEN_MIN_DURATION, GREEN_MAX_DURATION, YELLOW_MIN_DURATION, YELLOW_MAX_DURATION
    
    solution = []
    
    for phase_i in range(n_phases):
        # Determine valid duration options
        if phase_i < len(phase_types) and phase_types[phase_i]:
            duration_options = list(range(GREEN_MIN_DURATION, GREEN_MAX_DURATION + 1))
        else:
            duration_options = list(range(YELLOW_MIN_DURATION, YELLOW_MAX_DURATION + 1))
        
        # Higher exploration rate for robustness
        if random.random() < exploration_rate:
            chosen_duration = random.choice(duration_options)
        else:
            # Pheromone-guided selection
            probabilities = []
            
            for duration in duration_options:
                pheromone = pheromone_matrix.get(phase_i, {}).get(duration, 0.05)
                heuristic = 1.0  # Could be enhanced with multi-seed heuristics
                
                prob = (pheromone ** 1.0) * (heuristic ** 2.0)  # Use original ALPHA/BETA
                probabilities.append(prob)
            
            # Normalize and select
            total_prob = sum(probabilities)
            if total_prob > 0:
                probabilities = [p / total_prob for p in probabilities]
                chosen_duration = np.random.choice(duration_options, p=probabilities)
            else:
                chosen_duration = random.choice(duration_options)
        
        solution.append(chosen_duration)
    
    return solution

def update_robust_pheromones(pheromone_matrix, all_solutions, all_metrics, phase_types, evaporation_rate=0.1):
    """
    Update pheromones based on robust multi-seed performance.
    Solutions that work well across ALL seeds get more reinforcement.
    """
    n_phases = len(phase_types)
    
    # 1. EVAPORATION
    for phase_i in range(n_phases):
        if phase_i in pheromone_matrix:
            for duration in list(pheromone_matrix[phase_i].keys()):
                pheromone_matrix[phase_i][duration] *= (1 - evaporation_rate)
                if pheromone_matrix[phase_i][duration] < 0.01:
                    pheromone_matrix[phase_i][duration] = 0.01
    
    # 2. REINFORCEMENT based on robust performance
    valid_data = []
    for solution, metrics in zip(all_solutions, all_metrics):
        cost = calculate_robust_cost(metrics)
        if np.isfinite(cost) and len(solution) == n_phases:
            # Weight by how many seeds were successfully evaluated
            robustness_factor = metrics.get('seeds_evaluated', 0) / max(1, len(metrics.get('seed_details', [])))
            valid_data.append((solution, cost, robustness_factor))
    
    if not valid_data:
        return
    
    # Normalize costs for pheromone calculation
    costs = [cost for _, cost, _ in valid_data]
    min_cost = min(costs)
    max_cost = max(costs)
    cost_range = max_cost - min_cost if max_cost > min_cost else 1.0
    
    # Deposit pheromones
    for solution, cost, robustness_factor in valid_data:
        # Better and more robust solutions deposit more pheromone
        if cost_range > 0:
            normalized_cost = (cost - min_cost) / cost_range
            pheromone_amount = (1.0 - normalized_cost) * robustness_factor + 0.1
        else:
            pheromone_amount = robustness_factor + 0.1
        
        for phase_i, duration in enumerate(solution):
            if phase_i >= n_phases:
                break
            
            if phase_i not in pheromone_matrix:
                pheromone_matrix[phase_i] = {}
            if duration not in pheromone_matrix[phase_i]:
                pheromone_matrix[phase_i][duration] = 0.01
            
            pheromone_matrix[phase_i][duration] += pheromone_amount
    
    # 3. ELITE REINFORCEMENT for most robust solution
    if valid_data:
        # Find solution with best cost AND high robustness
        elite_solution, _, _ = min(valid_data, key=lambda x: x[1] / max(x[2], 0.1))
        elite_boost = 1.5
        
        for phase_i, duration in enumerate(elite_solution):
            if phase_i >= n_phases:
                break
            if phase_i in pheromone_matrix and duration in pheromone_matrix[phase_i]:
                pheromone_matrix[phase_i][duration] += elite_boost

# ============================================================================
# ROBUST BASELINE EVALUATION
# ============================================================================

def evaluate_robust_baseline_comparison(best_solution, phase_types, scenarios, temp_dir):
    """
    Evaluate baseline vs optimized across all training seeds for fair comparison.
    """
    print_progress("📊 Evaluating robust baseline comparison across all seeds...")
    
    # Create baseline solution
    baseline_solution = create_baseline_solution(phase_types, green_duration=30, yellow_duration=4)
    
    # Evaluate baseline across all seeds
    print_progress("   Evaluating baseline (30s green, 4s yellow) across all seeds...")
    baseline_metrics = evaluate_solution_multi_seed(baseline_solution, scenarios, temp_dir)
    baseline_cost = calculate_robust_cost(baseline_metrics)
    
    # Evaluate optimized solution across all seeds
    print_progress("   Evaluating optimized solution across all seeds...")
    optimized_metrics = evaluate_solution_multi_seed(best_solution, scenarios, temp_dir)
    optimized_cost = calculate_robust_cost(optimized_metrics)
    
    # Calculate improvement
    if baseline_cost != float('inf') and optimized_cost != float('inf'):
        improvement_percent = ((baseline_cost - optimized_cost) / baseline_cost) * 100
        absolute_improvement = baseline_cost - optimized_cost
    else:
        improvement_percent = 0
        absolute_improvement = 0
    
    comparison_results = {
        'baseline': {
            'solution': baseline_solution,
            'cost': baseline_cost,
            'metrics': baseline_metrics
        },
        'optimized': {
            'solution': best_solution,
            'cost': optimized_cost,
            'metrics': optimized_metrics
        },
        'improvement': {
            'percent': improvement_percent,
            'absolute': absolute_improvement
        },
        'robustness': {
            'baseline_seeds_evaluated': baseline_metrics.get('seeds_evaluated', 0),
            'optimized_seeds_evaluated': optimized_metrics.get('seeds_evaluated', 0),
            'total_seeds': len(scenarios)
        }
    }
    
    # Print comparison results
    baseline_seeds = baseline_metrics.get('seeds_evaluated', 0)
    optimized_seeds = optimized_metrics.get('seeds_evaluated', 0)
    total_seeds = len(scenarios)
    
    print_progress(f"📊 ROBUST BASELINE COMPARISON RESULTS:")
    print_progress(f"   Baseline (30s/4s): Cost = {baseline_cost:.1f} (evaluated on {baseline_seeds}/{total_seeds} seeds)")
    print_progress(f"   Optimized solution: Cost = {optimized_cost:.1f} (evaluated on {optimized_seeds}/{total_seeds} seeds)")
    
    if improvement_percent > 0:
        print_progress(f"   ✅ Robust Improvement: {improvement_percent:.1f}% better ({absolute_improvement:.1f} cost units)")
    elif improvement_percent < 0:
        print_progress(f"   ❌ Degradation: {abs(improvement_percent):.1f}% worse ({abs(absolute_improvement):.1f} cost units)")
    else:
        print_progress(f"   ➖ No significant difference")
    
    return comparison_results

# ============================================================================
# MAIN ROBUST ACO ALGORITHM
# ============================================================================

def run_robust_aco_optimization(
    config=None, 
    training_seeds=None,
    show_plots_override=None, 
    show_gui_override=None, 
    compare_baseline=True,
    sumo_config_file=None
):
    """
    Run robust ACO optimization across multiple traffic seeds.
    
    Args:
        config: Configuration dictionary (same as original ACO)
        training_seeds: List of seeds for training (if None, generates random seeds)
        show_plots_override: Control plot display
        show_gui_override: Control GUI launch
        compare_baseline: Whether to compare against baseline
        sumo_config_file: Base SUMO config file for scenario template
    
    Returns:
        Dictionary with optimization results including robustness metrics
    """
    print("🌱 ROBUST MULTI-SEED ACO OPTIMIZATION")
    print("=" * 60)
    
    # Setup configuration
    from .simple_aco import GRID_SIZE, N_VEHICLES, SIMULATION_TIME, N_ANTS, N_ITERATIONS
    from .simple_aco import EVAPORATION_RATE, EXPLORATION_RATE, SHOW_PLOTS, LAUNCH_SUMO_GUI
    
    # Apply config overrides (same as original)
    if config:
        GRID_SIZE = config.get('grid_size', GRID_SIZE)
        N_VEHICLES = config.get('n_vehicles', N_VEHICLES) 
        SIMULATION_TIME = config.get('simulation_time', SIMULATION_TIME)
        N_ANTS = config.get('n_ants', N_ANTS)
        N_ITERATIONS = config.get('n_iterations', N_ITERATIONS)
        EVAPORATION_RATE = config.get('evaporation_rate', EVAPORATION_RATE)
        EXPLORATION_RATE = config.get('exploration_rate', EXPLORATION_RATE)
    
    # Robust-specific config
    n_training_seeds = config.get('training_seeds', DEFAULT_TRAINING_SEEDS) if config else DEFAULT_TRAINING_SEEDS
    
    # Generate training seeds if not provided
    if training_seeds is None:
        base_seed = config.get('seed', 42) if config else 42
        training_seeds = [base_seed + i * 17 for i in range(n_training_seeds)]  # Use prime offset
    
    print_progress(f"📋 Robust Configuration:")
    print_progress(f"   Grid: {GRID_SIZE}x{GRID_SIZE}, Vehicles: {N_VEHICLES}, Time: {SIMULATION_TIME}s")
    print_progress(f"   ACO: {N_ANTS} ants × {N_ITERATIONS} iterations")
    print_progress(f"   Training Seeds: {len(training_seeds)} ({training_seeds})")
    print_progress(f"   Exploration Rate: {EXPLORATION_RATE:.2f} (increased for robustness)")
    
    # Extract base scenario config from SUMO file if provided
    base_config = {
        'grid_size': GRID_SIZE,
        'n_vehicles': N_VEHICLES,
        'simulation_time': SIMULATION_TIME,
        'traffic_pattern': config.get('traffic_pattern', 'commuter') if config else 'commuter'
    }
    
    paths = get_project_paths()
    
    try:
        # Generate multiple scenarios with different seeds
        scenarios = generate_multi_seed_scenarios(base_config, training_seeds)
        
        if not scenarios:
            print_progress("❌ Failed to generate any training scenarios")
            return {'success': False, 'error': 'No valid scenarios generated'}
        
        print_progress(f"✅ Generated {len(scenarios)} training scenarios")
        
        # Use first scenario to analyze traffic light structure
        sample_net_file = scenarios[0]['files']['network']
        phase_types, default_durations = analyze_traffic_light_phases(sample_net_file)
        n_phases = len(phase_types)
        
        # Initialize robust pheromone matrix
        pheromone_matrix = initialize_robust_pheromone_matrix(n_phases, phase_types)
        
        # Track optimization progress
        best_costs = []
        best_solutions = []
        best_metrics_history = []
        solution_performance_history = []
        
        global_best_cost = float('inf')
        global_best_solution = None
        global_best_metrics = None
        
        print_progress("🔄 Starting robust optimization iterations...")
        start_time = time.time()
        
        # Main robust ACO loop
        for iteration in range(N_ITERATIONS):
            print_progress(f"Iteration {iteration + 1}/{N_ITERATIONS}")
            
            solutions = []
            metrics_list = []
            
            # Include elite solution for stability
            if global_best_solution is not None:
                solutions.append(global_best_solution.copy())
                metrics_list.append(global_best_metrics)
                print_progress(f"   Elite solution injected: robust cost {global_best_cost:.1f}")
            
            # Generate ant solutions
            remaining_ants = N_ANTS - (1 if global_best_solution is not None else 0)
            for ant in range(remaining_ants):
                solution = generate_robust_ant_solution(n_phases, phase_types, pheromone_matrix, EXPLORATION_RATE)
                
                # Evaluate across all training seeds
                metrics = evaluate_solution_multi_seed(solution, scenarios, paths['temp'])
                cost = calculate_robust_cost(metrics)
                
                solutions.append(solution)
                metrics_list.append(metrics)
                
                # Update global best
                if cost < global_best_cost:
                    global_best_cost = cost
                    global_best_solution = solution.copy()
                    global_best_metrics = metrics
                    print_progress(f"   🌟 NEW ROBUST BEST: Ant {ant+1}, cost: {cost:.1f}")
                
                # Progress reporting
                avg_vehicles = metrics.get('vehicles', 0)
                seeds_evaluated = metrics.get('seeds_evaluated', 0)
                total_seeds = len(scenarios)
                
                if avg_vehicles > 0:
                    avg_time = metrics['total_time'] / avg_vehicles
                    print_progress(f"   Ant {ant+1}: {avg_vehicles:.1f}/{N_VEHICLES} avg vehicles, "
                                 f"{seeds_evaluated}/{total_seeds} seeds, avg time: {avg_time:.1f}s, cost: {cost:.1f}")
                else:
                    print_progress(f"   Ant {ant+1}: 0/{N_VEHICLES} vehicles, cost: ∞")
            
            # Update pheromones based on multi-seed performance
            update_robust_pheromones(pheromone_matrix, solutions, metrics_list, phase_types, EVAPORATION_RATE)
            
            # Adaptive seed weighting (learn which seeds are harder)
            # Flatten the seed details from all solutions this iteration
            all_seed_details = []
            for m in metrics_list:
                if isinstance(m, dict) and 'seed_details' in m:
                    all_seed_details.extend(m['seed_details'])
            
            solution_performance_history.append({
                'iteration': iteration,
                'solutions': solutions,
                'seed_details': all_seed_details  # Flattened list of seed detail dicts
            })
            
            if iteration >= 2:  # Start adapting after a few iterations
                update_seed_weights(scenarios, solution_performance_history)
            
            # Track progress
            best_costs.append(global_best_cost)
            
            # Ensure global_best_metrics is a dict, not list, and handle None case
            if global_best_metrics:
                if isinstance(global_best_metrics, dict):
                    best_metrics_history.append(global_best_metrics)
                else:
                    print_progress(f"⚠️  Unexpected metrics type: {type(global_best_metrics)}")
                    best_metrics_history.append({'total_time': 0, 'max_stop': 0, 'vehicles': 0})
            else:
                best_metrics_history.append({'total_time': 0, 'max_stop': 0, 'vehicles': 0})
        
        duration = time.time() - start_time
        print_progress(f"✅ Robust optimization completed in {duration:.1f} seconds")
        
        # Robust baseline comparison
        baseline_comparison = None
        if compare_baseline and global_best_solution is not None:
            baseline_comparison = evaluate_robust_baseline_comparison(
                global_best_solution, phase_types, scenarios, paths['temp']
            )
        
        # Create plots
        if show_plots_override is not None:
            show_plot = show_plots_override
        else:
            show_plot = True  # Default to showing plots
            
        if show_plot and len(best_costs) > 0:
            create_robust_optimization_plot(best_costs, best_metrics_history, scenarios, paths, baseline_comparison)
        
        # Launch GUI if requested
        if show_gui_override is not None:
            launch_gui = show_gui_override
        else:
            launch_gui = False  # Default to not launching GUI
            
        if launch_gui and global_best_solution is not None:
            from .simple_aco import launch_sumo_gui_with_solution
            sample_net = scenarios[0]['files']['network']
            sample_route = scenarios[0]['files']['routes']
            launch_sumo_gui_with_solution(global_best_solution, sample_net, sample_route, paths)
        
        # Cleanup temporary files
        cleanup_scenario_files(scenarios)
        
        # Return results (compatible with original interface)
        return {
            'success': True,
            'best_cost': global_best_cost,
            'best_solution': global_best_solution,
            'cost_history': best_costs,
            'metrics_history': best_metrics_history,
            'phase_types': phase_types,
            'n_phases': n_phases,
            'duration': duration,
            'baseline_comparison': baseline_comparison,
            'robustness': {
                'training_seeds': training_seeds,
                'scenarios_used': len(scenarios),
                'final_seed_weights': [s['weight'] for s in scenarios]
            }
        }
        
    except Exception as e:
        print_progress(f"❌ Robust optimization failed: {e}")
        print_progress(f"   Error type: {type(e)}")
        import traceback
        print_progress(f"   Traceback: {traceback.format_exc()}")
        
        if 'scenarios' in locals():
            cleanup_scenario_files(scenarios)
        return {'success': False, 'error': str(e)}

# ============================================================================
# ROBUST VISUALIZATION
# ============================================================================

def create_robust_optimization_plot(best_costs, best_metrics_history, scenarios, paths, baseline_comparison=None):
    """Create visualization showing robust optimization progress and seed performance."""
    
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Cost progression with robustness indicators
    plt.subplot(2, 2, 1)
    iterations = range(len(best_costs))
    plt.plot(iterations, best_costs, 'b-o', linewidth=2, markersize=6, label='Best Robust Cost')
    
    # Add baseline if available
    if baseline_comparison and isinstance(baseline_comparison, dict) and 'baseline' in baseline_comparison:
        baseline_cost = baseline_comparison['baseline'].get('cost', None)
        if baseline_cost is not None and baseline_cost != float('inf'):
            plt.axhline(y=baseline_cost, color='r', linestyle='--', linewidth=2, 
                       label=f'Baseline Cost ({baseline_cost:.1f})', alpha=0.8)
    
    plt.xlabel('Iteration')
    plt.ylabel('Robust Cost')
    plt.title('Robust ACO Progress')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(range(len(best_costs)))
    
    # Plot 2: Vehicle completion across seeds
    plt.subplot(2, 2, 2)
    if best_metrics_history:
        vehicle_completion = []
        for m in best_metrics_history:
            if isinstance(m, dict):
                vehicle_completion.append(m.get('vehicles', 0))
            else:
                vehicle_completion.append(0)
        
        plt.plot(iterations, vehicle_completion, 'g-s', linewidth=2, markersize=6, label='Avg Vehicles Completed')
        plt.axhline(y=20, color='r', linestyle=':', alpha=0.7, label='Target (20 vehicles)')
        plt.xlabel('Iteration')
        plt.ylabel('Vehicles Completed')
        plt.title('Vehicle Completion Robustness')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(range(len(best_costs)))
    
    # Plot 3: Seed performance distribution
    plt.subplot(2, 2, 3)
    if scenarios and len(scenarios) > 1:
        seed_weights = [s['weight'] for s in scenarios]
        seed_labels = [f"Seed {s['seed']}" for s in scenarios]
        plt.bar(range(len(seed_weights)), seed_weights)
        plt.xlabel('Training Seed')
        plt.ylabel('Final Weight')
        plt.title('Adaptive Seed Weights\n(Higher = More Challenging)')
        plt.xticks(range(len(seed_labels)), [f"S{s['seed']}" for s in scenarios], rotation=45)
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Robustness metrics
    plt.subplot(2, 2, 4)
    if best_metrics_history:
        seeds_evaluated = [m.get('seeds_evaluated', 0) for m in best_metrics_history]
        plt.plot(iterations, seeds_evaluated, 'purple', linewidth=2, marker='d', markersize=6, label='Seeds Successfully Evaluated')
        plt.axhline(y=len(scenarios), color='r', linestyle=':', alpha=0.7, label=f'Target ({len(scenarios)} seeds)')
        plt.xlabel('Iteration')
        plt.ylabel('Seeds Evaluated')
        plt.title('Training Robustness')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(range(len(best_costs)))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(paths['results'], 'robust_aco_optimization_progress.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print_progress(f"📊 Robust optimization plot saved to: {plot_path}")
    
    # Just show the plot by default - the caller controls whether plots are shown
    plt.show()

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_robust_solution(solution, phase_types, base_config, validation_seeds=None, temp_dir=None):
    """
    Validate the robust solution on completely new seeds not used in training.
    
    Args:
        solution: Optimized traffic light solution
        phase_types: Phase type information
        base_config: Base scenario configuration
        validation_seeds: Seeds for validation (if None, generates new ones)
        temp_dir: Temporary directory
    
    Returns:
        Validation results
    """
    if validation_seeds is None:
        # Generate fresh validation seeds (different from training)
        validation_seeds = [1000 + i * 23 for i in range(DEFAULT_VALIDATION_SEEDS)]
    
    if temp_dir is None:
        temp_dir = get_project_paths()['temp']
    
    print_progress(f"🔍 Validating solution on {len(validation_seeds)} new seeds...")
    
    # Generate validation scenarios
    validation_scenarios = generate_multi_seed_scenarios(base_config, validation_seeds)
    
    if not validation_scenarios:
        return {'success': False, 'error': 'Failed to generate validation scenarios'}
    
    try:
        # Evaluate solution on validation seeds
        validation_metrics = evaluate_solution_multi_seed(solution, validation_scenarios, temp_dir)
        validation_cost = calculate_robust_cost(validation_metrics)
        
        # Evaluate baseline on same validation seeds
        baseline_solution = create_baseline_solution(phase_types, 30, 4)
        baseline_val_metrics = evaluate_solution_multi_seed(baseline_solution, validation_scenarios, temp_dir)
        baseline_val_cost = calculate_robust_cost(baseline_val_metrics)
        
        # Calculate validation improvement
        if baseline_val_cost != float('inf') and validation_cost != float('inf'):
            val_improvement = ((baseline_val_cost - validation_cost) / baseline_val_cost) * 100
        else:
            val_improvement = 0
        
        # Cleanup
        cleanup_scenario_files(validation_scenarios)
        
        results = {
            'success': True,
            'validation_cost': validation_cost,
            'baseline_validation_cost': baseline_val_cost,
            'improvement_percent': val_improvement,
            'seeds_evaluated': validation_metrics.get('seeds_evaluated', 0),
            'total_validation_seeds': len(validation_seeds),
            'avg_vehicles_completed': validation_metrics.get('vehicles', 0)
        }
        
        print_progress(f"📊 VALIDATION RESULTS:")
        print_progress(f"   Validation Cost: {validation_cost:.1f}")
        print_progress(f"   Baseline Validation: {baseline_val_cost:.1f}")
        print_progress(f"   Validation Improvement: {val_improvement:.1f}%")
        print_progress(f"   Seeds Evaluated: {results['seeds_evaluated']}/{len(validation_seeds)}")
        
        return results
        
    except Exception as e:
        print_progress(f"❌ Validation failed: {e}")
        if 'validation_scenarios' in locals():
            cleanup_scenario_files(validation_scenarios)
        return {'success': False, 'error': str(e)}

# ============================================================================
# CONVENIENCE WRAPPER
# ============================================================================

class RobustACOTrafficOptimizer:
    """
    Robust version of ACOTrafficOptimizer that trains on multiple seeds.
    Drop-in replacement for the original with additional robustness features.
    """
    
    def __init__(
        self,
        sumo_config: Optional[str] = None,
        n_ants: int = 20,
        n_iterations: int = 10,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        verbose: bool = True,
        scenario_vehicles: int = None,
        simulation_time: int = None,
        show_plots: bool = True,
        show_sumo_gui: bool = False,
        compare_baseline: bool = True,
        # Robust-specific parameters
        training_seeds: int = 5,  # Number of training seeds
        exploration_rate: float = 0.25,  # Higher exploration for robustness
        validate_solution: bool = True  # Run validation on new seeds
    ) -> None:
        self.sumo_config = sumo_config
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.verbose = verbose
        self.scenario_vehicles = scenario_vehicles
        self.simulation_time = simulation_time
        self.show_plots = show_plots
        self.show_sumo_gui = show_sumo_gui
        self.compare_baseline = compare_baseline
        self.training_seeds = training_seeds
        self.exploration_rate = exploration_rate
        self.validate_solution = validate_solution
    
    def optimize(self) -> Tuple[Dict[str, Dict[str, int]], float, List[Dict[str, float]], Optional[Dict]]:
        """
        Run robust optimization across multiple seeds.
        
        Returns same format as original ACOTrafficOptimizer for compatibility.
        """
        # Build config
        config = {
            'n_ants': self.n_ants,
            'n_iterations': self.n_iterations,
            'stop_penalty': self.alpha,
            'evaporation_rate': max(0.01, min(0.9, self.rho)),
            'exploration_rate': self.exploration_rate,
            'n_vehicles': self.scenario_vehicles or 30,
            'simulation_time': self.simulation_time,
            'training_seeds': self.training_seeds
        }
        
        # Run robust optimization
        results = run_robust_aco_optimization(
            config=config,
            show_plots_override=self.show_plots,
            show_gui_override=self.show_sumo_gui,
            compare_baseline=self.compare_baseline,
            sumo_config_file=self.sumo_config
        )
        
        if not results.get('success'):
            raise RuntimeError(results.get('error', 'Robust optimization failed'))
        
        # Convert to original format
        durations = results.get('best_solution') or []
        phase_types = results.get('phase_types') or [True] * len(durations)
        best_cost = float(results.get('best_cost', float('inf')))
        
        solution_dict = {}
        for idx, dur in enumerate(durations):
            key = f'phase_{idx}'
            if idx < len(phase_types) and not phase_types[idx]:
                solution_dict[key] = {'yellow': int(dur)}
            else:
                solution_dict[key] = {'green': int(dur)}
        
        cost_history = results.get('cost_history') or []
        optimization_data = [{'best_cost': float(c)} for c in cost_history]
        
        baseline_comparison = results.get('baseline_comparison')
        
        # Add robustness info to baseline comparison
        if baseline_comparison and isinstance(baseline_comparison, dict):
            baseline_comparison['robustness_info'] = {
                'training_seeds': results.get('robustness', {}).get('training_seeds', []),
                'multi_seed_training': True,
                'exploration_rate': self.exploration_rate
            }
        
        # Optional validation on completely new seeds
        if self.validate_solution and durations:
            print_progress("\n🔍 VALIDATION ON UNSEEN SEEDS")
            print_progress("=" * 40)
            
            # Extract config for validation
            if self.sumo_config:
                base_config = {
                    'grid_size': config.get('grid_size', 5),
                    'n_vehicles': config.get('n_vehicles', 20),
                    'simulation_time': config.get('simulation_time', 2400),
                    'traffic_pattern': 'commuter'  # Could extract from config
                }
                
                validation_results = validate_robust_solution(
                    durations, phase_types, base_config, 
                    validation_seeds=None,  # Generate fresh seeds
                    temp_dir=get_project_paths()['temp']
                )
                
                if validation_results.get('success'):
                    # Add validation info to baseline comparison
                    if baseline_comparison and isinstance(baseline_comparison, dict):
                        baseline_comparison['validation'] = validation_results
        
        return solution_dict, best_cost, optimization_data, baseline_comparison

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example of how to use the robust ACO
    print("🧪 TESTING ROBUST ACO")
    
    # Test configuration
    config = {
        'grid_size': 3,
        'n_vehicles': 20, 
        'simulation_time': 1800,
        'n_ants': 10,
        'n_iterations': 5,
        'training_seeds': 3  # Train on 3 different seeds
    }
    
    results = run_robust_aco_optimization(config=config, compare_baseline=True)
    
    if results['success']:
        print(f"\n🎉 Robust optimization completed!")
        print(f"📊 Best robust cost: {results['best_cost']:.1f}")
        print(f"🌱 Trained on {len(results['robustness']['training_seeds'])} seeds")
    else:
        print(f"\n❌ Robust optimization failed: {results.get('error')}")
