"""
Simplified Ant Colony Optimization for Traffic Light Timing

This is a clean, simplified implementation that eliminates the complex "bins" system
and uses direct range sampling for traffic light phase durations.

Key improvements:
- No complex bins arrays or mapping logic
- Direct sampling from traffic engineering ranges
- Simple pheromone reinforcement per phase
- Much easier to understand and debug
- Fixes iteration stability issues

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
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Traffic Engineering Constraints
GREEN_MIN_DURATION = 20        # Minimum green/red phase (includes SUMO defaults)
GREEN_MAX_DURATION = 100       # Maximum green/red phase (includes SUMO defaults)
YELLOW_MIN_DURATION = 3        # Minimum yellow phase (safety standard)
YELLOW_MAX_DURATION = 6        # Maximum yellow phase (safety standard)

# ACO Algorithm Parameters - Improved for stability
N_ANTS = 15                    # Reduced from 20 for better focus
N_ITERATIONS = 12              # Increased for more exploration 
EVAPORATION_RATE = 0.05        # Reduced from 0.1 to preserve good solutions longer
EXPLORATION_RATE = 0.20        # Increased from 0.15 for better exploration
ALPHA = 25.0                   # Reduced penalty weight for more balanced optimization

# Scenario Configuration
GRID_SIZE = 3                  # Grid dimensions (2 = 2x2, 3 = 3x3, etc.)
N_VEHICLES = 30               # Number of vehicles in simulation
SIMULATION_TIME = 1200        # Increased from 600 to ensure all vehicles complete trips

# Display and Output
SHOW_PROGRESS = True          # Show detailed progress
SHOW_PLOTS = True            # Show optimization plots
LAUNCH_SUMO_GUI = False      # Launch SUMO GUI with results
SAVE_RESULTS = True          # Save results to files

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_progress(message, show_time=True):
    """Print progress message with optional timestamp."""
    if SHOW_PROGRESS:
        timestamp = f"[{datetime.now().strftime('%H:%M:%S')}] " if show_time else ""
        print(f"{timestamp}{message}")

def get_project_paths():
    """Get standardized project file paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels
    
    paths = {
        'project_root': project_root,
        'sumo_data': os.path.join(project_root, 'sumo_data'),
        'results': os.path.join(project_root, 'results'),
        'temp': os.path.join(script_dir, 'temp_files'),
    }
    
    # Ensure directories exist
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths

# ============================================================================
# PLOTTING AND VISUALIZATION
# ============================================================================

def create_optimization_plot(best_costs, best_metrics_history, paths, show_plot=None):
    """Create and save optimization progress plot."""
    # Use parameter if provided, otherwise use global setting
    should_show = show_plot if show_plot is not None else SHOW_PLOTS
    
    if not should_show and not True:  # Always save plots even if not showing
        return
        
    plt.figure(figsize=(14, 10))
    iterations = range(len(best_costs))
    
    # Extract time and stop metrics from history
    best_times = [metrics.get('total_time', 0) for metrics in best_metrics_history]
    best_maxstops = [metrics.get('max_stop', 0) for metrics in best_metrics_history]
    
    # Plot 1: Cost progression
    plt.subplot(3, 1, 1)
    plt.plot(iterations, best_costs, 'b-o', linewidth=2, markersize=6, label='Best Cost per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('ACO Traffic Light Optimization Progress - Cost Function')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add improvement annotation
    if len(best_costs) > 1:
        initial_cost = best_costs[0]
        final_cost = best_costs[-1]
        improvement = ((initial_cost - final_cost) / initial_cost) * 100
        plt.text(0.7, 0.9, f'Optimization Improvement: {improvement:.1f}%', 
                transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    # Plot 2: Travel time progression
    plt.subplot(3, 1, 2)
    plt.plot(iterations, best_times, 'g-x', linewidth=2, markersize=6, label='Best Total Travel Time')
    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('Total Travel Time Progress')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Max stop time progression
    plt.subplot(3, 1, 3)
    plt.plot(iterations, best_maxstops, 'r-s', linewidth=2, markersize=6, label='Best Max Stop Time')
    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('Maximum Stop Time Progress')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(paths['results'], 'aco_optimization_progress.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print_progress(f"📊 Optimization plot saved to: {plot_path}")
    
    if should_show:
        plt.show()
    else:
        plt.close()  # Close plot to free memory when not showing

# ============================================================================
# TRAFFIC LIGHT PHASE ANALYSIS
# ============================================================================

def analyze_traffic_light_phases(net_file):
    """
    Analyze traffic light phases to determine which are green/red vs yellow.
    Returns phase types and count information.
    """
    try:
        tree = ET.parse(net_file)
        root = tree.getroot()
        
        phase_types = []  # True = green/red, False = yellow
        phase_durations = []
        
        for tl_logic in root.findall('tlLogic'):
            for phase in tl_logic.findall('phase'):
                duration = int(float(phase.get('duration', '30')))
                state = phase.get('state', '')
                
                # Classify phase based on state and duration
                # Yellow phases typically have 'y' in state and short duration
                is_yellow = ('y' in state.lower() or duration <= 10)
                
                phase_types.append(not is_yellow)  # True = green/red, False = yellow
                phase_durations.append(duration)
        
        green_count = sum(1 for is_green in phase_types if is_green)
        yellow_count = len(phase_types) - green_count
        
        print_progress(f"🚦 Found {len(phase_types)} phases: {green_count} green/red, {yellow_count} yellow")
        
        return phase_types, phase_durations
        
    except Exception as e:
        print_progress(f"⚠️  Error analyzing phases: {e}")
        # Fallback: assume alternating green/yellow
        n_phases = 8  # Default assumption
        phase_types = [i % 2 == 0 for i in range(n_phases)]  # Alternating
        return phase_types, [30] * n_phases

# ============================================================================
# SIMPLIFIED ACO ALGORITHM
# ============================================================================

def generate_ant_solution(n_phases, phase_types, pheromone_weights=None):
    """
    Generate a traffic light solution using simplified direct sampling.
    
    Args:
        n_phases: Number of traffic light phases
        phase_types: List of booleans (True=green/red, False=yellow)
        pheromone_weights: Optional pheromone influence per phase
    
    Returns:
        List of phase durations
    """
    solution = []
    
    for i in range(n_phases):
        # Determine appropriate duration range
        if i < len(phase_types) and phase_types[i]:
            # Green/red phase - wider range including SUMO defaults
            min_dur, max_dur = GREEN_MIN_DURATION, GREEN_MAX_DURATION
        else:
            # Yellow phase - narrow safety range
            min_dur, max_dur = YELLOW_MIN_DURATION, YELLOW_MAX_DURATION
        
        # Random exploration vs pheromone guidance with improved logic
        if random.random() < EXPLORATION_RATE:
            # Pure exploration - random duration
            duration = random.randint(min_dur, max_dur)
        else:
            # Pheromone-guided selection with improved bias calculation
            if pheromone_weights is not None and i < len(pheromone_weights):
                influence = pheromone_weights[i]
                
                # More sophisticated pheromone influence with safe range checking
                if influence > 2.0:
                    # Very high pheromone - strong bias toward proven good values
                    range_size = max_dur - min_dur
                    if range_size <= 1:
                        duration = min_dur  # Handle edge case where range is too small
                    else:
                        center = min_dur + int(range_size * 0.6)  # Bias toward upper-middle
                        spread = max(1, range_size // 6)  
                        biased_min = max(min_dur, center - spread)
                        biased_max = min(max_dur, center + spread)
                        # Ensure valid range
                        if biased_max <= biased_min:
                            biased_max = biased_min + 1
                        duration = random.randint(biased_min, min(biased_max, max_dur))
                elif influence > 1.0:
                    # Medium pheromone - moderate bias
                    range_size = max_dur - min_dur
                    if range_size <= 1:
                        duration = min_dur
                    else:
                        center = min_dur + int(range_size * 0.5)  # Center of range
                        spread = max(1, range_size // 3)
                        biased_min = max(min_dur, center - spread)
                        biased_max = min(max_dur, center + spread)
                        # Ensure valid range
                        if biased_max <= biased_min:
                            biased_max = biased_min + 1
                        duration = random.randint(biased_min, min(biased_max, max_dur))
                else:
                    # Low pheromone - use full range but avoid extremes
                    range_size = max_dur - min_dur
                    if range_size <= 2:
                        duration = random.randint(min_dur, max_dur)
                    else:
                        margin = max(1, range_size // 8)
                        safe_min = min_dur + margin
                        safe_max = max_dur - margin
                        if safe_max <= safe_min:
                            duration = random.randint(min_dur, max_dur)
                        else:
                            duration = random.randint(safe_min, safe_max)
            else:
                # No pheromone info - use full range
                duration = random.randint(min_dur, max_dur)
        
        solution.append(duration)
    
    return solution

def update_pheromones(pheromone_weights, solutions, scores):
    """
    Update pheromone weights based on ant performance with improved stability.
    
    Args:
        pheromone_weights: Current pheromone weights per phase
        solutions: List of ant solutions
        scores: List of corresponding performance scores (lower is better)
    
    Returns:
        Updated pheromone weights
    """
    if not solutions or not scores:
        return pheromone_weights
    
    # Evaporate existing pheromones more gradually
    pheromone_weights *= (1 - EVAPORATION_RATE)
    
    # Filter valid solutions
    valid_pairs = [(sol, score) for sol, score in zip(solutions, scores) if score < float('inf')]
    if not valid_pairs:
        return pheromone_weights
    
    valid_solutions, valid_scores = zip(*valid_pairs)
    
    # Improved reward calculation with rank-based selection
    sorted_pairs = sorted(valid_pairs, key=lambda x: x[1])
    n_valid = len(sorted_pairs)
    
    # Give rewards based on ranking instead of just best/worst comparison
    for rank, (solution, score) in enumerate(sorted_pairs):
        # Top performers get exponentially higher rewards
        if rank < n_valid // 3:  # Top 1/3
            reward = 0.8 * (1 - rank / max(1, n_valid // 3))
        elif rank < 2 * n_valid // 3:  # Middle 1/3  
            reward = 0.3 * (1 - (rank - n_valid // 3) / max(1, n_valid // 3))
        else:  # Bottom 1/3
            reward = 0.1
        
        # Apply reward to each phase with smoother updates
        for i, duration in enumerate(solution):
            if i < len(pheromone_weights):
                pheromone_weights[i] += reward * 0.5  # Gentler updates
    
    # Ensure pheromone bounds with wider range
    pheromone_weights = np.clip(pheromone_weights, 0.2, 3.0)
    
    return pheromone_weights

# ============================================================================
# SUMO SIMULATION AND EVALUATION
# ============================================================================

def evaluate_solution(solution, net_file, route_file, temp_dir):
    """
    Evaluate a traffic light solution using SUMO simulation.
    
    Args:
        solution: List of phase durations
        net_file: SUMO network file
        route_file: SUMO route file
        temp_dir: Temporary directory for simulation files
    
    Returns:
        Dictionary with performance metrics
    """
    try:
        # Create temporary files for this evaluation
        temp_net_file = os.path.join(temp_dir, f"temp_{random.randint(1000,9999)}.net.xml")
        temp_cfg_file = temp_net_file.replace('.net.xml', '.sumocfg')
        temp_tripinfo_file = temp_net_file.replace('.net.xml', '_tripinfo.xml')
        
        # Copy and modify network file with new traffic light timings
        shutil.copy2(net_file, temp_net_file)
        apply_solution_to_network(temp_net_file, solution)
        
        # Create SUMO configuration
        create_sumo_config(temp_cfg_file, temp_net_file, route_file, temp_tripinfo_file)
        
        # Run SUMO simulation
        result = subprocess.run([
            'sumo', '-c', temp_cfg_file,
            '--no-warnings', '--no-step-log',
            '--time-to-teleport', '300'  # Allow teleport if stuck
        ], capture_output=True, text=True, timeout=60)
        
        # Parse results
        if os.path.exists(temp_tripinfo_file):
            metrics = parse_tripinfo_file(temp_tripinfo_file)
        else:
            metrics = {'total_time': float('inf'), 'max_stop': 0, 'vehicles': 0}
        
        # Cleanup temporary files
        for temp_file in [temp_net_file, temp_cfg_file, temp_tripinfo_file]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return metrics
        
    except Exception as e:
        print_progress(f"   ❌ Evaluation error: {e}")
        return {'total_time': float('inf'), 'max_stop': 0, 'vehicles': 0}

def apply_solution_to_network(net_file, solution):
    """Apply traffic light solution to network file."""
    try:
        tree = ET.parse(net_file)
        root = tree.getroot()
        
        phase_idx = 0
        for tl_logic in root.findall('tlLogic'):
            for phase in tl_logic.findall('phase'):
                if phase_idx < len(solution):
                    phase.set('duration', str(solution[phase_idx]))
                    phase_idx += 1
        
        tree.write(net_file, xml_declaration=True, encoding='UTF-8')
        
    except Exception as e:
        print_progress(f"   ⚠️  Error applying solution: {e}")

def create_sumo_config(cfg_file, net_file, route_file, tripinfo_file):
    """Create SUMO configuration file."""
    config_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{os.path.basename(net_file)}"/>
        <route-files value="{route_file}"/>
    </input>
    <output>
        <tripinfo-output value="{os.path.basename(tripinfo_file)}"/>
    </output>
    <time>
        <end value="{SIMULATION_TIME}"/>
    </time>
</configuration>'''
    
    with open(cfg_file, 'w') as f:
        f.write(config_content)

def parse_tripinfo_file(tripinfo_file):
    """Parse SUMO tripinfo output to extract performance metrics."""
    try:
        tree = ET.parse(tripinfo_file)
        root = tree.getroot()
        
        total_time = 0
        max_stop = 0
        vehicle_count = 0
        
        for trip in root.findall('tripinfo'):
            duration = float(trip.get('duration', '0'))
            waiting_time = float(trip.get('waitingTime', '0'))
            
            total_time += duration
            max_stop = max(max_stop, waiting_time)
            vehicle_count += 1
        
        return {
            'total_time': total_time,
            'max_stop': max_stop,
            'vehicles': vehicle_count
        }
        
    except Exception as e:
        print_progress(f"   ⚠️  Error parsing tripinfo: {e}")
        return {'total_time': float('inf'), 'max_stop': 0, 'vehicles': 0}

def calculate_cost(metrics):
    """Calculate cost function from simulation metrics."""
    total_time = metrics.get('total_time', float('inf'))
    max_stop = metrics.get('max_stop', 0)
    vehicles = metrics.get('vehicles', 1)
    
    if total_time == float('inf') or vehicles == 0:
        return float('inf')
    
    # Cost = average travel time + penalty for long stops
    avg_time = total_time / vehicles
    stop_penalty = ALPHA * max_stop
    
    return avg_time + stop_penalty

# ============================================================================
# MAIN OPTIMIZATION FUNCTION
# ============================================================================

def run_simplified_aco_optimization(config=None, show_plots_override=None):
    """
    Run the simplified ACO optimization.
    
    Args:
        config: Optional configuration dictionary
        show_plots_override: Boolean to control plot display (overrides global SHOW_PLOTS)
    
    Returns:
        Dictionary with optimization results
    """
    print("🐜 SIMPLIFIED ANT COLONY OPTIMIZATION")
    print("=" * 50)
    
    # Apply configuration if provided
    if config:
        global GRID_SIZE, N_VEHICLES, SIMULATION_TIME, N_ANTS, N_ITERATIONS
        GRID_SIZE = config.get('grid_size', GRID_SIZE)
        N_VEHICLES = config.get('n_vehicles', N_VEHICLES)
        SIMULATION_TIME = config.get('simulation_time', SIMULATION_TIME)
        N_ANTS = config.get('n_ants', N_ANTS)
        N_ITERATIONS = config.get('n_iterations', N_ITERATIONS)
    
    # Control plot display
    show_plot = show_plots_override if show_plots_override is not None else SHOW_PLOTS
    
    paths = get_project_paths()
    
    print_progress(f"📋 Configuration:")
    print_progress(f"   Grid: {GRID_SIZE}x{GRID_SIZE}, Vehicles: {N_VEHICLES}, Time: {SIMULATION_TIME}s")
    print_progress(f"   ACO: {N_ANTS} ants × {N_ITERATIONS} iterations")
    print_progress(f"   Constraints: Green {GREEN_MIN_DURATION}-{GREEN_MAX_DURATION}s, Yellow {YELLOW_MIN_DURATION}-{YELLOW_MAX_DURATION}s")
    
    try:
        # Setup scenario files (this would use existing scenario generation)
        net_file = os.path.join(paths['sumo_data'], f'grid_{GRID_SIZE}x{GRID_SIZE}.net.xml')
        route_file = os.path.join(paths['sumo_data'], f'grid_{GRID_SIZE}x{GRID_SIZE}.rou.xml')
        
        if not os.path.exists(net_file):
            print_progress("❌ Network file not found. Please generate scenario first.")
            return {'success': False, 'error': 'Missing network files'}
        
        # Analyze traffic light phases
        phase_types, default_durations = analyze_traffic_light_phases(net_file)
        n_phases = len(phase_types)
        
        # Initialize pheromone weights (one per phase)
        pheromone_weights = np.ones(n_phases)
        
        # Track optimization progress
        best_costs = []
        best_solutions = []
        best_metrics_history = []
        overall_best_cost = float('inf')
        overall_best_solution = None
        overall_best_metrics = None
        
        print_progress("🔄 Starting optimization iterations...")
        start_time = time.time()
        
        # Main ACO loop
        for iteration in range(N_ITERATIONS):
            print_progress(f"Iteration {iteration + 1}/{N_ITERATIONS}")
            
            # Generate ant solutions
            solutions = []
            scores = []
            
            for ant in range(N_ANTS):
                solution = generate_ant_solution(n_phases, phase_types, pheromone_weights)
                metrics = evaluate_solution(solution, net_file, route_file, paths['temp'])
                cost = calculate_cost(metrics)
                
                solutions.append(solution)
                scores.append(cost)
                
                completion = metrics.get('vehicles', 0)
                if completion > 0:
                    avg_time = metrics['total_time'] / completion
                    print_progress(f"   Ant {ant+1}: {completion}/{N_VEHICLES} vehicles, "
                                 f"avg time: {avg_time:.1f}s, cost: {cost:.1f}")
                else:
                    print_progress(f"   Ant {ant+1}: No vehicles completed, cost: ∞")
            
            # Update pheromones
            pheromone_weights = update_pheromones(pheromone_weights, solutions, scores)
            
            # Track best solution
            iteration_best_idx = np.argmin(scores)
            iteration_best_cost = scores[iteration_best_idx]
            iteration_best_solution = solutions[iteration_best_idx]
            
            # Find best metrics for this iteration
            iteration_best_metrics = None
            for i, (solution, score) in enumerate(zip(solutions, scores)):
                if i == iteration_best_idx:
                    iteration_best_metrics = evaluate_solution(solution, net_file, route_file, paths['temp'])
                    break
            
            best_costs.append(iteration_best_cost)
            best_metrics_history.append(iteration_best_metrics or {'total_time': 0, 'max_stop': 0, 'vehicles': 0})
            
            if iteration_best_cost < overall_best_cost:
                overall_best_cost = iteration_best_cost
                overall_best_solution = iteration_best_solution.copy()
                overall_best_metrics = iteration_best_metrics
                print_progress(f"   🎯 New best cost: {overall_best_cost:.1f}")
        
        duration = time.time() - start_time
        print_progress(f"✅ Optimization completed in {duration:.1f} seconds")
        
        # Create optimization plot
        if len(best_costs) > 0:
            create_optimization_plot(best_costs, best_metrics_history, paths, show_plot)
        
        # Return results
        return {
            'success': True,
            'best_cost': overall_best_cost,
            'best_solution': overall_best_solution,
            'cost_history': best_costs,
            'metrics_history': best_metrics_history,
            'phase_types': phase_types,
            'n_phases': n_phases,
            'duration': duration
        }
        
    except Exception as e:
        print_progress(f"❌ Optimization failed: {e}")
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    results = run_simplified_aco_optimization()
    if results['success']:
        print(f"\n🎉 Best cost achieved: {results['best_cost']:.1f}")
    else:
        print(f"\n❌ Optimization failed: {results.get('error', 'Unknown error')}")
