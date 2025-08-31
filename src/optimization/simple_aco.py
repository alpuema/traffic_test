"""
True Ant Colony Optimization for Traffic Light Timing

This implementation uses traditional pheromone matrices with inter-ant collaboration
while maintaining elite solution conservation for stability.

Key features:
- Traditional pheromone matrix with probabilistic construction
- All ants contribute to collective intelligence
- Elite solution conservation for stability
- Proper exploration/exploitation balance
- True ACO collaborative behavior

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
import platform
from datetime import datetime

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Traffic Engineering Constraints
GREEN_MIN_DURATION = 15        # Minimum green/red phase (includes SUMO defaults)
GREEN_MAX_DURATION = 100       # Maximum green/red phase (includes SUMO defaults)
YELLOW_MIN_DURATION = 3        # Minimum yellow phase (safety standard)
YELLOW_MAX_DURATION = 4        # Maximum yellow phase (safety standard)

# ACO Algorithm Parameters - Traditional ACO with pheromones
N_ANTS = 60                   # Number of ants per iteration
N_ITERATIONS = 5             # Iterations
EVAPORATION_RATE = 0.1        # Pheromone evaporation rate (traditional ACO)
EXPLORATION_RATE = 0.15       # Pure exploration probability
ALPHA = 1.0                   # Pheromone importance weight
BETA = 2.0                    # Heuristic importance weight  
WAITING_PENALTY = 2.0         # Penalty weight for waiting time

# Scenario Configuration
GRID_SIZE = 4                  # Grid dimensions (2 = 2x2, 3 = 3x3, etc.)
N_VEHICLES = 20               # Number of vehicles in simulation
SIMULATION_TIME = 13200         # Increased to accommodate industrial pattern late departures + extra travel time buffer

# Display and Output
SHOW_PROGRESS = True          # Show detailed progress
SHOW_PLOTS = True            # Show optimization plots
LAUNCH_SUMO_GUI = False      # Launch SUMO GUI with results
SAVE_RESULTS = True          # Save results to files

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

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

def create_optimization_plot(best_costs, best_metrics_history, paths, show_plot=None, baseline_comparison=None):
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
    
    # Add baseline cost as dashed horizontal line if available
    if baseline_comparison and 'baseline' in baseline_comparison:
        baseline_cost = baseline_comparison['baseline'].get('cost', None)
        if baseline_cost is not None and baseline_cost != float('inf'):
            plt.axhline(y=baseline_cost, color='r', linestyle='--', linewidth=2, 
                       label=f'Baseline Cost ({baseline_cost:.1f})', alpha=0.8)
    
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set integer x-ticks
    plt.xticks(range(len(best_costs)))
    
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
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set integer x-ticks
    plt.xticks(range(len(best_costs)))
    
    # Plot 3: Max stop time progression
    plt.subplot(3, 1, 3)
    plt.plot(iterations, best_maxstops, 'r-s', linewidth=2, markersize=6, label='Best Max Stop Time')
    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set integer x-ticks
    plt.xticks(range(len(best_costs)))
    
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
                
                # Prefer state-based yellow detection; fallback to duration heuristic
                is_yellow = ('y' in state.lower())
                if not is_yellow and duration <= 6:  # conservative fallback
                    is_yellow = True
                
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
# TRADITIONAL ACO ALGORITHM WITH PHEROMONES
# ============================================================================

def initialize_pheromone_matrix(n_phases, phase_types):
    """Initialize pheromone matrix with small uniform values."""
    pheromone_matrix = {}
    
    for phase_i in range(n_phases):
        pheromone_matrix[phase_i] = {}
        
        if phase_i < len(phase_types) and phase_types[phase_i]:
            # Green/red phase
            duration_range = range(GREEN_MIN_DURATION, GREEN_MAX_DURATION + 1)
        else:
            # Yellow phase
            duration_range = range(YELLOW_MIN_DURATION, YELLOW_MAX_DURATION + 1)
        
        # Initialize with small uniform pheromone levels
        for duration in duration_range:
            pheromone_matrix[phase_i][duration] = 0.1
    
    return pheromone_matrix

def generate_ant_solution(n_phases, phase_types, pheromone_matrix):
    """
    Generate a traffic light solution using pheromone-guided probabilistic construction.
    This is true ACO where each ant's choice is influenced by collective wisdom.

    Args:
        n_phases: Number of phases
        phase_types: True=green/red, False=yellow
        pheromone_matrix: Pheromone levels from previous ants

    Returns:
        List[int]: phase durations
    """
    solution = []

    for phase_i in range(n_phases):
        # Determine valid duration options for this phase
        if phase_i < len(phase_types) and phase_types[phase_i]:
            duration_options = list(range(GREEN_MIN_DURATION, GREEN_MAX_DURATION + 1))
        else:
            duration_options = list(range(YELLOW_MIN_DURATION, YELLOW_MAX_DURATION + 1))

        # Pure exploration with some probability
        if random.random() < EXPLORATION_RATE:
            chosen_duration = random.choice(duration_options)
        else:
            # Pheromone-guided selection (traditional ACO)
            probabilities = []
            
            for duration in duration_options:
                # Get pheromone level (with fallback)
                pheromone = pheromone_matrix.get(phase_i, {}).get(duration, 0.1)
                
                # Simple heuristic (could be enhanced with traffic flow data)
                heuristic = 1.0
                
                # ACO probability formula: τ^α × η^β
                prob = (pheromone ** ALPHA) * (heuristic ** BETA)
                probabilities.append(prob)
            
            # Normalize probabilities
            total_prob = sum(probabilities)
            if total_prob > 0:
                probabilities = [p / total_prob for p in probabilities]
                # Select duration based on collective ant wisdom
                chosen_duration = np.random.choice(duration_options, p=probabilities)
            else:
                chosen_duration = random.choice(duration_options)

        solution.append(chosen_duration)

    return solution

def update_pheromones(pheromone_matrix, all_solutions, all_costs, phase_types):
    """
    Update pheromones based on ALL ant solutions (collective intelligence).
    This is where true ACO collaboration happens.

    Args:
        pheromone_matrix: Current pheromone levels
        all_solutions: Solutions from all ants
        all_costs: Corresponding costs
        phase_types: Phase type information
    """
    n_phases = len(phase_types)
    
    # 1. EVAPORATION: Pheromones decay over time
    for phase_i in range(n_phases):
        if phase_i in pheromone_matrix:
            for duration in list(pheromone_matrix[phase_i].keys()):
                pheromone_matrix[phase_i][duration] *= (1 - EVAPORATION_RATE)
                # Remove very weak pheromone trails
                if pheromone_matrix[phase_i][duration] < 0.01:
                    pheromone_matrix[phase_i][duration] = 0.01

    # 2. REINFORCEMENT: All ants contribute (better solutions contribute more)
    valid_solutions = [(sol, cost) for sol, cost in zip(all_solutions, all_costs) 
                      if np.isfinite(cost) and len(sol) == n_phases]
    
    if not valid_solutions:
        return
    
    # Find best and worst costs for normalization
    costs = [cost for _, cost in valid_solutions]
    min_cost = min(costs)
    max_cost = max(costs)
    cost_range = max_cost - min_cost if max_cost > min_cost else 1.0

    for solution, cost in valid_solutions:
        # Better solutions (lower cost) deposit more pheromone
        if cost_range > 0:
            # Normalize cost to [0, 1] then invert for pheromone amount
            normalized_cost = (cost - min_cost) / cost_range
            pheromone_amount = (1.0 - normalized_cost) + 0.1  # Range: [0.1, 1.1]
        else:
            pheromone_amount = 1.0
        
        # Deposit pheromone on the path this ant took
        for phase_i, duration in enumerate(solution):
            if phase_i >= n_phases:
                break
                
            if phase_i not in pheromone_matrix:
                pheromone_matrix[phase_i] = {}
            if duration not in pheromone_matrix[phase_i]:
                pheromone_matrix[phase_i][duration] = 0.1
            
            pheromone_matrix[phase_i][duration] += pheromone_amount

    # 3. ELITE REINFORCEMENT: Give extra boost to the best solution
    if valid_solutions:
        best_solution, best_cost = min(valid_solutions, key=lambda x: x[1])
        elite_boost = 2.0  # Elite solutions get double reinforcement
        
        for phase_i, duration in enumerate(best_solution):
            if phase_i >= n_phases:
                break
            if phase_i in pheromone_matrix and duration in pheromone_matrix[phase_i]:
                pheromone_matrix[phase_i][duration] += elite_boost

# ============================================================================
# SUMO GUI VISUALIZATION
# ============================================================================

def launch_sumo_gui_with_solution(best_solution, net_file, route_file, paths):
    """
    Launch SUMO GUI with the optimized traffic light solution applied.
    
    Args:
        best_solution: List of optimized phase durations
        net_file: Original SUMO network file
        route_file: SUMO route file  
        paths: Project paths dictionary
    """
    try:
        print_progress("🖥️  Preparing SUMO GUI with optimized solution...")
        
        # Import sumolib for GUI binary check
        import sumolib
        
        # Create optimized network file
        gui_net_file = os.path.join(paths['results'], 'optimized_solution.net.xml')
        gui_cfg_file = os.path.join(paths['results'], 'optimized_solution.sumocfg')
        
        # Copy original network and apply optimized solution
        import shutil
        shutil.copy2(net_file, gui_net_file)
        apply_solution_to_network(gui_net_file, best_solution)
        
        # Create SUMO configuration for GUI
        create_gui_sumo_config(gui_cfg_file, gui_net_file, route_file)
        
        # Check for SUMO GUI binary
        try:
            sumo_gui = sumolib.checkBinary('sumo-gui')
        except:
            # Fallback: try common SUMO installation paths
            import platform
            if platform.system() == "Windows":
                possible_paths = [
                    r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe",
                    r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe", 
                    "sumo-gui.exe"
                ]
            else:
                possible_paths = ["/usr/bin/sumo-gui", "/usr/local/bin/sumo-gui", "sumo-gui"]
            
            sumo_gui = None
            for path in possible_paths:
                if shutil.which(path):
                    sumo_gui = path
                    break
            
            if not sumo_gui:
                raise FileNotFoundError("SUMO GUI binary not found")
        
        # Launch SUMO GUI
        print_progress(f"🚀 Launching SUMO GUI with optimized solution...")
        print_progress(f"   Network: {os.path.basename(gui_net_file)}")
        print_progress(f"   Config: {os.path.basename(gui_cfg_file)}")
        print_progress(f"   💡 Use the play button in SUMO GUI to start the simulation!")
        
        subprocess.Popen([sumo_gui, '-c', gui_cfg_file])
        
        # Give user instructions
        print_progress("")
        print_progress("🎮 SUMO GUI Controls:")
        print_progress("   • Click the Play button (►) to start simulation")
        print_progress("   • Use + and - to zoom in/out")
        print_progress("   • Right-click on intersections to see traffic light phases")
        print_progress("   • View → Time Display to see simulation time")
        print_progress("   • The simulation will show your optimized traffic light timings!")
        print_progress("")
        
        return True
        
    except Exception as e:
        print_progress(f"❌ Failed to launch SUMO GUI: {e}")
        print_progress("💡 Manual launch instructions:")
        print_progress(f"   1. Open command prompt in project directory")
        print_progress(f"   2. Run: sumo-gui -c results/optimized_solution.sumocfg")
        print_progress(f"   3. Click play to see your optimized traffic lights!")
        return False

def create_gui_sumo_config(cfg_file, net_file, route_file):
    """Create SUMO configuration file optimized for GUI visualization."""
    net_file_abs = os.path.abspath(net_file)
    route_file_abs = os.path.abspath(route_file)
    
    # Get the vtype file path
    sumo_data_dir = os.path.dirname(route_file_abs)
    vtype_file = os.path.join(sumo_data_dir, 'vtype.add.xml')
    vtype_file_abs = os.path.abspath(vtype_file)
    
    # Extended simulation time for better visualization
    gui_simulation_time = max(SIMULATION_TIME, 1800)  # At least 30 minutes for GUI
    
    config_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net_file_abs}"/>
        <route-files value="{route_file_abs}"/>
        <additional-files value="{vtype_file_abs}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{gui_simulation_time}"/>
        <step-length value="1"/>
    </time>
    <gui>
        <gui-settings-file value=""/>
        <window-size value="1000,800"/>
        <window-pos value="50,50"/>
        <start value="true"/>
    </gui>
    <processing>
        <time-to-teleport value="300"/>
        <max-depart-delay value="900"/>
    </processing>
</configuration>'''
    
    with open(cfg_file, 'w') as f:
        f.write(config_content)

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
        create_sumo_config(temp_cfg_file, temp_net_file, route_file, temp_tripinfo_file, SIMULATION_TIME)
        
        # Run SUMO simulation
        result = subprocess.run([
            'sumo', '-c', temp_cfg_file,
            '--no-warnings', '--no-step-log',
            '--time-to-teleport', '300'  # Allow more time before teleporting stuck vehicles
        ], capture_output=True, text=True, timeout=300)  # Increase timeout to 5 minutes
        
        # Debug: Check if simulation had errors
        if result.returncode != 0:
            print_progress(f"   ⚠️  SUMO simulation failed with return code {result.returncode}")
            if result.stderr:
                print_progress(f"   SUMO stderr: {result.stderr[:200]}")
        
        # Parse results
        if os.path.exists(temp_tripinfo_file):
            metrics = parse_tripinfo_file(temp_tripinfo_file)
            # Debug: Show vehicle completion info
            vehicles_completed = metrics.get('vehicles', 0)
            if vehicles_completed == 0:
                print_progress(f"   ⚠️  No vehicles completed in tripinfo file")
                # Check file size to see if it's empty
                file_size = os.path.getsize(temp_tripinfo_file)
                print_progress(f"   Tripinfo file size: {file_size} bytes")
            elif vehicles_completed < N_VEHICLES:
                print_progress(f"   ⚠️  Only {vehicles_completed}/{N_VEHICLES} vehicles completed")
                # Check SUMO output for clues about missing vehicles
                if result.stderr and ("teleport" in result.stderr.lower() or "collision" in result.stderr.lower()):
                    print_progress(f"   SUMO issues detected: {result.stderr[:100]}...")
        else:
            print_progress(f"   ⚠️  Tripinfo file not created: {temp_tripinfo_file}")
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

def create_sumo_config(cfg_file, net_file, route_file, tripinfo_file, sim_time=None):
    """Create SUMO configuration file."""
    # Use absolute paths to avoid path issues
    net_file_abs = os.path.abspath(net_file)
    route_file_abs = os.path.abspath(route_file)
    tripinfo_file_abs = os.path.abspath(tripinfo_file)
    
    # Get the vtype file path
    sumo_data_dir = os.path.dirname(route_file_abs)
    vtype_file = os.path.join(sumo_data_dir, 'vtype.add.xml')
    vtype_file_abs = os.path.abspath(vtype_file)
    
    # Use passed simulation time or global default
    simulation_time = sim_time if sim_time is not None else SIMULATION_TIME
    
    config_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net_file_abs}"/>
        <route-files value="{route_file_abs}"/>
        <additional-files value="{vtype_file_abs}"/>
    </input>
    <output>
        <tripinfo-output value="{tripinfo_file_abs}"/>
    </output>
    <time>
        <end value="{simulation_time}"/>
    </time>
</configuration>'''
    
    with open(cfg_file, 'w') as f:
        f.write(config_content)

def parse_tripinfo_file(tripinfo_file):
    """Parse SUMO tripinfo output to extract performance metrics."""
    try:
        tree = ET.parse(tripinfo_file)
        root = tree.getroot()

        total_time = 0.0
        max_stop = 0.0
        vehicle_count = 0
        waiting_times = []
        completed_vehicle_ids = []

        for trip in root.findall('tripinfo'):
            duration = float(trip.get('duration', '0'))
            waiting_time = float(trip.get('waitingTime', '0'))
            vehicle_id = trip.get('id', 'unknown')

            total_time += duration
            max_stop = max(max_stop, waiting_time)
            vehicle_count += 1
            waiting_times.append(waiting_time)
            completed_vehicle_ids.append(vehicle_id)

        # Debug info: show which vehicles completed
        if vehicle_count < N_VEHICLES:
            print_progress(f"   ⚠️  Only {vehicle_count}/{N_VEHICLES} vehicles completed")
            # Show some completed IDs for debugging
            if len(completed_vehicle_ids) > 0:
                sample_ids = completed_vehicle_ids[:5]  # Show first 5
                print_progress(f"   Completed vehicles (sample): {', '.join(sample_ids)}")

        wait_p95 = float(np.percentile(waiting_times, 95)) if waiting_times else 0.0
        avg_wait = float(np.mean(waiting_times)) if waiting_times else 0.0
        return {
            'total_time': total_time,
            'max_stop': max_stop,
            'wait_p95': wait_p95,
            'avg_wait': avg_wait,
            'vehicles': vehicle_count,
            'completed_ids': completed_vehicle_ids
        }
        
    except Exception as e:
        print_progress(f"   ⚠️  Error parsing tripinfo: {e}")
        return {'total_time': float('inf'), 'max_stop': 0, 'vehicles': 0}

def calculate_cost(metrics):
    """Calculate cost function from simulation metrics."""
    total_time = metrics.get('total_time', float('inf'))
    # Prefer robust 95th percentile waiting time
    wait_p95 = metrics.get('wait_p95', None)
    max_stop = metrics.get('max_stop', 0)
    vehicles = metrics.get('vehicles', 1)
    
    if total_time == float('inf') or vehicles == 0:
        return float('inf')
    
    # Cost = average travel time + penalty for high waiting time (robust)
    avg_time = total_time / vehicles
    wait_component = wait_p95 if wait_p95 is not None else max_stop
    # Cap waiting contribution to reduce outlier impact
    wait_component = min(wait_component, 60.0)
    return avg_time + WAITING_PENALTY * wait_component

# ============================================================================
# BASELINE COMPARISON FUNCTIONS
# ============================================================================

def explain_traffic_light_phases():
    """
    Explain how SUMO traffic light phases work.
    
    SUMO Traffic Light Structure:
    ============================
    
    Traffic lights in SUMO use a cycle of phases, where each phase has:
    - Duration: How long this phase lasts (what we optimize)
    - State: Which directions get green/yellow/red (fixed, ensures safety)
    
    Example 4-phase cycle:
    Phase 1: duration="42" state="GggrrrGGg"  ← North-South green, East-West red
    Phase 2: duration="3"  state="yyyrrrGyy"  ← North-South yellow transition
    Phase 3: duration="42" state="rrrGGgGrr"  ← East-West green, North-South red  
    Phase 4: duration="3"  state="rrryyyGrr"  ← East-West yellow transition
    
    State symbols:
    - G = Green (high priority)
    - g = Green (low priority, e.g., right turns)
    - r = Red (stop)
    - y = Yellow (prepare to stop)
    
    Why this is efficient:
    =====================
    1. ONLY DURATIONS are optimized (small search space)
    2. Red timing is AUTOMATIC - when one direction is green, conflicts are red
    3. Safety is GUARANTEED - conflicting flows can't be green simultaneously
    4. The ACO only changes HOW LONG each phase lasts, not WHICH directions are green
    
    This means:
    - Green duration of 30s → Cars from that direction get 30s to go
    - Automatically → Conflicting directions get 30s of red
    - Yellow duration of 4s → 4s warning before switching
    """
    pass

def create_baseline_solution(phase_types, green_duration=30, yellow_duration=4):
    """
    Create a baseline solution with uniform timings.
    
    Args:
        phase_types: List of booleans (True=green/red phase, False=yellow phase)
        green_duration: Duration for green/red phases in seconds
        yellow_duration: Duration for yellow phases in seconds
    
    Returns:
        List of phase durations for baseline comparison
    """
    baseline_solution = []
    for is_green_phase in phase_types:
        if is_green_phase:
            baseline_solution.append(green_duration)
        else:
            baseline_solution.append(yellow_duration)
    
    return baseline_solution

def evaluate_baseline_comparison(best_solution, phase_types, net_file, route_file, temp_dir):
    """
    Compare the optimized solution against a baseline uniform timing.
    
    Args:
        best_solution: Optimized phase durations
        phase_types: Phase type information
        net_file: SUMO network file
        route_file: SUMO route file
        temp_dir: Temporary directory
    
    Returns:
        Dictionary with comparison results
    """
    print_progress("📊 Evaluating baseline comparison...")
    
    # Create baseline solution (30s green, 4s yellow)
    baseline_solution = create_baseline_solution(phase_types, green_duration=30, yellow_duration=4)
    
    # Evaluate baseline solution
    print_progress("   Evaluating baseline (30s green, 4s yellow)...")
    baseline_metrics = evaluate_solution(baseline_solution, net_file, route_file, temp_dir)
    baseline_cost = calculate_cost(baseline_metrics)
    
    # Evaluate optimized solution
    print_progress("   Evaluating optimized solution...")
    optimized_metrics = evaluate_solution(best_solution, net_file, route_file, temp_dir)
    optimized_cost = calculate_cost(optimized_metrics)
    
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
        }
    }
    
    # Print comparison results
    print_progress(f"📊 BASELINE COMPARISON RESULTS:")
    print_progress(f"   Baseline (30s/4s): Cost = {baseline_cost:.1f}")
    print_progress(f"   Optimized solution: Cost = {optimized_cost:.1f}")
    
    if improvement_percent > 0:
        print_progress(f"   ✅ Improvement: {improvement_percent:.1f}% better ({absolute_improvement:.1f} cost units)")
    elif improvement_percent < 0:
        print_progress(f"   ❌ Degradation: {abs(improvement_percent):.1f}% worse ({abs(absolute_improvement):.1f} cost units)")
    else:
        print_progress(f"   ➖ No significant difference")
    
    return comparison_results

# ============================================================================
# MAIN OPTIMIZATION FUNCTION
# ============================================================================

def extract_files_from_sumo_config(sumo_config_file):
    """Extract network and route file paths from SUMO config file."""
    try:
        tree = ET.parse(sumo_config_file)
        root = tree.getroot()
        
        net_file = None
        route_file = None
        
        # Find net-file
        net_elem = root.find('.//net-file')
        if net_elem is not None:
            net_file = net_elem.get('value')
            # Convert to absolute path if relative
            if net_file and not os.path.isabs(net_file):
                net_file = os.path.join(os.path.dirname(sumo_config_file), net_file)
        
        # Find route-files  
        route_elem = root.find('.//route-files')
        if route_elem is not None:
            route_file = route_elem.get('value')
            # Convert to absolute path if relative
            if route_file and not os.path.isabs(route_file):
                route_file = os.path.join(os.path.dirname(sumo_config_file), route_file)
        
        return net_file, route_file
        
    except Exception as e:
        print_progress(f"⚠️  Error parsing SUMO config: {e}")
        return None, None

def run_traditional_aco_optimization(config=None, show_plots_override=None, show_gui_override=None, compare_baseline=True, sumo_config_file=None):
    """
    Run the simplified ACO optimization.
    
    Args:
        config: Optional configuration dictionary
        show_plots_override: Boolean to control plot display (overrides global SHOW_PLOTS)
        show_gui_override: Boolean to control GUI launch (overrides global LAUNCH_SUMO_GUI)
        sumo_config_file: Path to SUMO config file (if provided, network/route files will be extracted from it)
    
    Returns:
        Dictionary with optimization results
    """
    print("🐜 TRUE ANT COLONY OPTIMIZATION")
    print("=" * 50)
    
    # Apply configuration if provided
    if config:
        global GRID_SIZE, N_VEHICLES, SIMULATION_TIME, N_ANTS, N_ITERATIONS
        global EVAPORATION_RATE, EXPLORATION_RATE, ALPHA, BETA, WAITING_PENALTY

        GRID_SIZE = config.get('grid_size', GRID_SIZE)
        N_VEHICLES = config.get('n_vehicles', N_VEHICLES)
        SIMULATION_TIME = config.get('simulation_time', SIMULATION_TIME)
        N_ANTS = config.get('n_ants', N_ANTS)
        N_ITERATIONS = config.get('n_iterations', N_ITERATIONS)

        # ACO-specific parameters
        EVAPORATION_RATE = config.get('evaporation_rate', EVAPORATION_RATE)
        EXPLORATION_RATE = config.get('exploration_rate', EXPLORATION_RATE)
        ALPHA = config.get('pheromone_weight', ALPHA)  # Pheromone importance
        BETA = config.get('heuristic_weight', BETA)    # Heuristic importance
        WAITING_PENALTY = config.get('stop_penalty', WAITING_PENALTY)  # Cost function penalty

        print_progress(f"   Applied custom parameters:")
        print_progress(f"   Evaporation: {EVAPORATION_RATE}, Exploration: {EXPLORATION_RATE}, Penalty: {ALPHA}")
    
    # Control plot display and GUI launch
    show_plot = show_plots_override if show_plots_override is not None else SHOW_PLOTS
    launch_gui = show_gui_override if show_gui_override is not None else LAUNCH_SUMO_GUI
    
    paths = get_project_paths()
    
    print_progress(f"📋 Configuration:")
    print_progress(f"   Grid: {GRID_SIZE}x{GRID_SIZE}, Vehicles: {N_VEHICLES}, Time: {SIMULATION_TIME}s")
    print_progress(f"   ACO: {N_ANTS} ants × {N_ITERATIONS} iterations")
    print_progress(f"   Constraints: Green {GREEN_MIN_DURATION}-{GREEN_MAX_DURATION}s, Yellow {YELLOW_MIN_DURATION}-{YELLOW_MAX_DURATION}s")
    
    try:
        # Setup scenario files
        net_file = None
        route_file = None
        
        # If SUMO config file is provided, extract file paths from it
        if sumo_config_file and os.path.exists(sumo_config_file):
            print_progress(f"📁 Using SUMO config: {os.path.basename(sumo_config_file)}")
            net_file, route_file = extract_files_from_sumo_config(sumo_config_file)
        
        # Fallback to default file path logic if no config provided or extraction failed
        if not net_file or not route_file or not os.path.exists(net_file) or not os.path.exists(route_file):
            net_file = os.path.join(paths['sumo_data'], f'grid_{GRID_SIZE}x{GRID_SIZE}.net.xml')
            route_file = os.path.join(paths['sumo_data'], f'grid_{GRID_SIZE}x{GRID_SIZE}.rou.xml')
            if not os.path.exists(route_file):
                alt = os.path.join(paths['sumo_data'], f'grid_{GRID_SIZE}x{GRID_SIZE}.rou.alt.xml')
                if os.path.exists(alt):
                    route_file = alt

        if not os.path.exists(net_file):
            print_progress("❌ Network file not found. Please generate scenario first.")
            return {'success': False, 'error': 'Missing network files'}
            
        if not os.path.exists(route_file):
            print_progress(f"❌ Route file not found: {route_file}")
            return {'success': False, 'error': 'Missing route file'}
            
        print_progress(f"📁 Using network file: {os.path.basename(net_file)}")
        print_progress(f"📁 Using route file: {os.path.basename(route_file)}")

        # Analyze traffic light phases
        phase_types, default_durations = analyze_traffic_light_phases(net_file)
        n_phases = len(phase_types)

        # Initialize pheromone matrix for traditional ACO
        pheromone_matrix = initialize_pheromone_matrix(n_phases, phase_types)

        # Track optimization progress
        best_costs = []
        best_solutions = []
        best_metrics_history = []
        overall_best_cost = float('inf')
        overall_best_solution = None
        overall_best_metrics = None

        # Track the absolute best
        global_best_cost = float('inf')
        global_best_solution = None
        global_best_metrics = None

        print_progress("🔄 Starting optimization iterations...")
        start_time = time.time()

        # Main ACO loop
        for iteration in range(N_ITERATIONS):
            print_progress(f"Iteration {iteration + 1}/{N_ITERATIONS}")

            # Generate ant solutions
            solutions = []
            scores = []
            metrics_list = []

            # Optional: include elite solution for stability (no re-eval needed)
            if global_best_solution is not None:
                solutions.append(global_best_solution.copy())
                scores.append(global_best_cost)
                metrics_list.append(global_best_metrics)
                print_progress(f"   Elite solution injected: cost {global_best_cost:.1f}")

            # Generate remaining ant solutions
            remaining_ants = N_ANTS - (1 if global_best_solution is not None else 0)
            for ant in range(remaining_ants):
                solution = generate_ant_solution(n_phases, phase_types, pheromone_matrix)
                metrics = evaluate_solution(solution, net_file, route_file, paths['temp'])
                cost = calculate_cost(metrics)

                solutions.append(solution)
                scores.append(cost)
                metrics_list.append(metrics)

                # Update global best immediately when found
                if cost < global_best_cost:
                    global_best_cost = cost
                    global_best_solution = solution.copy()
                    global_best_metrics = metrics
                    print_progress(f"   🌟 NEW GLOBAL BEST: Ant {ant+1}, cost: {cost:.1f}")

                completion = metrics.get('vehicles', 0)
                if completion > 0:
                    avg_time = metrics['total_time'] / completion
                    print_progress(f"   Ant {ant+1}: {completion}/{N_VEHICLES} vehicles completed, "
                                   f"avg time: {avg_time:.1f}s, cost: {cost:.1f}")
                else:
                    print_progress(f"   Ant {ant+1}: 0/{N_VEHICLES} vehicles completed, cost: ∞")

            # Update pheromones based on ALL ant solutions (collective intelligence)
            update_pheromones(pheromone_matrix, solutions, scores, phase_types)

            # Track best solution with stability checks
            iteration_best_idx = int(np.argmin(scores))
            iteration_best_cost = scores[iteration_best_idx]
            iteration_best_metrics = metrics_list[iteration_best_idx]

            # Always track the global best (not iteration best) for stability
            best_costs.append(global_best_cost)
            best_metrics_history.append(iteration_best_metrics or {'total_time': 0, 'max_stop': 0, 'vehicles': 0})

            # Update overall tracking (legacy compatibility)
            if global_best_cost < overall_best_cost:
                overall_best_cost = global_best_cost
                overall_best_solution = global_best_solution.copy()
                overall_best_metrics = iteration_best_metrics

        duration = time.time() - start_time
        print_progress(f"✅ Optimization completed in {duration:.1f} seconds")

        # Baseline comparison if requested
        baseline_comparison = None
        if compare_baseline and overall_best_solution is not None:
            baseline_comparison = evaluate_baseline_comparison(
                overall_best_solution, phase_types, net_file, route_file, paths['temp']
            )

        # Create optimization plot
        if len(best_costs) > 0:
            create_optimization_plot(best_costs, best_metrics_history, paths, show_plot, baseline_comparison)

        # Launch SUMO GUI with optimized solution if requested
        if launch_gui and overall_best_solution is not None:
            print_progress("")
            launch_sumo_gui_with_solution(overall_best_solution, net_file, route_file, paths)

        # Return results
        return {
            'success': True,
            'best_cost': overall_best_cost,
            'best_solution': overall_best_solution,
            'cost_history': best_costs,
            'metrics_history': best_metrics_history,
            'phase_types': phase_types,
            'n_phases': n_phases,
            'duration': duration,
            'baseline_comparison': baseline_comparison
        }
        
    except Exception as e:
        print_progress(f"❌ Optimization failed: {e}")
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    results = run_traditional_aco_optimization()
    if results['success']:
        print(f"\n🎉 Best cost achieved: {results['best_cost']:.1f}")
    else:
        print(f"\n❌ Optimization failed: {results.get('error', 'Unknown error')}")
