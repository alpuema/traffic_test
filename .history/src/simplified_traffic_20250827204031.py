"""
Simplified Traffic Pattern Generation for SUMO

This module provides clean, easy-to-understand traffic pattern generation
with different modes (random, commuter, commercial, etc.) and proper
seed support for reproducible results.

Author: Traffic Optimization System  
Date: August 2025
"""

import xml.etree.ElementTree as ET
import subprocess
import os
import random
import json
from datetime import datetime

# ============================================================================
# TRAFFIC PATTERN CONFIGURATIONS
# ============================================================================

TRAFFIC_PATTERNS = {
    'commuter': {
        'description': 'Rush hour commuter pattern - suburbs to downtown (perimeter to center)',
        'explanation': 'Vehicles start from outer edges (suburbs/residential) and travel to inner city center (downtown/business district). Mimics typical morning rush hour but balanced to avoid extreme concentration.',
        'source_weights': {'perimeter': 2.0, 'center': 0.5},
        'sink_weights': {'center': 2.0, 'perimeter': 0.5},
        'time_distribution': 'early_heavy'
    },
    
    'industrial': {
        'description': 'Industrial corridor pattern - horizontal left-to-right traffic flow',
        'explanation': 'Vehicles travel horizontally across the grid, primarily from left side to right side. Simulates industrial zones with main cargo/freight corridors but includes some return traffic.',
        'source_weights': {'left_edge': 2.0, 'right_edge': 0.5},
        'sink_weights': {'right_edge': 2.0, 'left_edge': 0.5}, 
        'time_distribution': 'shift_based'
    },
    
    'random': {
        'description': 'Completely random vehicle origins and destinations',
        'explanation': 'Pure random selection - useful for testing and comparison. No spatial patterns or correlations.',
        'source_weights': {'all': 1.0},
        'sink_weights': {'all': 1.0},
        'time_distribution': 'uniform'
    }
}

# ============================================================================
# CORE TRAFFIC GENERATION FUNCTIONS
# ============================================================================

def generate_network_and_routes(grid_size, n_vehicles, sim_time, pattern='balanced', seed=None, output_dir=None):
    """
    Generate SUMO network and route files with specified traffic pattern.
    
    Args:
        grid_size: Size of grid network (e.g., 3 for 3x3)
        n_vehicles: Number of vehicles to generate
        sim_time: Simulation duration in seconds
        pattern: Traffic pattern name from TRAFFIC_PATTERNS
        seed: Random seed for reproducible results
        output_dir: Directory to save files (defaults to sumo_data)
    
    Returns:
        Dictionary with file paths and generation info
    """
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
    
    # Setup output directory
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        output_dir = os.path.join(project_root, 'sumo_data')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths
    net_file = os.path.join(output_dir, f'grid_{grid_size}x{grid_size}.net.xml')
    route_file = os.path.join(output_dir, f'grid_{grid_size}x{grid_size}.rou.xml') 
    trips_file = os.path.join(output_dir, f'grid_{grid_size}x{grid_size}.trips.xml')
    sumocfg_file = os.path.join(output_dir, f'grid_{grid_size}x{grid_size}.sumocfg')
    vtype_file = os.path.join(output_dir, 'vtype.add.xml')
    
    print(f"🏗️  Generating {grid_size}x{grid_size} grid with {n_vehicles} vehicles ({pattern} pattern)")
    
    try:
        # Generate network
        success = generate_grid_network(grid_size, net_file)
        if not success:
            return {'success': False, 'error': 'Network generation failed'}
        
        # Generate vehicle types
        create_vehicle_types_file(vtype_file)
        
        # Generate traffic based on pattern
        pattern_config = TRAFFIC_PATTERNS.get(pattern, TRAFFIC_PATTERNS['balanced'])
        success = generate_traffic_pattern(
            net_file, trips_file, n_vehicles, sim_time, pattern_config, seed
        )
        if not success:
            return {'success': False, 'error': 'Traffic generation failed'}
        
        # Convert trips to routes
        success = convert_trips_to_routes(net_file, trips_file, route_file)
        if not success:
            return {'success': False, 'error': 'Route conversion failed'}
        
        # Create SUMO configuration
        create_sumocfg_file(sumocfg_file, net_file, route_file, vtype_file, sim_time)
        
        print(f"✅ Generated complete scenario: {sumocfg_file}")
        
        return {
            'success': True,
            'files': {
                'network': net_file,
                'routes': route_file,
                'trips': trips_file,
                'config': sumocfg_file,
                'vtypes': vtype_file
            },
            'pattern': pattern,
            'seed': seed,
            'grid_size': grid_size,
            'n_vehicles': n_vehicles,
            'sim_time': sim_time
        }
        
    except Exception as e:
        print(f"❌ Error generating scenario: {e}")
        return {'success': False, 'error': str(e)}

def generate_grid_network(grid_size, output_file):
    """Generate a grid network using SUMO's netgenerate."""
    try:
        # Create junction names for traffic light placement
        junction_names = []
        for row in range(grid_size):
            for col in range(grid_size):
                junction_names.append(f"{chr(65 + col)}{row}")
        
        # Generate grid network with traffic lights
        result = subprocess.run([
            'netgenerate',
            '--grid',
            '--grid.number', str(grid_size),
            '--grid.length', '200',  # 200m edges
            '--default.lanenumber', '1',
            '--default.speed', '13.89',  # 50 km/h
            '--output-file', output_file,
            '--tls.set', ','.join(junction_names),  # Add traffic lights
            '--tls.default-type', 'static'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ✅ Network generated: {output_file}")
            return True
        else:
            print(f"   ❌ Network generation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error generating network: {e}")
        return False

def generate_traffic_pattern(net_file, trips_file, n_vehicles, sim_time, pattern_config, seed):
    """Generate traffic trips based on specified pattern."""
    try:
        # Parse network to get available edges
        edges = get_network_edges(net_file)
        if len(edges) < 2:
            print(f"   ❌ Insufficient edges found: {len(edges)}")
            return False
        
        # Categorize edges by location
        edge_categories = categorize_edges(edges)
        
        # Generate trips based on pattern
        trips = []
        departure_window = sim_time * 0.8  # Use first 80% for departures
        
        for i in range(n_vehicles):
            # Calculate departure time based on pattern
            depart_time = calculate_departure_time(i, n_vehicles, departure_window, pattern_config)
            
            # Select source and destination based on pattern weights
            from_edge = select_weighted_edge(edge_categories, pattern_config['source_weights'])
            to_edge = select_weighted_edge(edge_categories, pattern_config['sink_weights'])
            
            # Ensure different source and destination
            attempts = 0
            while from_edge == to_edge and attempts < 10:
                to_edge = select_weighted_edge(edge_categories, pattern_config['sink_weights'])
                attempts += 1
            
            trips.append({
                'id': f'trip_{i}',
                'depart': f'{depart_time:.2f}',
                'from': from_edge,
                'to': to_edge
            })
        
        # Write trips file
        write_trips_file(trips_file, trips)
        print(f"   ✅ Generated {n_vehicles} trips: {trips_file}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error generating traffic pattern: {e}")
        return False

def get_network_edges(net_file):
    """Extract usable edges from network file."""
    try:
        tree = ET.parse(net_file)
        root = tree.getroot()
        
        edges = []
        for edge in root.findall('edge'):
            edge_id = edge.get('id')
            # Skip internal junction edges
            if edge_id and not edge_id.startswith(':'):
                edges.append(edge_id)
        
        return edges
        
    except Exception as e:
        print(f"   ⚠️  Error parsing network: {e}")
        return []

def categorize_edges(edges):
    """Categorize edges by their likely function (perimeter, center, etc.)."""
    # This is a simplified categorization based on edge names
    # In a real scenario, you might use coordinates or other attributes
    
    categories = {
        'all': edges,
        'perimeter': [],
        'center': [],
        'residential': [],
        'commercial': [],
        'industrial': [],
        'main_roads': []
    }
    
    for edge in edges:
        # Simple heuristic based on edge naming
        if any(char in edge for char in ['A0', 'A2', '0A', '2A']):  # Corner/edge positions
            categories['perimeter'].append(edge)
        elif 'B1' in edge or '1B' in edge:  # Center area
            categories['center'].append(edge)
        else:
            categories['residential'].append(edge)
    
    # Fill empty categories with all edges as fallback
    for category in categories:
        if category != 'all' and not categories[category]:
            categories[category] = edges
    
    return categories

def select_weighted_edge(edge_categories, weights):
    """Select an edge based on weighted categories."""
    # Create weighted pool of edges
    weighted_edges = []
    
    for category, weight in weights.items():
        if category in edge_categories:
            for edge in edge_categories[category]:
                # Add edge multiple times based on weight
                weighted_edges.extend([edge] * int(weight * 10))
    
    if not weighted_edges:
        weighted_edges = edge_categories.get('all', ['edge1', 'edge2'])
    
    return random.choice(weighted_edges)

def calculate_departure_time(vehicle_idx, total_vehicles, departure_window, pattern_config):
    """Calculate departure time based on pattern distribution."""
    time_dist = pattern_config.get('time_distribution', 'uniform')
    
    if time_dist == 'uniform':
        return vehicle_idx * (departure_window / total_vehicles)
    
    elif time_dist == 'early_heavy':
        # More vehicles depart early (commuter rush)
        base_time = vehicle_idx * (departure_window / total_vehicles)
        early_bias = departure_window * 0.2 * (1 - vehicle_idx / total_vehicles)
        return max(1.0, base_time - early_bias)  # Ensure minimum 1 second
    
    elif time_dist == 'mid_heavy':
        # Peak in middle of time window (shopping pattern)
        base_time = vehicle_idx * (departure_window / total_vehicles)
        mid_point = departure_window * 0.5
        if mid_point > 0:
            mid_bias = departure_window * 0.1 * (1 - abs(base_time - mid_point) / mid_point)
        else:
            mid_bias = 0
        return max(1.0, base_time + mid_bias)  # Ensure minimum 1 second
    
    elif time_dist == 'shift_based':
        # Industrial shift pattern - concentrated at shift times
        shift_starts = [0.1 * departure_window, 0.6 * departure_window]
        closest_shift = min(shift_starts, key=lambda x: abs(x - vehicle_idx * departure_window / total_vehicles))
        base_time = closest_shift + random.uniform(-departure_window * 0.05, departure_window * 0.05)
        return max(1.0, base_time)
    
    else:  # 'normal' or unknown - use uniform with slight randomization
        base_time = vehicle_idx * (departure_window / total_vehicles)
        randomization = random.uniform(-departure_window * 0.02, departure_window * 0.02)
        return max(1.0, base_time + randomization)  # Ensure minimum 1 second

def write_trips_file(trips_file, trips):
    """Write trips to SUMO trips XML file."""
    content = ['<?xml version="1.0" encoding="UTF-8"?>']
    content.append('<trips>')
    
    for trip in trips:
        content.append(f'    <trip id="{trip["id"]}" depart="{trip["depart"]}" '
                      f'from="{trip["from"]}" to="{trip["to"]}"/>')
    
    content.append('</trips>')
    
    with open(trips_file, 'w') as f:
        f.write('\n'.join(content))

def convert_trips_to_routes(net_file, trips_file, route_file):
    """Convert trips to routes using SUMO's duarouter."""
    try:
        result = subprocess.run([
            'duarouter',
            '--net-file', net_file,
            '--trip-files', trips_file,
            '--output-file', route_file,
            '--ignore-errors'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ✅ Routes generated: {route_file}")
            return True
        else:
            print(f"   ❌ Route generation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error converting trips to routes: {e}")
        return False

def create_vehicle_types_file(vtype_file):
    """Create vehicle types definition file."""
    content = '''<?xml version="1.0" encoding="UTF-8"?>
<additional>
    <vType id="car" accel="2.0" decel="4.5" sigma="0.5" length="5.0" maxSpeed="20.0"/>
    <vType id="bus" accel="1.0" decel="3.0" sigma="0.2" length="12.0" maxSpeed="15.0"/>
</additional>'''
    
    with open(vtype_file, 'w') as f:
        f.write(content)

def create_sumocfg_file(sumocfg_file, net_file, route_file, vtype_file, sim_time):
    """Create SUMO configuration file."""
    content = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{os.path.basename(net_file)}"/>
        <route-files value="{os.path.basename(route_file)}"/>
        <additional-files value="{os.path.basename(vtype_file)}"/>
    </input>
    <output>
        <tripinfo-output value="tripinfo.xml"/>
    </output>
    <time>
        <end value="{sim_time}"/>
    </time>
    <processing>
        <time-to-teleport value="300"/>
    </processing>
</configuration>'''
    
    with open(sumocfg_file, 'w') as f:
        f.write(content)

# ============================================================================
# SOLUTION SAVING AND LOADING
# ============================================================================

def save_optimized_solution(solution, metadata, output_dir):
    """
    Save an optimized traffic light solution for later evaluation.
    
    Args:
        solution: List of phase durations
        metadata: Dictionary with optimization info (seed, pattern, etc.)
        output_dir: Directory to save solution files
    
    Returns:
        Dictionary with saved file paths
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    solution_name = f"solution_{metadata.get('pattern', 'unknown')}_{timestamp}"
    
    # Save solution data
    solution_file = os.path.join(output_dir, f"{solution_name}.json")
    solution_data = {
        'solution': solution,
        'metadata': metadata,
        'timestamp': timestamp,
        'version': '2.0'
    }
    
    with open(solution_file, 'w') as f:
        json.dump(solution_data, f, indent=2)
    
    print(f"💾 Solution saved: {solution_file}")
    
    return {'solution_file': solution_file, 'name': solution_name}

def load_solution(solution_file):
    """Load a previously saved optimization solution."""
    try:
        with open(solution_file, 'r') as f:
            data = json.load(f)
        
        print(f"📂 Loaded solution: {solution_file}")
        return data
        
    except Exception as e:
        print(f"❌ Error loading solution: {e}")
        return None

def evaluate_solution_with_new_seed(solution_file, new_seed, n_vehicles=None, sim_time=None):
    """
    Evaluate a saved solution with a different random seed.
    
    Args:
        solution_file: Path to saved solution file
        new_seed: New random seed for traffic generation
        n_vehicles: Optional override for number of vehicles
        sim_time: Optional override for simulation time
    
    Returns:
        Dictionary with evaluation results
    """
    # Load solution
    solution_data = load_solution(solution_file)
    if not solution_data:
        return {'success': False, 'error': 'Could not load solution'}
    
    solution = solution_data['solution']
    metadata = solution_data['metadata']
    
    # Use original parameters or overrides
    grid_size = metadata.get('grid_size', 3)
    pattern = metadata.get('pattern', 'balanced')
    n_vehicles = n_vehicles or metadata.get('n_vehicles', 30)
    sim_time = sim_time or metadata.get('sim_time', 600)
    
    print(f"🔄 Re-evaluating solution with new seed {new_seed}")
    
    # Generate new scenario with different seed
    scenario_result = generate_network_and_routes(
        grid_size, n_vehicles, sim_time, pattern, new_seed
    )
    
    if not scenario_result['success']:
        return {'success': False, 'error': 'Scenario generation failed'}
    
    # Apply solution and evaluate
    # (This would integrate with the evaluation system)
    
    return {
        'success': True,
        'original_seed': metadata.get('seed'),
        'new_seed': new_seed,
        'solution': solution,
        'scenario_files': scenario_result['files']
    }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def list_available_patterns():
    """List all available traffic patterns."""
    print("📋 Available Traffic Patterns:")
    print("=" * 40)
    
    for name, config in TRAFFIC_PATTERNS.items():
        print(f"🚦 {name}: {config['description']}")
    
    return list(TRAFFIC_PATTERNS.keys())

if __name__ == "__main__":
    # Example usage
    print("🧪 Testing traffic pattern generation")
    
    result = generate_network_and_routes(
        grid_size=3,
        n_vehicles=20,
        sim_time=600,
        pattern='commuter',
        seed=42
    )
    
    if result['success']:
        print("✅ Test successful!")
        print(f"Files generated: {list(result['files'].keys())}")
    else:
        print(f"❌ Test failed: {result['error']}")
