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
        'description': 'Rush hour commuter pattern - 90% perimeter to 90% center',
        'explanation': 'Vehicles start from outer edges (95% perimeter ONLY) and travel to city center (95% center). Clear commuter flow pattern.',
        'source_weights': {'perimeter_only': 98.0, 'all': 0.1},
        'sink_weights': {'center_only': 98.0, 'all': 0.1},
        'time_distribution': 'early_heavy'
    },
    
    'industrial': {
        'description': 'Industrial corridor pattern - 90% leftmost to 90% rightmost',
        'explanation': 'Industrial traffic: 100% departures from column 0 ONLY, 100% arrivals at rightmost column ONLY. Pure left-to-right flow.',
        'source_weights': {'column_0_only': 100.0},
        'sink_weights': {'rightmost_column_only': 100.0}, 
        'time_distribution': 'shift_based'
    },
    
    'random': {
        'description': 'Completely random vehicle origins and destinations',
        'explanation': 'Pure random selection - useful for testing and comparison. No spatial patterns or correlations.',
        'source_weights': {'all': 10.0},
        'sink_weights': {'all': 10.0},
        'time_distribution': 'uniform'
    }
}

# ============================================================================
# CORE TRAFFIC GENERATION FUNCTIONS
# ============================================================================

def generate_network_and_routes(grid_size, n_vehicles, sim_time, pattern='commuter', seed=None, output_dir=None):
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
        script_dir = os.path.dirname(os.path.abspath(__file__))  # src/
        project_root = os.path.dirname(script_dir)  # my_grid_simulation/
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
        pattern_config = TRAFFIC_PATTERNS.get(pattern, TRAFFIC_PATTERNS['random'])
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


def create_traffic_scenario(grid_size, n_vehicles, simulation_time, pattern='commuter', seed=None):
    """
    Compatibility wrapper used by examples.

    Returns a dict including 'config_file' pointing to the generated .sumocfg file.
    """
    result = generate_network_and_routes(
        grid_size=grid_size,
        n_vehicles=n_vehicles,
        sim_time=simulation_time,
        pattern=pattern,
        seed=seed,
    )
    if not result.get('success'):
        raise RuntimeError(result.get('error', 'Failed to create traffic scenario'))
    files = result.get('files', {})
    return {
        'config_file': files.get('config'),
        'files': files,
        'pattern': result.get('pattern'),
        'seed': result.get('seed'),
        'grid_size': result.get('grid_size'),
        'n_vehicles': result.get('n_vehicles'),
        'simulation_time': result.get('sim_time'),
    }

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
    """Categorize edges by their location and function for pattern generation."""
    categories = {
        'all': edges,
        'perimeter': [],
        'center': [],
        'perimeter_only': [],      # Edges that ONLY touch perimeter (for strict commuter pattern)
        'center_only': [],         # Edges that are ONLY internal (for strict commuter pattern)
        'column_0_only': [],       # Edges that start ONLY from column 0 (for strict industrial)
        'rightmost_column_only': [], # Edges that end ONLY at rightmost column (for strict industrial)
        'left_edge': [],
        'right_edge': [],
        'leftmost_edge': [],    # Edges involving leftmost column
        'rightmost_edge': [],   # Edges involving rightmost column
        'leftmost_column': [],  # Edges strictly from column 0 only
        'rightmost_column': [], # Edges strictly to rightmost column only
        'top_edge': [],
        'bottom_edge': [],
        'outer_ring': [],
        'inner_core': []
    }
    
    # Detect grid size by finding max coordinates in edges
    max_row_idx = 0
    max_col_idx = 0
    
    for edge in edges:
        if len(edge) == 4:
            start_row, start_col, end_row, end_col = edge[0], edge[1], edge[2], edge[3]
            if start_row.isalpha() and start_col.isdigit() and end_row.isalpha() and end_col.isdigit():
                start_row_idx = ord(start_row) - ord('A')
                end_row_idx = ord(end_row) - ord('A')
                start_col_idx = int(start_col)
                end_col_idx = int(end_col)
                
                max_row_idx = max(max_row_idx, start_row_idx, end_row_idx)
                max_col_idx = max(max_col_idx, start_col_idx, end_col_idx)
    
    for edge in edges:
        # SUMO grid edges follow pattern like "A0B0", "B1C1", etc.
        # Format: [StartRow][StartCol][EndRow][EndCol]
        # A=top, B=middle, C=bottom (rows)
        # 0=left, 1=middle, 2=right (cols)
        
        if len(edge) == 4:
            start_row, start_col, end_row, end_col = edge[0], edge[1], edge[2], edge[3]
            
            # Convert to coordinates for easier analysis
            # A=0, B=1, C=2, D=3... (row indices)
            # 0, 1, 2, 3... (column indices)
            start_row_idx = ord(start_row) - ord('A') if start_row.isalpha() else 0
            end_row_idx = ord(end_row) - ord('A') if end_row.isalpha() else 0
            start_col_idx = int(start_col) if start_col.isdigit() else 0
            end_col_idx = int(end_col) if end_col.isdigit() else 0
            
            # Determine edge type based on start and end positions
            
            # STRICT COLUMN 0 ONLY: Edges that start from column 0 AND don't go to inner areas
            if start_col_idx == 0:
                categories['column_0_only'].append(edge)
                categories['leftmost_column'].append(edge)
                categories['leftmost_edge'].append(edge)
                categories['left_edge'].append(edge)
            
            # STRICT RIGHTMOST COLUMN ONLY: Edges that end at the rightmost column AND come from appropriate sources
            if end_col_idx == max_col_idx:
                categories['rightmost_column_only'].append(edge)
                categories['rightmost_column'].append(edge)
                categories['rightmost_edge'].append(edge)
                categories['right_edge'].append(edge)
            
            # LEFT EDGES: Any edge involving leftmost column (broader definition)
            if start_col_idx == 0 or end_col_idx == 0:
                if edge not in categories['left_edge']:
                    categories['left_edge'].append(edge)
            
            # RIGHT EDGES: Any edge involving rightmost column (broader definition)  
            if start_col_idx == max_col_idx or end_col_idx == max_col_idx:
                if edge not in categories['right_edge']:
                    categories['right_edge'].append(edge)
            
            # TOP EDGES: Start from or go to top row (A)
            if start_row_idx == 0 or end_row_idx == 0:
                categories['top_edge'].append(edge)
            
            # BOTTOM EDGES: Start from or go to bottom row
            if start_row_idx == max_row_idx or end_row_idx == max_row_idx:
                categories['bottom_edge'].append(edge)
            
            # PERIMETER EDGES: Any edge touching the grid boundary
            is_perimeter = (start_row_idx == 0 or start_row_idx == max_row_idx or 
                           start_col_idx == 0 or start_col_idx == max_col_idx or
                           end_row_idx == 0 or end_row_idx == max_row_idx or 
                           end_col_idx == 0 or end_col_idx == max_col_idx)
                           
            # STRICT PERIMETER ONLY: Edges that START from boundary (for origins)
            is_boundary_start = (start_row_idx == 0 or start_row_idx == max_row_idx or 
                                start_col_idx == 0 or start_col_idx == max_col_idx)
            is_boundary_end = (end_row_idx == 0 or end_row_idx == max_row_idx or 
                              end_col_idx == 0 or end_col_idx == max_col_idx)
            
            if is_perimeter:
                categories['perimeter'].append(edge)
                categories['outer_ring'].append(edge)
            
            # Add to perimeter_only ONLY if it starts from boundary (for commuter origins)
            if is_boundary_start:
                categories['perimeter_only'].append(edge)
            
            # STRICT CENTER ONLY: Different rules for origins vs destinations
            start_is_internal = (0 < start_row_idx < max_row_idx and 0 < start_col_idx < max_col_idx)
            end_is_internal = (0 < end_row_idx < max_row_idx and 0 < end_col_idx < max_col_idx)
            
            # Center category (broad)
            if start_is_internal and end_is_internal:
                categories['center'].append(edge)
                categories['inner_core'].append(edge)
            elif start_is_internal or end_is_internal:
                categories['center'].append(edge)
                
            # Center_only category (strict - for destinations, edges that END in center)
            if end_is_internal:
                categories['center_only'].append(edge)
    
    # Ensure no empty categories - use fallback distributions
    total_edges = len(edges)
    if total_edges > 0:
        # Strict fallbacks for new categories
        if not categories['perimeter_only']:
            # Fallback: edges that start from boundary rows/columns only
            for edge in edges:
                if len(edge) == 4 and edge[0].isalpha() and edge[1].isdigit():
                    start_row_idx = ord(edge[0]) - ord('A')
                    start_col_idx = int(edge[1])
                    if (start_row_idx == 0 or start_row_idx == max_row_idx or 
                        start_col_idx == 0 or start_col_idx == max_col_idx):
                        categories['perimeter_only'].append(edge)
            
        if not categories['center_only']:
            # Fallback: edges that end in internal positions
            for edge in edges:
                if len(edge) == 4 and edge[2].isalpha() and edge[3].isdigit():
                    end_row_idx = ord(edge[2]) - ord('A')
                    end_col_idx = int(edge[3])
                    if (0 < end_row_idx < max_row_idx and 0 < end_col_idx < max_col_idx):
                        categories['center_only'].append(edge)
                        
        if not categories['column_0_only']:
            # Fallback: edges that start with column 0
            categories['column_0_only'] = [e for e in edges if len(e) >= 2 and e[1] == '0'][:max(1, total_edges//4)]
            
        if not categories['rightmost_column_only']:
            # Fallback: edges that end with max column index
            max_col_str = str(max_col_idx)
            categories['rightmost_column_only'] = [e for e in edges if len(e) >= 4 and e[3] == max_col_str][:max(1, total_edges//4)]
            
        # Existing fallbacks
        if not categories['leftmost_column']:
            categories['leftmost_column'] = [e for e in edges if len(e) >= 2 and e[1] == '0'][:max(1, total_edges//4)]
            
        if not categories['rightmost_column']:
            max_col_str = str(max_col_idx)
            categories['rightmost_column'] = [e for e in edges if len(e) >= 4 and e[3] == max_col_str][:max(1, total_edges//4)]
            
        if not categories['leftmost_edge']:
            categories['leftmost_edge'] = [e for e in edges if len(e) >= 2 and e[1] == '0'][:max(1, total_edges//4)]
            
        if not categories['rightmost_edge']:
            max_col_str = str(max_col_idx)
            categories['rightmost_edge'] = [e for e in edges if len(e) >= 4 and e[3] == max_col_str][:max(1, total_edges//4)]
            
        if not categories['center']:
            mid_start = total_edges // 3
            mid_end = 2 * total_edges // 3
            categories['center'] = edges[mid_start:mid_end] if mid_end > mid_start else [edges[total_edges//2]]
            
        if not categories['perimeter']:
            categories['perimeter'] = []
            for edge in edges:
                if len(edge) == 4 and edge[0].isalpha() and edge[1].isdigit():
                    start_row_idx = ord(edge[0]) - ord('A')
                    start_col_idx = int(edge[1])
                    end_row_idx = ord(edge[2]) - ord('A') 
                    end_col_idx = int(edge[3])
                    
                    if (start_row_idx == 0 or start_row_idx == max_row_idx or start_col_idx == 0 or start_col_idx == max_col_idx or
                        end_row_idx == 0 or end_row_idx == max_row_idx or end_col_idx == 0 or end_col_idx == max_col_idx):
                        categories['perimeter'].append(edge)
            
            if not categories['perimeter']:  # Ultimate fallback
                categories['perimeter'] = edges[:total_edges//2] + edges[-total_edges//2:]
    
    # Final fallback - ensure no completely empty categories
    for category in categories:
        if category != 'all' and not categories[category]:
            categories[category] = edges.copy()
    
    return categories

def select_weighted_edge(edge_categories, weights):
    """Select an edge based on weighted categories with exclusive selection."""
    # Create weighted pool of edges with exclusive selection
    weighted_edges = []
    
    # Process weights to handle exclusive categories
    for category, weight in weights.items():
        if category in edge_categories and weight > 0:
            category_edges = edge_categories[category]
            
            # For exclusive patterns, remove edges that might be in overlapping categories
            if category == 'perimeter_only':
                # Ensure edges are ONLY from perimeter, not also in center categories
                exclusive_edges = [e for e in category_edges 
                                 if e not in edge_categories.get('center_only', []) 
                                 and e not in edge_categories.get('center', [])]
                if exclusive_edges:
                    category_edges = exclusive_edges
                    
            elif category == 'center_only':
                # Ensure edges are ONLY for center, not also in perimeter origin categories  
                exclusive_edges = category_edges  # center_only is already exclusive for destinations
                
            elif category == 'column_0_only':
                # Ensure edges are ONLY from column 0
                exclusive_edges = [e for e in category_edges 
                                 if len(e) >= 2 and e[1] == '0']
                if exclusive_edges:
                    category_edges = exclusive_edges
                    
            elif category == 'rightmost_column_only':
                # Already exclusive by definition
                exclusive_edges = category_edges
            
            # Add edges multiple times based on weight
            for edge in category_edges:
                weighted_edges.extend([edge] * int(weight * 10))
    
    if not weighted_edges:
        # For exclusive patterns, avoid falling back to 'all'
        # Try to find any available edges from exclusive categories first
        for category in ['column_0_only', 'rightmost_column_only', 'perimeter_only', 'center_only']:
            if category in edge_categories and edge_categories[category]:
                weighted_edges = edge_categories[category]
                break
        
        # If still no edges, then fall back to 'all'
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
            # Sort route file by departure time (SUMO requires this)
            sort_route_file_by_departure_time(route_file)
            print(f"   ✅ Routes generated: {route_file}")
            return True
        else:
            print(f"   ❌ Route generation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error converting trips to routes: {e}")
        return False

def sort_route_file_by_departure_time(route_file):
    """Sort vehicles in route file by departure time (required by SUMO)."""
    try:
        tree = ET.parse(route_file)
        root = tree.getroot()
        
        # Extract all vehicle elements with their departure times
        vehicles = []
        for vehicle in root.findall('vehicle'):
            depart_time = float(vehicle.get('depart', '0'))
            vehicles.append((depart_time, vehicle))
        
        # Sort by departure time
        vehicles.sort(key=lambda x: x[0])
        
        # Clear existing vehicles and add them back in sorted order
        for vehicle in root.findall('vehicle'):
            root.remove(vehicle)
        
        for _, vehicle in vehicles:
            root.append(vehicle)
        
        # Write sorted file
        tree.write(route_file, xml_declaration=True, encoding='UTF-8')
        print(f"   ✅ Route file sorted by departure time")
        
    except Exception as e:
        print(f"   ⚠️  Warning: Could not sort route file: {e}")
        # Don't fail the whole process for this

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
        Dictionary with evaluation results including performance metrics
    """
    # Load solution
    solution_data = load_solution(solution_file)
    if not solution_data:
        return {'success': False, 'error': 'Could not load solution'}
    
    solution = solution_data['solution']
    metadata = solution_data['metadata']
    
    # Use original parameters or overrides
    grid_size = metadata.get('grid_size', 3)
    pattern = metadata.get('pattern', 'random')
    n_vehicles = n_vehicles or metadata.get('n_vehicles', 30)
    sim_time = sim_time or metadata.get('sim_time', 600)
    
    print(f"🔄 Re-evaluating solution with new seed {new_seed}")
    
    # Generate new scenario with different seed
    scenario_result = generate_network_and_routes(
        grid_size, n_vehicles, sim_time, pattern, new_seed
    )
    
    if not scenario_result['success']:
        return {'success': False, 'error': 'Scenario generation failed'}
    
    # Now actually evaluate the performance using SUMO simulation
    try:
        # Import the evaluation function from ACO module
        from .optimization.simple_aco import evaluate_solution, calculate_cost
        import tempfile
        import os
        
        # Create temporary directory for simulation
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set simulation time in the ACO module's global variable
            import src.optimization.simple_aco as aco_module
            original_sim_time = aco_module.SIMULATION_TIME
            aco_module.SIMULATION_TIME = sim_time
            
            try:
                # Run evaluation
                metrics = evaluate_solution(
                    solution=solution,
                    net_file=scenario_result['files']['network'],
                    route_file=scenario_result['files']['routes'],
                    temp_dir=temp_dir
                )
                
                # Calculate cost using the same function as optimization
                cost = calculate_cost(metrics)
                
                # Average travel time per vehicle
                avg_time = metrics.get('total_time', 0) / max(metrics.get('vehicles', 1), 1)
                
                print(f"   ✅ Evaluation completed:")
                print(f"      • Vehicles: {metrics.get('vehicles', 0)}/{n_vehicles}")
                print(f"      • Avg travel time: {avg_time:.1f}s")
                print(f"      • Cost: {cost:.1f}")
                
                return {
                    'success': True,
                    'original_seed': metadata.get('seed'),
                    'new_seed': new_seed,
                    'solution': solution,
                    'scenario_files': scenario_result['files'],
                    'metrics': metrics,
                    'cost': cost,
                    'avg_travel_time': avg_time
                }
            finally:
                # Restore original simulation time
                aco_module.SIMULATION_TIME = original_sim_time
            
    except Exception as e:
        print(f"   ❌ Evaluation failed: {e}")
        return {
            'success': False,
            'error': f'Simulation evaluation failed: {e}',
            'original_seed': metadata.get('seed'),
            'new_seed': new_seed,
            'scenario_generated': True
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
        grid_size=4,
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
