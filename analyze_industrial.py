#!/usr/bin/env python3
"""
Analyze Industrial Pattern Vehicle Completion Issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import xml.etree.ElementTree as ET
from src.simplified_traffic import create_traffic_scenario
import subprocess

def analyze_industrial_pattern():
    """Analyze what's wrong with the industrial pattern."""
    
    print("🏭 ANALYZING INDUSTRIAL PATTERN VEHICLE COMPLETION")
    print("=" * 70)
    
    # Create industrial scenario
    scenario = create_traffic_scenario(
        grid_size=5,
        n_vehicles=20,
        simulation_time=2400,
        pattern='industrial',
        seed=42
    )
    
    config_file = scenario['config_file']
    route_file = config_file.replace('.sumocfg', '.rou.xml')
    
    # Parse route file to analyze all expected vehicles
    print("\n📋 ANALYZING EXPECTED VEHICLES:")
    expected_vehicles = []
    try:
        tree = ET.parse(route_file)
        root = tree.getroot()
        
        for vehicle in root.findall('vehicle'):
            veh_id = vehicle.get('id')
            depart_time = float(vehicle.get('depart'))
            route_elem = vehicle.find('route')
            route_edges = route_elem.get('edges').split() if route_elem is not None else []
            
            expected_vehicles.append({
                'id': veh_id,
                'depart': depart_time,
                'route_edges': route_edges,
                'route_length': len(route_edges)
            })
        
        # Sort by departure time
        expected_vehicles.sort(key=lambda x: x['depart'])
        
        print(f"Total expected vehicles: {len(expected_vehicles)}")
        print("\nDeparture schedule:")
        for i, veh in enumerate(expected_vehicles):
            time_until_end = 2400 - veh['depart']
            route_str = ' -> '.join(veh['route_edges'][:3] + ['...'] + veh['route_edges'][-2:]) if len(veh['route_edges']) > 5 else ' -> '.join(veh['route_edges'])
            
            print(f"  {veh['id']}: depart={veh['depart']:.0f}s, route_len={veh['route_length']}, time_left={time_until_end:.0f}s")
            print(f"    Route: {route_str}")
            
            # Flag potentially problematic vehicles
            if time_until_end < 300:  # Less than 5 minutes
                print(f"    ⚠️  WARNING: Only {time_until_end:.0f}s left for completion!")
            elif veh['route_length'] > 8:  # Very long route
                print(f"    ⚠️  WARNING: Long route ({veh['route_length']} edges)!")
                
    except Exception as e:
        print(f"❌ Error reading routes: {e}")
        return
    
    # Run basic SUMO simulation
    print(f"\n🚗 RUNNING BASIC SUMO SIMULATION:")
    tripinfo_file = config_file.replace('.sumocfg', '_industrial_analysis.xml')
    
    result = subprocess.run([
        'sumo', '-c', config_file,
        '--tripinfo-output', tripinfo_file,
        '--no-warnings', '--no-step-log',
        '--time-to-teleport', '600',  # Give vehicles extra time
        '--summary-output', tripinfo_file.replace('.xml', '_summary.xml')
    ], capture_output=True, text=True, timeout=300)
    
    # Parse results
    completed_vehicles = []
    try:
        if os.path.exists(tripinfo_file):
            tree = ET.parse(tripinfo_file)
            root = tree.getroot()
            
            for trip in root.findall('tripinfo'):
                veh_id = trip.get('id')
                depart = float(trip.get('depart', '0'))
                arrival = float(trip.get('arrival', '0'))
                duration = float(trip.get('duration', '0'))
                
                completed_vehicles.append({
                    'id': veh_id,
                    'depart': depart,
                    'arrival': arrival,
                    'duration': duration
                })
                
            completed_vehicles.sort(key=lambda x: x['depart'])
            
    except Exception as e:
        print(f"❌ Error parsing tripinfo: {e}")
        return
    
    print(f"\n📊 COMPLETION RESULTS:")
    print(f"Expected: {len(expected_vehicles)} vehicles")
    print(f"Completed: {len(completed_vehicles)} vehicles")
    print(f"Missing: {len(expected_vehicles) - len(completed_vehicles)} vehicles")
    
    # Find which vehicles are missing
    completed_ids = {v['id'] for v in completed_vehicles}
    expected_ids = {v['id'] for v in expected_vehicles}
    missing_ids = expected_ids - completed_ids
    
    print(f"\n❌ MISSING VEHICLES: {len(missing_ids)}")
    for veh_id in sorted(missing_ids):
        # Find the expected vehicle info
        veh_info = next(v for v in expected_vehicles if v['id'] == veh_id)
        time_until_end = 2400 - veh_info['depart']
        
        print(f"   {veh_id}: depart={veh_info['depart']:.0f}s, route_len={veh_info['route_length']}, time_left={time_until_end:.0f}s")
        
        # Check for route issues
        route_edges = veh_info['route_edges']
        if route_edges:
            from_edge = route_edges[0]
            to_edge = route_edges[-1]
            print(f"     Route: {from_edge} -> ... -> {to_edge}")
            
            # Check if route makes sense for industrial pattern
            if not (from_edge.startswith('A') or from_edge.startswith('B') or from_edge.startswith('C') or from_edge.startswith('D') or from_edge.startswith('E')):
                print(f"     ⚠️  Unusual start edge: {from_edge}")
            if not to_edge.endswith('4'):  # Should end at rightmost column (4)
                print(f"     ⚠️  Route doesn't end at rightmost column: {to_edge}")
    
    # Check SUMO errors
    if result.stderr:
        print(f"\n🔍 SUMO STDERR MESSAGES:")
        stderr_lines = result.stderr.strip().split('\n')
        for line in stderr_lines[-15:]:
            if line.strip() and ('error' in line.lower() or 'warning' in line.lower() or 'teleport' in line.lower()):
                print(f"   {line.strip()}")

if __name__ == "__main__":
    analyze_industrial_pattern()
