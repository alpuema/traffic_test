#!/usr/bin/env python3
"""
Analyze which specific vehicles are not completing their routes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import xml.etree.ElementTree as ET
from src.simplified_traffic import create_traffic_scenario
import subprocess

def analyze_missing_vehicles():
    """Analyze which vehicles are missing and why."""
    
    print("🔍 ANALYZING MISSING VEHICLES")
    print("=" * 50)
    
    # Create scenario
    scenario = create_traffic_scenario(
        grid_size=5,
        n_vehicles=20,
        simulation_time=2400,
        pattern='commuter',
        seed=42
    )
    
    # Run simple SUMO simulation
    config_file = scenario['config_file']
    tripinfo_file = config_file.replace('.sumocfg', '_analysis_tripinfo.xml')
    
    result = subprocess.run([
        'sumo', '-c', config_file,
        '--tripinfo-output', tripinfo_file,
        '--no-warnings', '--no-step-log',
        '--time-to-teleport', '600',
        '--summary-output', tripinfo_file.replace('_tripinfo.xml', '_summary.xml')
    ], capture_output=True, text=True, timeout=300)
    
    # Parse route file to see all expected vehicles
    route_file = scenario.get('routes') or config_file.replace('.sumocfg', '.rou.xml')
    
    print("\n📋 EXPECTED VEHICLES (from route file):")
    expected_vehicles = set()
    try:
        tree = ET.parse(route_file)
        root = tree.getroot()
        
        for vehicle in root.findall('vehicle'):
            veh_id = vehicle.get('id')
            depart_time = vehicle.get('depart')
            route_edges = vehicle.find('route').get('edges').split()
            expected_vehicles.add(veh_id)
            
            print(f"   {veh_id}: depart={depart_time}s, route_length={len(route_edges)} edges")
            if len(route_edges) > 8:  # Flag very long routes
                print(f"      ⚠️  LONG ROUTE: {' -> '.join(route_edges[:3])} ... {' -> '.join(route_edges[-3:])}")
                
    except Exception as e:
        print(f"❌ Error reading route file: {e}")
        return
    
    print(f"\n📊 Total expected vehicles: {len(expected_vehicles)}")
    
    # Parse tripinfo to see which completed
    print("\n📋 COMPLETED VEHICLES (from tripinfo):")
    completed_vehicles = set()
    try:
        if os.path.exists(tripinfo_file):
            tree = ET.parse(tripinfo_file)
            root = tree.getroot()
            
            for trip in root.findall('tripinfo'):
                veh_id = trip.get('id')
                depart = trip.get('depart')
                arrival = trip.get('arrival')
                duration = trip.get('duration')
                route_length = trip.get('routeLength', '0')
                
                completed_vehicles.add(veh_id)
                print(f"   {veh_id}: depart={depart}s, arrive={arrival}s, duration={duration}s, route_length={route_length}m")
                
        else:
            print("❌ Tripinfo file not found")
            
    except Exception as e:
        print(f"❌ Error reading tripinfo: {e}")
        return
    
    # Find missing vehicles
    missing_vehicles = expected_vehicles - completed_vehicles
    
    print(f"\n❌ MISSING VEHICLES: {len(missing_vehicles)}")
    for veh_id in sorted(missing_vehicles):
        print(f"   {veh_id}")
    
    # Check SUMO summary for teleports
    summary_file = tripinfo_file.replace('_tripinfo.xml', '_summary.xml')
    if os.path.exists(summary_file):
        print(f"\n📊 SUMO SUMMARY STATISTICS:")
        try:
            tree = ET.parse(summary_file)
            root = tree.getroot()
            
            # Look for vehicle statistics
            for step in root.findall('step'):
                time_attr = step.get('time')
                if time_attr and float(time_attr) == 2400:  # Final step
                    vehicles = step.find('vehicles')
                    if vehicles is not None:
                        loaded = vehicles.get('loaded', '0')
                        inserted = vehicles.get('inserted', '0') 
                        running = vehicles.get('running', '0')
                        waiting = vehicles.get('waiting', '0')
                        teleports = vehicles.get('teleports', '0')
                        
                        print(f"   At simulation end (t=2400s):")
                        print(f"   Loaded: {loaded}, Inserted: {inserted}, Running: {running}")
                        print(f"   Waiting: {waiting}, Teleports: {teleports}")
                        break
                        
        except Exception as e:
            print(f"❌ Error reading summary: {e}")
    
    # Show SUMO stderr for additional clues
    if result.stderr:
        print(f"\n🔍 SUMO MESSAGES:")
        stderr_lines = result.stderr.strip().split('\n')
        for line in stderr_lines[-20:]:  # Last 20 lines
            if line.strip() and ('teleport' in line.lower() or 'error' in line.lower() or 'warning' in line.lower()):
                print(f"   {line.strip()}")

if __name__ == "__main__":
    analyze_missing_vehicles()
