#!/usr/bin/env python3
"""
Debug Vehicle Completion Issue

This script will help us understand why only 18/20 vehicles are completing
their routes regardless of simulation time.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.simplified_traffic import create_traffic_scenario
import subprocess
import xml.etree.ElementTree as ET

def debug_vehicle_completion():
    """Debug vehicle completion issue with different simulation times."""
    
    # Test with progressively longer simulation times
    test_times = [1500, 2400, 3600, 4800]  # 25, 40, 60, 80 minutes
    
    for sim_time in test_times:
        print(f"\n{'='*60}")
        print(f"TESTING WITH {sim_time} SECONDS ({sim_time/60:.1f} MINUTES)")
        print(f"{'='*60}")
        
        # Create traffic scenario
        scenario = create_traffic_scenario(
            grid_size=5,
            n_vehicles=20,
            simulation_time=sim_time,
            pattern='commuter',
            seed=42
        )
        
        if not scenario:
            print("❌ Failed to create scenario")
            continue
            
        config_file = scenario['config_file']
        print(f"✅ Created scenario: {config_file}")
        
        # Run SUMO simulation
        tripinfo_file = config_file.replace('.sumocfg', '_debug_tripinfo.xml')
        
        try:
            result = subprocess.run([
                'sumo', '-c', config_file,
                '--tripinfo-output', tripinfo_file,
                '--no-warnings', '--no-step-log',
                '--time-to-teleport', '600',  # Give vehicles more time before teleporting
                '--ignore-route-errors'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"❌ SUMO failed: {result.stderr[:200]}")
                continue
                
            # Parse tripinfo
            completed_vehicles, total_expected = parse_tripinfo(tripinfo_file)
            print(f"📊 Results: {completed_vehicles}/{total_expected} vehicles completed")
            
            if result.stderr:
                stderr_lines = result.stderr.strip().split('\n')[-10:]  # Last 10 lines
                print(f"🔍 SUMO messages (last 10 lines):")
                for line in stderr_lines:
                    if line.strip():
                        print(f"   {line.strip()}")
            
            # Check route file for departure times
            route_file = scenario.get('routes') or config_file.replace('.sumocfg', '.rou.xml')
            if os.path.exists(route_file):
                analyze_departure_times(route_file, sim_time)
            
            # If we got all vehicles, we found the minimum time needed
            if completed_vehicles >= 20:
                print(f"🎉 SUCCESS! All vehicles completed with {sim_time} seconds")
                break
                
        except Exception as e:
            print(f"❌ Error running simulation: {e}")
            continue

def parse_tripinfo(tripinfo_file):
    """Parse tripinfo file to count completed vehicles."""
    try:
        if not os.path.exists(tripinfo_file):
            print(f"⚠️  Tripinfo file not found: {tripinfo_file}")
            return 0, 20
            
        tree = ET.parse(tripinfo_file)
        root = tree.getroot()
        
        completed_vehicles = len(root.findall('tripinfo'))
        print(f"📄 Tripinfo file contains {completed_vehicles} completed trips")
        
        # Show details of first few and last few completed vehicles
        trips = root.findall('tripinfo')
        if trips:
            print("🚗 First 3 completed vehicles:")
            for trip in trips[:3]:
                veh_id = trip.get('id', 'unknown')
                depart = trip.get('depart', '0')
                arrival = trip.get('arrival', '0')
                duration = trip.get('duration', '0')
                print(f"   {veh_id}: depart={depart}s, arrive={arrival}s, duration={duration}s")
            
            if len(trips) > 3:
                print("🚗 Last 3 completed vehicles:")
                for trip in trips[-3:]:
                    veh_id = trip.get('id', 'unknown')
                    depart = trip.get('depart', '0')
                    arrival = trip.get('arrival', '0') 
                    duration = trip.get('duration', '0')
                    print(f"   {veh_id}: depart={depart}s, arrive={arrival}s, duration={duration}s")
        
        return completed_vehicles, 20
        
    except Exception as e:
        print(f"⚠️  Error parsing tripinfo: {e}")
        return 0, 20

def analyze_departure_times(route_file, sim_time):
    """Analyze departure times in route file."""
    try:
        tree = ET.parse(route_file)
        root = tree.getroot()
        
        departures = []
        for vehicle in root.findall('vehicle'):
            depart_time = float(vehicle.get('depart', '0'))
            veh_id = vehicle.get('id', 'unknown')
            departures.append((depart_time, veh_id))
        
        departures.sort()
        
        if departures:
            earliest = departures[0][0]
            latest = departures[-1][0]
            buffer_time = sim_time - latest
            
            print(f"⏰ Departure times: earliest={earliest}s, latest={latest}s")
            print(f"⏰ Buffer time for latest vehicle: {buffer_time}s ({buffer_time/60:.1f} minutes)")
            
            if buffer_time < 300:  # Less than 5 minutes
                print(f"⚠️  WARNING: Only {buffer_time}s buffer may not be enough!")
            
            print(f"⏰ Latest departing vehicles:")
            for depart_time, veh_id in departures[-5:]:
                print(f"   {veh_id}: departs at {depart_time}s")
                
    except Exception as e:
        print(f"⚠️  Error analyzing departures: {e}")

if __name__ == "__main__":
    debug_vehicle_completion()
