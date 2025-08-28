#!/usr/bin/env python3
"""
SUMO Default Analysis Tool

Analyzes what SUMO's default traffic light timings are and why they're hard to beat.

Author: Alfonso Rato  
Date: August 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import Config

def analyze_sumo_defaults():
    """Analyze SUMO's default traffic light settings."""
    
    print("🔍 SUMO Default Traffic Light Analysis")
    print("=" * 60)
    
    try:
        # Configure for analysis
        config = Config()
        config.grid_size = 3  # Standard grid for analysis
        config.verbose = 2
        
        print(f"📋 Analyzing {config.grid_size}x{config.grid_size} grid network...")
        
        # Import ACO to get network generation
        import src.optimization.simple_aco as aco
        
        # Generate a test network
        paths = aco.get_file_paths()
        os.makedirs(paths['temp_dir'], exist_ok=True)
        
        test_net = os.path.join(paths['temp_dir'], 'analysis_net.xml')
        
        print("🏗️  Generating test network...")
        if aco.generate_grid_network(config.grid_size, test_net):
            
            # Get default durations
            default_durations = aco.get_default_durations(test_net)
            phase_types = aco.analyze_phase_types(test_net)
            
            print(f"\n📊 Analysis Results:")
            print(f"   Total phases found: {len(default_durations)}")
            print(f"   Default durations: {default_durations}")
            
            print(f"\n🚦 Phase Type Analysis:")
            green_red_phases = []
            yellow_phases = []
            
            for i, (duration, is_green_red) in enumerate(zip(default_durations, phase_types)):
                phase_type = "Green/Red" if is_green_red else "Yellow"
                if is_green_red:
                    green_red_phases.append(duration)
                else:
                    yellow_phases.append(duration)
                print(f"   Phase {i+1}: {duration}s ({phase_type})")
            
            print(f"\n📈 Statistical Summary:")
            if green_red_phases:
                print(f"   Green/Red phases: {len(green_red_phases)} phases")
                print(f"     Range: {min(green_red_phases)}-{max(green_red_phases)} seconds")
                print(f"     Average: {sum(green_red_phases)/len(green_red_phases):.1f} seconds")
                print(f"     Values: {sorted(set(green_red_phases))}")
                
            if yellow_phases:
                print(f"   Yellow phases: {len(yellow_phases)} phases")
                print(f"     Range: {min(yellow_phases)}-{max(yellow_phases)} seconds")
                print(f"     Average: {sum(yellow_phases)/len(yellow_phases):.1f} seconds")
                print(f"     Values: {sorted(set(yellow_phases))}")
            
            print(f"\n💡 Traffic Engineering Insights:")
            
            # Analyze against Simple ACO search space
            from src.optimization.simple_aco import (
                GREEN_MIN_DURATION, GREEN_MAX_DURATION, 
                YELLOW_MIN_DURATION, YELLOW_MAX_DURATION
            )
            
            print(f"   SUMO defaults vs Simple ACO search space:")
            print(f"     Green phase range: {GREEN_MIN_DURATION}-{GREEN_MAX_DURATION}s")
            print(f"     Yellow phase range: {YELLOW_MIN_DURATION}-{YELLOW_MAX_DURATION}s")
            
            # Check if defaults are already in optimal ranges
            green_in_range = all(GREEN_MIN_DURATION <= d <= GREEN_MAX_DURATION 
                               for d in green_red_phases)
            yellow_in_range = all(YELLOW_MIN_DURATION <= d <= YELLOW_MAX_DURATION 
                                for d in yellow_phases)
            
            print(f"\n🎯 Optimization Potential:")
            if green_in_range and yellow_in_range:
                print(f"   ✅ SUMO defaults are ALREADY within optimal ranges!")
                print(f"   ✅ Green phases: All within {GREEN_MIN_DURATION}-{GREEN_MAX_DURATION}s")
                print(f"   ✅ Yellow phases: All within {YELLOW_MIN_DURATION}-{YELLOW_MAX_DURATION}s")
                print(f"   💡 This explains why ACO struggles to improve on defaults.")
            else:
                print(f"   ⚠️  Some phases outside optimal ranges - optimization possible")
                
            print(f"\n🔧 Recommendations:")
            print(f"   1. Simple ACO uses direct range sampling (no complex bins)")
            print(f"   2. Traffic engineering constraints built-in")
            print(f"   3. Lower evaporation rate (0.1-0.3) to preserve good solutions")
            print(f"   4. Focus on small improvements rather than dramatic changes")
            print(f"   5. Consider that SUMO defaults might already be near-optimal!")
            
        else:
            print("❌ Failed to generate test network")
            
        # Cleanup
        if os.path.exists(test_net):
            os.remove(test_net)
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_sumo_defaults()
