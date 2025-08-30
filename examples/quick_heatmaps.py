#!/usr/bin/env python3
"""
Quick Heatmap Generator - Runs with sensible defaults

This script generates traffic pattern heatmaps without requiring user input.
Uses default settings: 3x3 grid, 100 vehicles, all patterns.
"""

import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_pattern_heatmaps import generate_traffic_pattern_heatmaps

def main():
    """Run heatmap generation with defaults."""
    
    print("🚦 Quick Traffic Pattern Heatmap Generator")
    print("=" * 50)
    print("Using default settings:")
    print("  Grid Size: 4x4")
    print("  Vehicles: 100") 
    print("  Patterns: commuter, industrial, random")
    print()
    
    # Generate heatmaps with defaults
    success = generate_traffic_pattern_heatmaps(
        grid_size=4,
        n_vehicles=100,
        patterns=['commuter', 'industrial', 'random']
    )
    
    if success:
        print("\n✅ Heatmap generation completed successfully!")
        print("💡 The heatmaps show how different traffic patterns distribute vehicle")
        print("   origins (red) and destinations (blue) across the grid.")
        print("💡 File saved as: results/traffic_pattern_heatmaps.png")
    else:
        print("\n❌ Heatmap generation failed.")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
