#!/usr/bin/env python3
"""
Main Entry Point for Traffic Light Optimization

This is the primary entry point for the traffic light optimization system.
It provides a clean interface to the optimization functionality in the src directory.

Usage:
    python main.py              # Run interactive optimization
    python main.py --help       # Show all options
    
Author: Traffic Optimization System
Date: August 2025
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    from optimize import main
    main()
