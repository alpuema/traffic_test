"""
Traffic Light Optimization System

A comprehensive system for traffic light signal optimization using Ant Colony
Optimization (ACO) with advanced traffic pattern generation and analysis.

Author: Alfonso Rato
Date: August 2025
"""

__version__ = "1.0.0"
__author__ = "Alfonso Rato"

# Main imports for easy access
from .config import Config
from .traffic_patterns import TrafficPatternGenerator, create_traffic_pattern_examples

__all__ = [
    'Config',
    'TrafficPatternGenerator', 
    'create_traffic_pattern_examples'
]
