"""
Configuration module for the ACO Traffic Light Optimization system.

This module centralizes all configuration settings and ensures consistent
directory structure across the entire project.

Author: Alfonso Rato  
Date: August 2025
"""

import os
from datetime import datetime
from typing import Dict, List, Any


class Config:
    """Main configuration class for the ACO optimization system."""
    
    def __init__(self):
        # Project structure
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_dir = os.path.join(self.project_root, 'results')
        self.sumo_data_dir = os.path.join(self.project_root, 'sumo_data')
        self.temp_dir = os.path.join(self.results_dir, 'temp')
        
        # Ensure results directory structure
        self._ensure_directories()
        
        # Run identification
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Default ACO parameters  
        self.n_ants = 20
        self.n_iterations = 5
        self.duration_bins = list(range(10, 61, 2))  # 10-60 seconds, 2s steps
        self.evaporation = 0.5
        self.pheromone_init = 1.0
        self.alpha = 30.0  # Weight for stop time penalty
        self.beta = 1000.0  # Penalty weight for missing vehicles
        
        # Simulation parameters
        self.grid_size = 3
        self.n_vehicles = 200
        self.simulation_time = 1800  # 30 minutes
        
        # Traffic pattern configuration
        self.traffic_pattern = "realistic"  # Options: "random", "realistic", "commuter", "commercial", "custom"
        self.custom_sources = None  # Custom source weights: {edge_pattern: weight, ...}
        self.custom_sinks = None    # Custom sink weights: {edge_pattern: weight, ...}
        
        # Predefined traffic patterns
        self.traffic_patterns = {
            "random": {
                "description": "Random uniform distribution (original behavior)",
                "sources": "uniform",
                "sinks": "uniform"
            },
            "realistic": {
                "description": "Realistic urban pattern with popular directions",
                "sources": {"*_0": 2.0, "*_1": 1.5, "*_2": 1.0},  # Prefer certain directions
                "sinks": {"*_0": 2.0, "*_1": 1.5, "*_2": 1.0},
                "od_flows": [  # Origin-Destination flows
                    {"from": "north", "to": "south", "weight": 3.0},
                    {"from": "west", "to": "east", "weight": 2.5},
                    {"from": "east", "to": "center", "weight": 2.0}
                ]
            },
            "commuter": {
                "description": "Rush hour pattern - suburban to city center",
                "sources": {"perimeter": 3.0, "center": 0.5},
                "sinks": {"center": 4.0, "perimeter": 1.0},
                "peak_hours": [{"start": 0.2, "end": 0.4, "multiplier": 2.0}]  # 20-40% of simulation
            },
            "commercial": {
                "description": "Commercial district pattern - distributed destinations",
                "sources": {"perimeter": 2.0, "center": 1.0}, 
                "sinks": "uniform",
                "vehicle_mix": {"car": 0.7, "truck": 0.3}
            }
        }
        
        # Display and output settings
        self.show_progress = True
        self.show_plots = True
        self.launch_sumo_gui = False
        self.verbosity = 1  # 0=silent, 1=summary, 2=full detail
        
    def _ensure_directories(self):
        """Create necessary directory structure."""
        dirs_to_create = [
            self.results_dir,
            os.path.join(self.results_dir, 'optimization'),
            os.path.join(self.results_dir, 'sensitivity_analysis'), 
            os.path.join(self.results_dir, 'diagnosis'),
            os.path.join(self.results_dir, 'plots'),
            os.path.join(self.results_dir, 'temp'),
            os.path.join(self.results_dir, 'final_solutions')
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_output_dir(self, analysis_type: str) -> str:
        """Get the appropriate output directory for different analysis types."""
        type_mapping = {
            'optimization': 'optimization',
            'sensitivity': 'sensitivity_analysis', 
            'diagnosis': 'diagnosis',
            'plots': 'plots',
            'temp': 'temp',
            'final': 'final_solutions'
        }
        
        subdir = type_mapping.get(analysis_type, analysis_type)
        return os.path.join(self.results_dir, subdir)
    
    def get_temp_dir(self, unique_name: str = None) -> str:
        """Get a temporary directory with optional unique naming."""
        if unique_name:
            temp_path = os.path.join(self.temp_dir, f"{self.run_id}_{unique_name}")
        else:
            temp_path = os.path.join(self.temp_dir, self.run_id)
        
        os.makedirs(temp_path, exist_ok=True)
        return temp_path
    
    def cleanup_temp_files(self, keep_latest: int = 3):
        """Clean up old temporary files, keeping only the latest N runs."""
        if not os.path.exists(self.temp_dir):
            return
            
        # List all temp directories
        temp_dirs = []
        for item in os.listdir(self.temp_dir):
            item_path = os.path.join(self.temp_dir, item)
            if os.path.isdir(item_path):
                temp_dirs.append((item, os.path.getctime(item_path)))
        
        # Sort by creation time (newest first)
        temp_dirs.sort(key=lambda x: x[1], reverse=True)
        
        # Remove old directories
        for dir_name, _ in temp_dirs[keep_latest:]:
            dir_path = os.path.join(self.temp_dir, dir_name)
            try:
                import shutil
                shutil.rmtree(dir_path)
                print(f"🧹 Cleaned up old temp directory: {dir_name}")
            except Exception as e:
                print(f"⚠️ Could not remove temp directory {dir_name}: {e}")
    
    def set_traffic_pattern(self, pattern_name: str):
        """Set the traffic pattern for vehicle generation."""
        if pattern_name in self.traffic_patterns or pattern_name == "custom":
            self.traffic_pattern = pattern_name
            if pattern_name == "custom":
                print(f"🚦 Traffic pattern set to: {pattern_name}")
                print(f"   User-defined custom pattern (configure with set_custom_traffic_sources)")
            else:
                print(f"🚦 Traffic pattern set to: {pattern_name}")
                print(f"   {self.traffic_patterns[pattern_name]['description']}")
        else:
            available = list(self.traffic_patterns.keys()) + ["custom"]
            print(f"❌ Unknown traffic pattern: {pattern_name}")
            print(f"   Available patterns: {', '.join(available)}")
    
    def set_custom_traffic_sources(self, sources: Dict[str, float], sinks: Dict[str, float] = None):
        """Set custom traffic sources and sinks with weights.
        
        Args:
            sources: Dictionary of {edge_pattern: weight} for vehicle origins
            sinks: Dictionary of {edge_pattern: weight} for vehicle destinations
                  If None, uses same as sources
        
        Example:
            config.set_custom_traffic_sources({
                "top_*": 3.0,     # High traffic from top edges
                "bottom_*": 1.0,  # Low traffic from bottom edges  
                "*center*": 2.0   # Medium traffic from center areas
            })
        """
        self.traffic_pattern = "custom"
        self.custom_sources = sources
        self.custom_sinks = sinks or sources
        print(f"🎯 Custom traffic pattern configured")
        print(f"   Sources: {sources}")
        print(f"   Sinks: {self.custom_sinks}")
    
    def get_traffic_pattern_info(self) -> Dict[str, Any]:
        """Get information about the current traffic pattern."""
        if self.traffic_pattern == "custom":
            return {
                "name": "custom",
                "description": "User-defined custom pattern",
                "sources": self.custom_sources,
                "sinks": self.custom_sinks
            }
        else:
            return self.traffic_patterns.get(self.traffic_pattern, {})
    
    def list_available_patterns(self):
        """Print available traffic patterns and their descriptions."""
        print("🚦 Available Traffic Patterns:")
        print("=" * 50)
        for name, info in self.traffic_patterns.items():
            status = "✅" if name == self.traffic_pattern else "  "
            print(f"{status} {name:12} - {info['description']}")
        
        if self.traffic_pattern == "custom":
            print(f"✅ {'custom':12} - User-defined pattern")
        
        print("\n💡 Usage:")
        print("   config.set_traffic_pattern('commuter')")
        print("   config.set_custom_traffic_sources({'north_*': 3.0, 'south_*': 1.0})")


class SensitivityConfig(Config):
    """Configuration class specifically for sensitivity analysis."""
    
    def __init__(self):
        super().__init__()
        
        # Override defaults for sensitivity analysis
        self.show_progress = False  # Disable for batch runs
        self.show_plots = False
        self.launch_sumo_gui = False
        self.verbosity = 1
        
        # Analysis-specific settings
        self.save_detailed_logs = True
        self.create_plots = True
        self.n_runs_per_config = 3


# Global configuration instance
config = Config()
