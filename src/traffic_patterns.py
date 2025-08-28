"""
Advanced Traffic Pattern Generation for SUMO Simulations

This module provides sophisticated traffic pattern generation with configurable
sources, sinks, and realistic traffic flows for better analysis of traffic
light optimization performance.

Author: Alfonso Rato
Date: August 2025
"""

import os
import random
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Any, Optional
import re
import numpy as np


class TrafficPatternGenerator:
    """Generate realistic traffic patterns for SUMO simulations."""
    
    def __init__(self, config):
        """Initialize with configuration object."""
        self.config = config
        self.edge_classifications = {}
        self.weighted_sources = {}
        self.weighted_sinks = {}
    
    def classify_network_edges(self, net_file: str) -> Dict[str, List[str]]:
        """Classify network edges by location (perimeter, center, directions)."""
        try:
            tree = ET.parse(net_file)
            root = tree.getroot()
            
            edges = []
            edge_coords = {}
            
            # Get edges and their coordinates
            for edge in root.findall('edge'):
                edge_id = edge.get('id')
                if edge_id and not edge_id.startswith(':'):  # Skip internal junction edges
                    edges.append(edge_id)
                    
                    # Get edge coordinates from lanes
                    lane = edge.find('lane')
                    if lane is not None:
                        shape = lane.get('shape', '')
                        if shape:
                            coords = self._parse_coordinates(shape)
                            edge_coords[edge_id] = coords
            
            # Analyze grid structure
            grid_size = self.config.grid_size
            edge_length = 200  # Standard SUMO grid edge length
            grid_bounds = {
                'min_x': 0,
                'max_x': (grid_size - 1) * edge_length,
                'min_y': 0, 
                'max_y': (grid_size - 1) * edge_length,
                'center_x': (grid_size - 1) * edge_length / 2,
                'center_y': (grid_size - 1) * edge_length / 2
            }
            
            # Classify edges
            classifications = {
                'perimeter': [],
                'center': [],
                'north': [],
                'south': [],
                'east': [],
                'west': [],
                'horizontal': [],
                'vertical': []
            }
            
            for edge_id in edges:
                coords = edge_coords.get(edge_id)
                if not coords:
                    continue
                
                start_x, start_y = coords[0]
                end_x, end_y = coords[-1]
                
                # Determine if edge is on perimeter or center
                is_perimeter = (
                    start_x <= grid_bounds['min_x'] + 50 or start_x >= grid_bounds['max_x'] - 50 or
                    start_y <= grid_bounds['min_y'] + 50 or start_y >= grid_bounds['max_y'] - 50 or
                    end_x <= grid_bounds['min_x'] + 50 or end_x >= grid_bounds['max_x'] - 50 or
                    end_y <= grid_bounds['min_y'] + 50 or end_y >= grid_bounds['max_y'] - 50
                )
                
                center_threshold = edge_length * 0.4
                is_center = (
                    abs(start_x - grid_bounds['center_x']) < center_threshold and
                    abs(start_y - grid_bounds['center_y']) < center_threshold
                )
                
                if is_perimeter:
                    classifications['perimeter'].append(edge_id)
                if is_center:
                    classifications['center'].append(edge_id)
                
                # Determine direction
                if abs(end_x - start_x) > abs(end_y - start_y):  # Horizontal edge
                    classifications['horizontal'].append(edge_id)
                    if end_x > start_x:
                        classifications['east'].append(edge_id)
                    else:
                        classifications['west'].append(edge_id)
                else:  # Vertical edge
                    classifications['vertical'].append(edge_id)
                    if end_y > start_y:
                        classifications['north'].append(edge_id)
                    else:
                        classifications['south'].append(edge_id)
            
            self.edge_classifications = classifications
            
            print(f"🗺️ Network edge classification:")
            for category, edges in classifications.items():
                if edges:
                    print(f"   {category}: {len(edges)} edges")
            
            return classifications
            
        except Exception as e:
            print(f"❌ Error classifying network edges: {e}")
            return {}
    
    def _parse_coordinates(self, shape_str: str) -> List[Tuple[float, float]]:
        """Parse SUMO shape coordinates."""
        coords = []
        for coord_pair in shape_str.split():
            try:
                x, y = map(float, coord_pair.split(','))
                coords.append((x, y))
            except ValueError:
                continue
        return coords
    
    def _match_edge_pattern(self, edge_id: str, pattern: str) -> bool:
        """Check if edge ID matches a pattern (supports wildcards)."""
        # Convert shell-like wildcards to regex
        regex_pattern = pattern.replace('*', '.*').replace('?', '.')
        return re.match(f'^{regex_pattern}$', edge_id) is not None
    
    def _get_edges_by_classification(self, classification: str) -> List[str]:
        """Get edges that match a classification."""
        if classification in self.edge_classifications:
            return self.edge_classifications[classification]
        return []
    
    def calculate_weighted_edges(self, net_file: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate weighted source and sink edges based on traffic pattern."""
        # First classify the network
        self.classify_network_edges(net_file)
        
        pattern_info = self.config.get_traffic_pattern_info()
        
        if self.config.traffic_pattern == "random":
            # Original random behavior - all edges equal weight
            all_edges = []
            for edges in self.edge_classifications.values():
                all_edges.extend(edges)
            all_edges = list(set(all_edges))  # Remove duplicates
            
            uniform_weight = 1.0
            sources = {edge: uniform_weight for edge in all_edges}
            sinks = {edge: uniform_weight for edge in all_edges}
            
        elif self.config.traffic_pattern == "custom":
            sources = self._apply_pattern_weights(self.config.custom_sources)
            sinks = self._apply_pattern_weights(self.config.custom_sinks)
            
        else:
            # Apply predefined pattern weights
            source_pattern = pattern_info.get('sources', {})
            sink_pattern = pattern_info.get('sinks', source_pattern)
            
            sources = self._apply_pattern_weights(source_pattern)
            sinks = self._apply_pattern_weights(sink_pattern)
        
        self.weighted_sources = sources
        self.weighted_sinks = sinks
        
        print(f"🎯 Traffic pattern '{self.config.traffic_pattern}' applied:")
        print(f"   {len(sources)} source edges, {len(sinks)} sink edges")
        if sources:
            max_source = max(sources.values())
            print(f"   Source weights range: {min(sources.values()):.1f} - {max_source:.1f}")
        
        return sources, sinks
    
    def _apply_pattern_weights(self, pattern_weights: Dict[str, Any]) -> Dict[str, float]:
        """Apply pattern weights to edges."""
        if pattern_weights == "uniform":
            # All edges get equal weight
            all_edges = []
            for edges in self.edge_classifications.values():
                all_edges.extend(edges)
            all_edges = list(set(all_edges))
            return {edge: 1.0 for edge in all_edges}
        
        weighted_edges = {}
        
        for pattern, weight in pattern_weights.items():
            matching_edges = []
            
            # Check if pattern matches a classification
            if pattern in self.edge_classifications:
                matching_edges = self.edge_classifications[pattern]
            else:
                # Check pattern matching for all edges
                all_edges = []
                for edges in self.edge_classifications.values():
                    all_edges.extend(edges)
                all_edges = list(set(all_edges))
                
                for edge in all_edges:
                    if self._match_edge_pattern(edge, pattern):
                        matching_edges.append(edge)
            
            # Apply weights
            for edge in matching_edges:
                weighted_edges[edge] = weighted_edges.get(edge, 0) + weight
        
        return weighted_edges
    
    def generate_realistic_trips(self, net_file: str, n_vehicles: int, sim_time: int, 
                               output_trips_file: str) -> bool:
        """Generate trips using the configured traffic pattern."""
        print(f"🚗 Generating {n_vehicles} trips with '{self.config.traffic_pattern}' pattern...")
        
        try:
            # Calculate weighted edges
            sources, sinks = self.calculate_weighted_edges(net_file)
            
            if not sources or not sinks:
                print("⚠️ No weighted edges found, falling back to random generation")
                return self._generate_random_trips(net_file, n_vehicles, sim_time, output_trips_file)
            
            # Create weighted lists for sampling
            source_edges = list(sources.keys())
            source_weights = [sources[edge] for edge in source_edges]
            
            sink_edges = list(sinks.keys())
            sink_weights = [sinks[edge] for edge in sink_edges]
            
            # Normalize weights
            source_weights = np.array(source_weights) / sum(source_weights)
            sink_weights = np.array(sink_weights) / sum(sink_weights)
            
            # Generate trips
            trip_content = ['<?xml version="1.0" encoding="UTF-8"?>']
            trip_content.append('<trips xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/trips_file.xsd">')
            
            # Calculate departure times
            departure_window = sim_time * 0.75
            departure_interval = departure_window / n_vehicles if n_vehicles > 0 else 1.0
            
            pattern_info = self.config.get_traffic_pattern_info()
            peak_hours = pattern_info.get('peak_hours', [])
            
            for i in range(n_vehicles):
                # Calculate base departure time
                depart_time = i * departure_interval + random.uniform(0, min(departure_interval * 0.3, 5.0))
                depart_time = min(depart_time, departure_window)
                
                # Apply peak hour multipliers
                for peak in peak_hours:
                    peak_start = sim_time * peak['start']
                    peak_end = sim_time * peak['end']
                    if peak_start <= depart_time <= peak_end:
                        # Increase probability of departure during peak
                        if random.random() < (peak['multiplier'] - 1.0) / peak['multiplier']:
                            depart_time *= random.uniform(0.8, 1.2)  # Some variation
                
                # Select source and destination based on weights
                from_edge = np.random.choice(source_edges, p=source_weights)
                to_edge = np.random.choice(sink_edges, p=sink_weights)
                
                # Ensure different source and destination
                attempts = 0
                while to_edge == from_edge and attempts < 10:
                    to_edge = np.random.choice(sink_edges, p=sink_weights)
                    attempts += 1
                
                # Determine vehicle type if specified in pattern
                vtype = "car"  # default
                vehicle_mix = pattern_info.get('vehicle_mix', {})
                if vehicle_mix:
                    types = list(vehicle_mix.keys())
                    type_weights = list(vehicle_mix.values())
                    type_weights = np.array(type_weights) / sum(type_weights)
                    vtype = np.random.choice(types, p=type_weights)
                
                trip_content.append(
                    f'    <trip id="trip_{i}" depart="{depart_time:.2f}" from="{from_edge}" to="{to_edge}" '
                    f'type="{vtype}" departLane="random" departSpeed="random"/>'
                )
            
            trip_content.append('</trips>')
            
            # Write trips file
            with open(output_trips_file, 'w') as f:
                f.write('\n'.join(trip_content))
            
            # Print pattern statistics
            self._print_pattern_statistics(sources, sinks)
            
            print(f"   ✅ Realistic trips created: {output_trips_file}")
            return True
            
        except Exception as e:
            print(f"   ❌ Realistic trip generation failed: {e}")
            print("   🔄 Falling back to random generation...")
            return self._generate_random_trips(net_file, n_vehicles, sim_time, output_trips_file)
    
    def _generate_random_trips(self, net_file: str, n_vehicles: int, sim_time: int, 
                              output_trips_file: str) -> bool:
        """Fallback to original random trip generation."""
        try:
            tree = ET.parse(net_file)
            root = tree.getroot()
            
            edges = []
            for edge in root.findall('edge'):
                edge_id = edge.get('id')
                if edge_id and not edge_id.startswith(':'):
                    edges.append(edge_id)
            
            if len(edges) < 2:
                print(f"   ❌ Insufficient edges found: {len(edges)}")
                return False
            
            trip_content = ['<?xml version="1.0" encoding="UTF-8"?>']
            trip_content.append('<trips xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/trips_file.xsd">')
            
            departure_window = sim_time * 0.75
            departure_interval = departure_window / n_vehicles if n_vehicles > 0 else 1.0
            
            for i in range(n_vehicles):
                depart_time = i * departure_interval + random.uniform(0, min(departure_interval * 0.3, 5.0))
                depart_time = min(depart_time, departure_window)
                
                from_edge = random.choice(edges)
                to_edge = random.choice([e for e in edges if e != from_edge])
                
                trip_content.append(f'    <trip id="trip_{i}" depart="{depart_time:.2f}" from="{from_edge}" to="{to_edge}" departLane="random" departSpeed="random"/>')
            
            trip_content.append('</trips>')
            
            with open(output_trips_file, 'w') as f:
                f.write('\n'.join(trip_content))
            
            print(f"   ✅ Random trips created: {output_trips_file}")
            return True
            
        except Exception as e:
            print(f"   ❌ Random trip generation failed: {e}")
            return False
    
    def _print_pattern_statistics(self, sources: Dict[str, float], sinks: Dict[str, float]):
        """Print traffic pattern statistics."""
        print(f"   📊 Traffic pattern statistics:")
        
        # Show top sources and sinks
        sorted_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)[:5]
        sorted_sinks = sorted(sinks.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"      Top sources: {', '.join([f'{edge}({w:.1f})' for edge, w in sorted_sources])}")
        print(f"      Top sinks:   {', '.join([f'{edge}({w:.1f})' for edge, w in sorted_sinks])}")
        
        # Calculate directional bias
        directional_sources = {}
        for direction in ['north', 'south', 'east', 'west']:
            direction_edges = self.edge_classifications.get(direction, [])
            total_weight = sum(sources.get(edge, 0) for edge in direction_edges)
            if total_weight > 0:
                directional_sources[direction] = total_weight
        
        if directional_sources:
            max_dir = max(directional_sources, key=directional_sources.get)
            print(f"      Dominant flow direction: {max_dir} ({directional_sources[max_dir]:.1f})")
    
    def analyze_traffic_light_priorities(self, net_file: str) -> Dict[str, Dict[str, float]]:
        """Analyze which traffic light phases should be prioritized based on traffic pattern."""
        try:
            tree = ET.parse(net_file)
            root = tree.getroot()
            
            # Get traffic light information
            tl_priorities = {}
            
            for tl_logic in root.findall('tlLogic'):
                tl_id = tl_logic.get('id')
                if not tl_id:
                    continue
                
                # Find incoming edges for this traffic light
                incoming_edges = []
                for junction in root.findall('junction'):
                    if junction.get('id') == tl_id and junction.get('type') == 'traffic_light':
                        incLanes = junction.get('incLanes', '').split()
                        for lane in incLanes:
                            edge_id = lane.split('_')[0]  # Extract edge ID from lane
                            if edge_id not in incoming_edges:
                                incoming_edges.append(edge_id)
                
                # Calculate priority based on traffic sources
                phase_priorities = {}
                sources = self.weighted_sources
                
                for i, phase in enumerate(tl_logic.findall('phase')):
                    state = phase.get('state', '')
                    phase_priority = 0.0
                    
                    # Simple heuristic: phases with 'G' (green) get priority based on source weights
                    green_count = state.count('G')
                    if green_count > 0:
                        # Calculate average source weight for edges served by this phase
                        total_source_weight = sum(sources.get(edge, 0) for edge in incoming_edges)
                        phase_priority = total_source_weight * green_count / len(state) if len(state) > 0 else 0
                    
                    phase_priorities[f'phase_{i}'] = phase_priority
                
                tl_priorities[tl_id] = phase_priorities
            
            print(f"🚦 Traffic light priority analysis:")
            for tl_id, priorities in tl_priorities.items():
                max_phase = max(priorities, key=priorities.get) if priorities else "unknown"
                max_priority = priorities.get(max_phase, 0)
                print(f"   {tl_id}: highest priority = {max_phase} ({max_priority:.2f})")
            
            return tl_priorities
            
        except Exception as e:
            print(f"❌ Error analyzing traffic light priorities: {e}")
            return {}


def create_traffic_pattern_examples():
    """Create example configurations for different traffic patterns."""
    examples = {
        "rush_hour_to_center": {
            "description": "Morning rush hour - everyone going to city center",
            "pattern": "commuter",
            "settings": {
                "sources": {"perimeter": 4.0, "center": 0.5}, 
                "sinks": {"center": 5.0, "perimeter": 0.5},
                "peak_hours": [{"start": 0.1, "end": 0.4, "multiplier": 2.5}]
            }
        },
        
        "shopping_district": {
            "description": "Shopping district - distributed origins and destinations",
            "pattern": "commercial",
            "settings": {
                "sources": "uniform",
                "sinks": {"center": 2.0, "perimeter": 1.0},
                "vehicle_mix": {"car": 0.8, "truck": 0.2}
            }
        },
        
        "highway_corridor": {
            "description": "Highway corridor - strong directional flow",
            "pattern": "custom", 
            "sources": {"west": 3.0, "east": 0.5},
            "sinks": {"east": 3.0, "west": 0.5}
        },
        
        "industrial_area": {
            "description": "Industrial area - trucks to/from specific zones",
            "pattern": "custom",
            "sources": {"south": 2.0, "perimeter": 1.0},
            "sinks": {"north": 2.0, "center": 1.5},
            "vehicle_mix": {"truck": 0.6, "car": 0.4}
        }
    }
    
    return examples
