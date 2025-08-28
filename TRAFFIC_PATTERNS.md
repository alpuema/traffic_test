# Traffic Pattern Analysis System

This system provides sophisticated traffic pattern generation and analysis capabilities for SUMO simulations, enabling detailed study of how traffic light optimization strategies respond to different traffic scenarios.

## 🚗 Overview

The traffic pattern system replaces simple random vehicle spawning with realistic, configurable traffic flows that mirror real-world scenarios. This enables analysis of:

- **Directional Priority**: How traffic lights prioritize different flow directions
- **Pattern-Specific Optimization**: How ACO adapts to different traffic scenarios  
- **Realistic Traffic Simulation**: More accurate representation of urban traffic
- **Scenario-Based Analysis**: Targeted testing of specific traffic conditions

## 🎯 Available Traffic Patterns

### Built-in Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| `random` | Random origins/destinations (original behavior) | Baseline comparison |
| `realistic` | Balanced urban traffic with perimeter bias | General urban analysis |
| `commuter` | Rush hour: perimeter → center flow | Morning/evening rush |
| `commercial` | Shopping/business: distributed → concentrated | Business districts |

### Custom Patterns

Create your own patterns by specifying source/sink weights:

```python
config.set_traffic_pattern("custom")
config.set_custom_traffic_sources({
    "west": 3.0,        # Heavy traffic from west
    "perimeter": 1.5,   # Some edge traffic
    "center": 0.5       # Light center traffic
})
config.custom_sinks = {
    "east": 3.0,        # Heavy traffic to east  
    "center": 2.0,      # Some center destinations
    "*": 0.5            # Light traffic elsewhere
}
```

## 🛠️ Usage

### Quick Start

```python
from config import Config
from traffic_patterns import TrafficPatternGenerator

# Set up configuration
config = Config()
config.set_traffic_pattern("commuter")  # Rush hour pattern

# Generate traffic
generator = TrafficPatternGenerator(config)
success = generator.generate_realistic_trips(
    net_file="network.net.xml",
    n_vehicles=100,
    sim_time=600,
    output_trips_file="trips.trips.xml"
)
```

### Integration with ACO

The ACO optimizer automatically uses the configured traffic pattern:

```python
from config import Config
```python
from optimization.simple_aco import run_simplified_aco_optimization

config = Config()
config.set_traffic_pattern("realistic")
config.n_vehicles = 100

results = run_simplified_aco_optimization(config)  # Uses realistic traffic pattern
```

### Demonstration Scripts

Run the demo scripts to see the system in action:

```bash
# Show available patterns
python demo_traffic_patterns.py --list

# Demonstrate a specific pattern
python demo_traffic_patterns.py commuter

# Compare all patterns
python demo_traffic_patterns.py --compare

# Custom pattern examples
python examples_traffic_patterns.py
```

## 📊 Pattern Configuration Details

### Source/Sink Weight Patterns

Patterns can use:

- **Edge Classifications**: `perimeter`, `center`, `north`, `south`, `east`, `west`
- **Wildcard Matching**: `*_0` (all edges ending in _0), `2_*` (all edges starting with 2_)
- **Specific Edges**: `2_3to3_3`, `0_0to1_0`
- **Uniform Distribution**: `"uniform"` for equal weights

### Vehicle Mix

Specify different vehicle types:

```python
pattern_info = {
    'vehicle_mix': {
        'car': 0.7,      # 70% cars
        'truck': 0.2,    # 20% trucks  
        'bus': 0.1       # 10% buses
    }
}
```

### Peak Hours

Add time-based traffic intensity:

```python
pattern_info = {
    'peak_hours': [
        {'start': 0.1, 'end': 0.4, 'multiplier': 2.5},  # Morning rush
        {'start': 0.6, 'end': 0.8, 'multiplier': 2.0}   # Evening rush
    ]
}
```

## 🔍 Analysis Features

### Traffic Light Priority Analysis

The system automatically analyzes which traffic light phases should be prioritized:

```python
priorities = generator.analyze_traffic_light_priorities(net_file)
# Output: {'tl_1': {'phase_0': 2.5, 'phase_1': 1.8, ...}, ...}
```

### Pattern Statistics

View pattern characteristics:

- Dominant flow direction
- Source/sink distribution  
- Edge utilization
- Directional bias strength

### Network Classification

Automatic edge classification:

- **Perimeter edges**: Network boundaries
- **Center edges**: Network core
- **Directional edges**: North, south, east, west flows
- **Type edges**: Horizontal vs vertical

## 📁 File Structure

```
my_grid_simulation/
├── traffic_patterns.py          # Core traffic pattern system
├── demo_traffic_patterns.py     # Demonstration script
├── examples_traffic_patterns.py # Custom pattern examples
├── config.py                    # Enhanced configuration system
└── results/
    ├── pattern_analysis/         # Pattern-specific results
    ├── temp/                     # Temporary files
    └── plots/                    # Analysis visualizations
```

## 🚀 Advanced Usage

### Scenario-Specific Analysis

```python
# Highway corridor study
config.set_traffic_pattern("custom")
config.set_custom_traffic_sources({"west": 5.0, "perimeter": 0.5})
config.custom_sinks = {"east": 5.0, "perimeter": 0.5}

# Industrial area study  
config.set_custom_traffic_sources({"south": 4.0})  # Port traffic
config.custom_sinks = {"north": 3.0, "center": 2.0}  # Factories/distribution

# Event traffic study
config.set_custom_traffic_sources("uniform")  # From everywhere
config.custom_sinks = {"2_2": 10.0}  # Stadium location
```

### Pattern Comparison Studies

```python
patterns = ["random", "realistic", "commuter", "commercial"]
results = {}

for pattern in patterns:
    config.set_traffic_pattern(pattern)
    results[pattern] = run_simplified_aco_optimization(config)
    
# Compare optimization effectiveness across patterns
```

### Custom Network Analysis

For non-grid networks, the system adapts by:

1. Parsing network topology
2. Classifying edges by location/direction
3. Applying weights based on geometric analysis
4. Falling back to random generation if needed

## 🎛️ Configuration Options

Key configuration parameters in `config.py`:

```python
# Traffic pattern selection
traffic_pattern = "realistic"        # Built-in pattern name
custom_sources = {}                  # Custom source weights  
custom_sinks = {}                    # Custom sink weights

# Pattern behavior
peak_hour_factor = 2.0               # Rush hour multiplier
vehicle_mix = {"car": 1.0}           # Vehicle type distribution
pattern_randomness = 0.1             # Variability in pattern application
```

## 🔬 Research Applications

This system enables research into:

- **Adaptive Traffic Control**: How signals adapt to different flow patterns
- **Pattern Recognition**: Identifying optimal strategies per traffic type  
- **Real-World Validation**: Testing with realistic traffic distributions
- **Scenario Planning**: Evaluating infrastructure changes
- **Multi-Objective Optimization**: Balancing different traffic priorities

## 📈 Expected Benefits

- **More Realistic Analysis**: Traffic patterns reflect real urban conditions
- **Better Optimization**: ACO can find pattern-specific improvements
- **Directional Insights**: Understanding which flows get priority
- **Practical Applications**: Results applicable to real traffic management
- **Research Value**: Enables sophisticated traffic engineering studies

## 🔧 Troubleshooting

### Common Issues

1. **Pattern not applied**: Ensure traffic pattern is set before creating optimizer
2. **No weighted edges**: Check network file exists and is valid
3. **Random fallback**: Verify edge classifications work with your network  
4. **Performance issues**: Reduce verbosity level for large simulations

### Debug Mode

Enable verbose output to see pattern application:

```python
config.verbose = 2  # Maximum detail
config.set_traffic_pattern("realistic")
```

## 🔗 Integration

The traffic pattern system integrates seamlessly with:

- **ACO Optimization** (`aco.py`): Automatic pattern-based trip generation
- **Sensitivity Analysis** (`sensitivity_analysis.py`): Pattern-aware parameter sweeps  
- **Configuration Management** (`config.py`): Centralized pattern settings
- **Result Organization** (`cleanup_repository.py`): Pattern-specific result storage

---

*This traffic pattern system transforms simple random vehicle spawning into sophisticated, realistic traffic flow simulation that enables meaningful analysis of traffic light optimization strategies.*
