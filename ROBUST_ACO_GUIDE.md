# Robust Multi-Seed ACO Documentation

## Overview

The Robust ACO (Ant Colony Optimization) algorithm addresses the overfitting problem you identified where "when evaluating with other fits performance is worse than the baseline." This implementation trains on multiple traffic scenarios simultaneously to find solutions that generalize well to unseen traffic patterns.

## Key Benefits

✅ **Prevents Overfitting**: Trains on N different traffic seeds instead of just one
✅ **Better Generalization**: Solutions work well on new/unseen traffic patterns  
✅ **Robust Performance**: Less sensitive to specific traffic scenario quirks
✅ **Adaptive Learning**: Focuses more on challenging scenarios during training
✅ **Validation Testing**: Tests final solution on completely fresh seeds

## Quick Start

```python
from src.optimization.robust_aco import RobustACOTrafficOptimizer

# Create optimizer with robust parameters
optimizer = RobustACOTrafficOptimizer(
    sumo_config="path/to/scenario.sumocfg",
    training_seeds=5,          # Train on 5 different traffic scenarios
    exploration_rate=0.25,     # Higher exploration for robustness
    validate_solution=True     # Test on fresh seeds
)

# Run optimization
best_solution, cost, data, comparison = optimizer.optimize()
```

## Key Parameters

### Robust-Specific Parameters
- **`training_seeds`**: Number of different traffic scenarios to train on (3-10 recommended)
- **`exploration_rate`**: ACO exploration rate (0.20-0.30 for robustness)  
- **`validate_solution`**: Test final solution on new seeds (True recommended)
- **`adaptive_weighting`**: Focus more on challenging scenarios (True default)
- **`consensus_threshold`**: Agreement needed across seeds (0.6-0.8 range)

### Standard ACO Parameters
- **`n_ants`**: Ants per iteration (can be lower since each evaluation is more expensive)
- **`n_iterations`**: Optimization iterations (can be lower due to robust training)
- **`alpha`**: Stop penalty weight (1.0 default)
- **`beta`**: Heuristic weight (2.0 default)
- **`rho`**: Pheromone evaporation rate (0.05-0.10 recommended)

## How It Works

### 1. Multi-Seed Training
```
Seed 42:   Traffic Pattern A → Evaluate solution → Cost A
Seed 123:  Traffic Pattern B → Evaluate solution → Cost B  
Seed 456:  Traffic Pattern C → Evaluate solution → Cost C
           
Final Cost = weighted_average(Cost A, Cost B, Cost C)
```

### 2. Adaptive Weighting
- Seeds with higher costs get more weight in training
- Helps algorithm focus on challenging scenarios
- Prevents solutions that work only on "easy" traffic patterns

### 3. Consensus Building  
- Solutions must work well across most training seeds
- Filters out solutions that excel on one seed but fail on others
- Builds robust pheromone trails from multiple perspectives

### 4. Validation Testing
- Final solution tested on completely new seeds (never seen during training)
- True measure of generalization capability
- Helps detect remaining overfitting issues

## Performance Expectations

### Training Time
- **Robust ACO**: ~3-5x longer than regular ACO
- **Reason**: Each solution evaluated on multiple seeds
- **Recommendation**: Use fewer ants/iterations but higher quality training

### Solution Quality
- **Training Performance**: May be slightly worse than single-seed ACO
- **Generalization**: Significantly better on unseen traffic patterns
- **Robustness**: More consistent performance across different scenarios

## Recommended Configurations

### Fast Development (2-3 minutes)
```python
training_seeds=3
n_ants=10
n_iterations=5
exploration_rate=0.25
```

### Balanced Performance (5-8 minutes)
```python
training_seeds=5
n_ants=15
n_iterations=8
exploration_rate=0.25
```

### High Robustness (10-15 minutes)
```python
training_seeds=8
n_ants=20
n_iterations=10
exploration_rate=0.30
```

## Example Usage

### Basic Robust Optimization
```python
# examples/robust_aco_example.py shows full example
python examples/robust_aco_example.py
```

### Comparing Regular vs Robust
```python
# Run both optimizers on same scenario
regular_optimizer = ACOTrafficOptimizer(config_file, n_ants=20, n_iterations=10)
robust_optimizer = RobustACOTrafficOptimizer(config_file, training_seeds=5, n_ants=15, n_iterations=8)

# Compare results on multiple test seeds
```

### Custom Traffic Patterns
```python
# Train robust ACO on specific pattern
for pattern in ['commuter', 'industrial', 'random']:
    scenario = create_traffic_scenario(pattern=pattern)
    optimizer = RobustACOTrafficOptimizer(sumo_config=scenario['config_file'])
    solution, cost, data, comparison = optimizer.optimize()
```

## Troubleshooting

### Performance Issues
- **Problem**: Robust ACO takes too long
- **Solution**: Reduce `training_seeds` or `n_iterations`, use smaller grid sizes for testing

### Poor Generalization
- **Problem**: Validation improvement is negative
- **Solution**: Increase `training_seeds`, raise `exploration_rate`, check scenario diversity

### Inconsistent Results
- **Problem**: Results vary significantly between runs  
- **Solution**: Increase `n_iterations`, ensure sufficient `training_seeds`, check SUMO installation

### Memory Issues
- **Problem**: Out of memory during multi-seed training
- **Solution**: Reduce `n_ants`, close other applications, use smaller grid sizes

## File Locations

- **Implementation**: `src/optimization/robust_aco.py`
- **Example Script**: `examples/robust_aco_example.py` 
- **Test Script**: `test_robust_aco.py`
- **Documentation**: This file

## Integration with Existing Code

The robust ACO is designed to be a drop-in replacement for regular ACO:

```python
# Old way (single-seed, prone to overfitting)
from src.optimize import ACOTrafficOptimizer
optimizer = ACOTrafficOptimizer(config_file, n_ants=20, n_iterations=10)

# New way (multi-seed, robust)
from src.optimization.robust_aco import RobustACOTrafficOptimizer  
optimizer = RobustACOTrafficOptimizer(config_file, training_seeds=5, n_ants=15, n_iterations=8)

# Same interface for optimization
best_solution, cost, data, comparison = optimizer.optimize()
```

**No changes needed to existing functionality** - your current scripts continue to work unchanged.
