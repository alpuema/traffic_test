# Clean Traffic Light Optimization System

A simplified, easy-to-understand traffic light optimization system using Ant Colony Optimization (ACO). This version eliminates the complex "bins" system and provides a clean interface for traffic optimization with multiple traffic patterns and seed-based reproducibility.

## 🎯 Key Improvements

### ✅ **Simplified Algorithm**
- **No complex bins**: Direct range sampling (20-100s green, 3-6s yellow)
- **Reduced memory**: 26× less memory usage (24 vs 624 pheromone elements)
- **No array mapping**: Eliminates index mismatch errors that caused iteration issues
- **Easy debugging**: Clean, understandable code structure

### ✅ **Traffic Engineering**
- **Proper constraints**: Green phases 20-100s (includes SUMO's 42s, 90s defaults)
- **Safety standards**: Yellow phases 3-6s (traffic engineering requirement)  
- **Multiple patterns**: Random, commuter, commercial, industrial, balanced

### ✅ **Reproducible Results**
- **Seed support**: Consistent results with same seed
- **Solution saving**: Save optimized solutions for later evaluation
- **Cross-evaluation**: Test saved solutions with different traffic seeds

## 🚀 Quick Start

### Basic Optimization
```bash
python optimize.py --grid 3 --vehicles 30 --pattern commuter --seed 42
```

### Evaluate Existing Solution with New Traffic
```bash
python optimize.py --evaluate solution_commuter_20250827.json --new-seed 123
```

### List Available Traffic Patterns
```bash
python optimize.py --list-patterns
```

## 📁 Clean Project Structure

```
📁 my_grid_simulation/
├── 🎯 optimize.py                    # Main optimization tool
├── 🧪 test_simplified.py             # System verification test
├── 📋 README_CLEAN.md                # This file
├── 
├── 📁 src/
│   ├── 📁 optimization/
│   │   └── 🐜 simple_aco.py          # Simplified ACO algorithm
│   └── 🚦 simplified_traffic.py      # Clean traffic generation
│
├── 📁 sumo_data/                     # Generated SUMO files
│   ├── grid_NxN.net.xml              # Network files
│   ├── grid_NxN.rou.xml              # Route files
│   └── grid_NxN.sumocfg              # SUMO configurations
│
├── 📁 results/                       # Optimization results
│   └── solution_*.json               # Saved solutions
│
└── 📁 examples/                      # Legacy examples (can be removed)
```

## 🛠️ Configuration Options

### Scenario Parameters
- `--grid N`: Grid size (2, 3, 4, 5, etc.)
- `--vehicles N`: Number of vehicles (10-100+)
- `--time N`: Simulation time in seconds (300-1200+)
- `--pattern NAME`: Traffic pattern (see below)
- `--seed N`: Random seed for reproducible results

### ACO Parameters  
- `--ants N`: Number of ants per iteration (10-50)
- `--iterations N`: Number of optimization iterations (5-20)

### Traffic Patterns
- **`random`**: Completely random origins and destinations
- **`commuter`**: Rush hour pattern (suburbs → downtown)
- **`commercial`**: Shopping district (distributed → concentrated)
- **`industrial`**: Industrial zones (residential → industrial)
- **`balanced`**: Realistic balanced urban traffic

## 📊 Example Workflows

### Workflow 1: Basic Optimization
```bash
# Optimize 3x3 grid with commuter traffic
python optimize.py --grid 3 --vehicles 30 --pattern commuter --seed 42

# Results saved to: results/solution_commuter_20250827_153045.json
```

### Workflow 2: Cross-Evaluation
```bash
# Train on commuter pattern
python optimize.py --grid 3 --vehicles 30 --pattern commuter --seed 42

# Test same solution on different traffic (new seed)
python optimize.py --evaluate results/solution_commuter_*.json --new-seed 999

# Test with different traffic volume
python optimize.py --evaluate results/solution_commuter_*.json --new-seed 999 --vehicles 50
```

### Workflow 3: Pattern Comparison
```bash
# Optimize for different traffic patterns
python optimize.py --grid 3 --pattern random --seed 100
python optimize.py --grid 3 --pattern commuter --seed 100  
python optimize.py --grid 3 --pattern commercial --seed 100

# Compare results across patterns
```

## 🔬 Technical Details

### Eliminated Complexity
The old system had:
- `DURATION_BINS = [10,12,14,...,60]` (26 values)
- `GREEN_DURATION_BINS = [20,22,24,...,100]` (41 values) 
- `YELLOW_DURATION_BINS = [3,4,5,6]` (4 values)
- Complex mapping between different bin arrays
- Pheromone matrix: 24 phases × 26 bins = 624 elements
- Fallback mechanisms for missing bins

### New Simple System
The new system uses:
- Direct range sampling: `random.randint(20, 100)` for green phases
- Direct range sampling: `random.randint(3, 6)` for yellow phases  
- Simple pheromone: 24 phases × 1 weight = 24 elements
- No mapping, no fallbacks, no errors

### Why This Fixes Iteration Issues
The iteration degradation was caused by:
1. **Array size mismatches** between different bin systems
2. **Pheromone mapping errors** when solutions used bins not in main array
3. **"Closest bin" fallback** corrupting pheromone trails

The simple system eliminates all of these by using direct sampling.

## 🧪 Verification

Run the test suite to verify everything works:

```bash
python test_simplified.py
```

Expected output:
```
✅ Traffic generation successful
✅ Traffic engineering constraints satisfied  
✅ Pheromone reinforcement working
🎉 ALL TESTS PASSED!
```

## 🔄 Migration from Old System

If you have existing results from the complex system:

1. **Use the new system**: Much more reliable and easier to debug
2. **Re-run optimizations**: Results should be better due to fixed iteration issues
3. **Clean up**: The `examples/` directory contains the old complex examples

## 📈 Expected Performance

With the simplified system, you should see:
- ✅ **Stable iterations**: No more degradation after first iteration
- ✅ **Better convergence**: Cleaner pheromone reinforcement  
- ✅ **Faster execution**: Less computational overhead
- ✅ **Easier debugging**: Transparent algorithm behavior

## 🎯 Next Steps

1. Run `python test_simplified.py` to verify your system
2. Try basic optimization: `python optimize.py --grid 3 --vehicles 20`
3. Experiment with different traffic patterns
4. Save and cross-evaluate solutions with different seeds

The system is now clean, reliable, and ready for production use! 🚀
