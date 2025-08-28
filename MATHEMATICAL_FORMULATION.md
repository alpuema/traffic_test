# Mathematical Formulation: Traffic Light Optimization using Ant Colony Optimization

## Problem Definition

### Grid Network Structure
In an m×m SUMO grid network, intersections have varying complexity:

**Intersection Types**:
1. **Corner Intersections** (4 total): 2 incoming edges
   - Simple T-junction behavior
   - Single phase: 90s green (state="GG")
   - Low traffic complexity

2. **Edge Intersections** (4(m-2) total): 3 incoming edges  
   - More complex traffic patterns
   - 4-phase cycle: 42s-3s-42s-3s pattern
   - Moderate traffic complexity

3. **Interior Intersections** ((m-2)² total): 4 incoming edges
   - Highest traffic complexity
   - 4-phase cycle: 42s-3s-42s-3s pattern  
   - Maximum optimization potential

### Decision Variables
The optimization problem involves optimizing traffic light phase durations across a grid network:

- **Solution Vector**: `x = [d₁, d₂, ..., dₙ]` where:
  - `dᵢ` = duration (in seconds) of the i-th traffic light phase
  - `n = Σ(phases per traffic light)` = total number of phases across all traffic lights
  - **ACTUAL STRUCTURE**: For an m×m grid, the number of phases varies by intersection type:
    - **Corner intersections** (4 intersections): 1 phase each (90s duration)
    - **Edge intersections** (4(m-2) intersections): 4 phases each (42s, 3s, 42s, 3s pattern)
    - **Interior intersections** ((m-2)² intersections): 4 phases each (42s, 3s, 42s, 3s pattern)
  - **3×3 grid example**: 9 intersections → 24 total phases (4 corners × 1 + 4 edges × 4 + 1 center × 4)
  - **4×4 grid example**: 16 intersections → 52 total phases (4 corners × 1 + 8 edges × 4 + 4 centers × 4)

### Search Space
The search space is **heterogeneous** based on traffic engineering principles:

#### Smart Traffic Engineering Mode (`USE_TRAFFIC_ENGINEERING_RULES = True`):
- **Green/Red phases**: `dᵢ ∈ {20, 22, 24, ..., 100}` seconds
  - Discrete set: `D_green = {20, 22, 24, 26, ..., 98, 100}` (41 values)
- **Yellow phases**: `dᵢ ∈ {3, 4, 5, 6}` seconds  
  - Discrete set: `D_yellow = {3, 4, 5, 6}` (4 values)

#### Legacy Mode (`USE_TRAFFIC_ENGINEERING_RULES = False`):
- **All phases**: `dᵢ ∈ {10, 12, 14, ..., 60}` seconds
  - Discrete set: `D_legacy = {10, 12, 14, 16, ..., 58, 60}` (26 values)

### Phase Type Classification
Each phase is classified as:
- **Green/Red phase** (`φᵢ = 1`): Optimizable with extended duration range (42s and 90s in SUMO defaults)
- **Yellow phase** (`φᵢ = 0`): Safety-critical with restricted duration range (3s in SUMO defaults)

**ACTUAL PHASE PATTERNS IN SUMO GRIDS**:
- **Corner intersections**: Single 90s green phase (state="GG")
- **Edge/Interior intersections**: 4-phase cycle:
  1. 42s green phase (states like "GggrrrGGg")
  2. 3s yellow transition (states like "yyyrrrGyy")
  3. 42s green phase for perpendicular direction (states like "rrrGGgGrr")
  4. 3s yellow transition (states like "rrryyyGrr")

Classification heuristic:
```
φᵢ = {
  0  if traffic_state contains 'y' OR default_duration ≤ 6 seconds
  1  otherwise (green phases: 42s or 90s durations)
}
```

## Objective Function

### Cost Function
The optimization minimizes:
```
f(x) = T_total + α × T_max_stop
```

Where:
- `T_total` = Total travel time of all vehicles (seconds)
- `T_max_stop` = Maximum cumulative waiting time of any single vehicle (seconds)
- `α` = Stop time penalty weight (default: 30.0)

### Performance Metrics Calculation
1. **Total Travel Time**: `T_total = Σᵢ(tᵢ)` where `tᵢ` = travel time of vehicle i
2. **Maximum Stop Time**: `T_max_stop = max(wᵢ)` where `wᵢ` = waiting time of vehicle i
3. **Incomplete Trip Penalty**: Additional penalty of 1000 seconds per vehicle that doesn't complete its journey

## Ant Colony Optimization Algorithm

### Pheromone Matrix Structure
- **Matrix Size**: `τ ∈ ℝⁿ×ᵐ` where:
  - `n` = number of decision variables (total phases across all intersections)
  - `m` = size of duration bins (varies by phase type)
- **Heterogeneous Structure**: Different bins for different phase types:
  - Green/Red phases: 41 bins (20-100s range, step=2)
  - Yellow phases: 4 bins (3-6s range, safety-constrained)
- **Initialization**: `τᵢⱼ(0) = τ₀` (default: 1.0)

### Pheromone Seeding (Default Starting Point)
When `USE_DEFAULT_STARTING_POINT = True`:
```
τᵢⱼ(0) = {
  τ₀ + 2.0  if dⱼ = d_default[i]
  τ₀        otherwise
}
```
Where `d_default` is SUMO's default phase durations.

### Solution Construction

#### Basic Ant Solution:
For each phase `i`, select duration based on pheromone probability:
```
P(dⱼ|i) = τᵢⱼ(t) / Σₖ τᵢₖ(t)
```

#### Coordinated Ant Solution:
When `USE_COORDINATION = True`, add coordination influence:
```
P_coord(dⱼ|i) = (1-λ) × P_base(dⱼ|i) + λ × P_coord_bias(dⱼ|i)
```

Where:
- `λ` = coordination factor (default: 0.2)
- `P_coord_bias(dⱼ|i) = exp(-|dⱼ - dᵢ₋₁|/10) / Z` (favor similar durations)
- `Z` = normalization constant

### Pheromone Update Rule
After all ants complete their solutions:

1. **Evaporation**:
   ```
   τᵢⱼ(t+1) = (1 - ρ) × τᵢⱼ(t)
   ```
   Where `ρ` = evaporation rate (default: 0.3)

2. **Proportional Reward**:
   ```
   τᵢⱼ(t+1) += Σₖ Rₖ × δᵢⱼᵏ
   ```
   
   Where:
   - `δᵢⱼᵏ = 1` if ant k chose duration j for phase i, 0 otherwise
   - `Rₖ` = reward for ant k based on relative performance:
   
   ```
   Rₖ = 0.1 + 0.9 × (f_worst - fₖ)/(f_worst - f_best)
   ```

## Algorithm Parameters

### Core ACO Parameters
- **Population Size**: `m = 40` ants per iteration
- **Iterations**: `T = 5` iterations
- **Evaporation Rate**: `ρ = 0.3`
- **Initial Pheromone**: `τ₀ = 1.0`

### Problem-Specific Parameters
- **Stop Time Weight**: `α = 30.0` (cost function)
- **Coordination Factor**: `λ = 0.2` (phase coordination)
- **Grid Size**: `g ∈ {2, 3, 4, 5}` (network dimensions)
- **Simulation Time**: `T_sim = 1800` seconds (default)
- **Vehicle Count**: `N_vehicles` varies by configuration (20-100 typical)

### Search Space Parameters
- **Green/Red Phase Range**: `[20, 100]` seconds (step: 2) - 41 values
- **Yellow Phase Range**: `[3, 6]` seconds (all values) - 4 values
- **Legacy Range**: `[10, 60]` seconds (step: 2) - 26 values (compatibility mode)

## Key Implementation Details

### Traffic Engineering Constraints
1. **Phase Type Recognition**: Automatic classification of green vs yellow phases
2. **Safety Constraints**: Yellow phases restricted to 3-6 seconds
3. **Duration Discretization**: All durations rounded to even numbers

### Coordination Mechanism
- **Spatial Coordination**: Adjacent phases influence each other
- **Exponential Decay**: `exp(-distance/10)` for coordination bias
- **Sequential Processing**: Each phase considers the previous phase duration

### Default Solution Integration
- **SUMO Baseline**: Starts optimization from proven traffic engineering defaults
  - **Corner phases**: 90s continuous green
  - **Standard phases**: 42s green + 3s yellow alternating cycle
- **Pheromone Seeding**: Gives extra weight to default durations in initial pheromone matrix
- **Engineering Validation**: SUMO defaults represent established traffic engineering practice, making significant improvements challenging but not impossible

### Convergence Criteria
- **Fixed Iterations**: Algorithm runs for exactly T iterations
- **Best Solution Tracking**: Maintains best solution found across all iterations
- **Performance Metrics**: Tracks cost, travel time, and stop time separately

## Computational Complexity
- **Solution Space Size**: `|D_green|^{n_green} × |D_yellow|^{n_yellow}`
  - **3×3 grid example**: 16 green phases + 8 yellow phases
    - `41^{16} × 4^{8} = 10^{26} × 10^{1.2} ≈ 10^{27}` combinations
  - **4×4 grid example**: 32 green phases + 20 yellow phases  
    - `41^{32} × 4^{20} = 10^{51} × 10^{6} ≈ 10^{57}` combinations
  - Exponential growth makes exhaustive search infeasible for even small grids
- **Per-Iteration Cost**: O(m × n × S) where S is SUMO simulation time
- **Total Evaluations**: `m × T = 40 × 5 = 200` SUMO simulations

**Key Insight**: SUMO's default durations (42s green, 90s corner, 3s yellow) are already within or near the optimal engineering ranges, explaining why significant improvements are challenging to achieve.

## Worked Example: 3×3 Grid Network

### Network Structure:
- **9 intersections total**: A0, A1, A2, B0, B1, B2, C0, C1, C2
- **4 corner intersections**: A0, A2, C0, C2 (1 phase each = 4 phases)
- **4 edge intersections**: A1, B0, B2, C1 (4 phases each = 16 phases)  
- **1 interior intersection**: B1 (4 phases each = 4 phases)
- **Total: 24 phases** to optimize

### Solution Vector Structure:
```
x = [d₁, d₂, ..., d₂₄] where:
- d₁ = A0 phase duration (90s default, corner type)
- d₂, d₃, d₄, d₅ = A1 phases (42s, 3s, 42s, 3s default)
- d₆ = A2 phase duration (90s default, corner type)
- ... (continues for all 24 phases)
```

### Phase Type Classification:
- **16 green/red phases**: φᵢ = 1 (corner 90s phases + edge/interior 42s phases)  
- **8 yellow phases**: φᵢ = 0 (edge/interior 3s transition phases)

### Search Space Size:
- Green phases: 41 possible values each → 41¹⁶ combinations
- Yellow phases: 4 possible values each → 4⁸ combinations  
- **Total: 41¹⁶ × 4⁸ ≈ 10²⁷ combinations**

This demonstrates why metaheuristic optimization is essential and why the heterogeneous search space design is critical for tractable optimization.
