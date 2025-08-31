#!/usr/bin/env python3
"""
Save & Evaluate Example: Complete Train-Test Pipeline

This example demonstrates how to:
1. Train an optimization on one traffic seed
2. Save the optimized solution 
3. Load and evaluate it on different traffic seeds
4. Compare performance across multiple seeds

This is essential for validating traffic light solutions work well
across different traffic conditions, not just the training scenario.
"""

import sys
import os
import time
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simplified_traffic import (
    generate_network_and_routes, 
    save_optimized_solution, 
    load_solution,
    evaluate_solution_with_new_seed
)
from src.optimize import ACOTrafficOptimizer, evaluate_existing_solution
from src.optimization.simple_aco import create_baseline_solution, evaluate_solution, calculate_cost

def evaluate_baseline_on_seed(test_seed, training_config, phase_types):
    """
    Evaluate baseline solution (30s green, 4s yellow) on a specific seed.
    
    Args:
        test_seed: Random seed for traffic generation
        training_config: Configuration dict with scenario parameters
        phase_types: Phase type information from training
        
    Returns:
        Dictionary with baseline evaluation results
    """
    try:
        # Generate new scenario with test seed
        scenario = generate_network_and_routes(
            grid_size=training_config["grid_size"],
            n_vehicles=training_config["n_vehicles"],
            sim_time=training_config["simulation_time"],
            pattern=training_config["traffic_pattern"],
            seed=test_seed
        )
        
        if not scenario['success']:
            return {'success': False, 'error': 'Scenario generation failed'}
        
        # Create baseline solution
        baseline_solution = create_baseline_solution(phase_types, green_duration=30, yellow_duration=4)
        
        # Evaluate baseline solution
        temp_dir = "temp_baseline"
        os.makedirs(temp_dir, exist_ok=True)
        
        metrics = evaluate_solution(
            baseline_solution, 
            scenario['files']['network'], 
            scenario['files']['routes'], 
            temp_dir
        )
        
        cost = calculate_cost(metrics)
        avg_time = metrics['total_time'] / metrics['vehicles'] if metrics['vehicles'] > 0 else 0
        
        # Cleanup
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        return {
            'success': True,
            'cost': cost,
            'avg_time': avg_time,
            'metrics': metrics
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def main():
    print("🎯 SAVE & EVALUATE DEMO: Complete Train-Test Pipeline")
    print("=" * 65)
    print()
    
    # ========================================================================
    # STEP 1: TRAIN ON SEED 42
    # ========================================================================
    
    print("🎓 STEP 1: Training Phase")
    print("-" * 30)
    
    training_config = {
        "grid_size": 5,
        "n_vehicles": 20,
        "simulation_time": 1400,
        "traffic_pattern": "industrial",
        "random_seed": 42,  # Training seed
        "n_ants": 50,
        "n_iterations": 20,
        "alpha": 5.0,
        "evaporation_rate": 0.3,
        "verbose": False
    }
    
    print(f"📋 Training Configuration:")
    print(f"   Scenario: {training_config['grid_size']}x{training_config['grid_size']} grid, {training_config['n_vehicles']} vehicles")
    print(f"   Pattern: {training_config['traffic_pattern']}")
    print(f"   Training Seed: {training_config['random_seed']}")
    print(f"   ACO: {training_config['n_ants']} ants × {training_config['n_iterations']} iterations")
    print()
    
    # Generate training scenario
    print("🏗️ Generating training scenario...")
    scenario = generate_network_and_routes(
        grid_size=training_config["grid_size"],
        n_vehicles=training_config["n_vehicles"],
        sim_time=training_config["simulation_time"],
        pattern=training_config["traffic_pattern"],
        seed=training_config["random_seed"]
    )
    
    if not scenario['success']:
        print(f"❌ Training scenario generation failed: {scenario['error']}")
        return
        
    print("✅ Training scenario generated")
    
    # Run optimization
    print("🐜 Running ACO optimization...")
    optimizer = ACOTrafficOptimizer(
        sumo_config=scenario['files']['config'],
        n_ants=training_config["n_ants"],
        n_iterations=training_config["n_iterations"],
        alpha=training_config["alpha"],
        rho=training_config["evaporation_rate"],
        verbose=training_config["verbose"],
        scenario_vehicles=training_config["n_vehicles"],
        simulation_time=training_config["simulation_time"],
        compare_baseline=True  # Enable baseline comparison
    )
    
    start_time = time.time()
    best_solution_dict, best_cost, optimization_data, baseline_comparison = optimizer.optimize()
    training_time = time.time() - start_time
    
    print(f"✅ Training completed in {training_time:.1f}s")
    print(f"   Best training cost: {best_cost:.1f}")
    
    # Show training baseline comparison
    if baseline_comparison:
        print("📊 Training Baseline Comparison:")
        baseline_cost = baseline_comparison['baseline']['cost']
        optimized_cost = baseline_comparison['optimized']['cost']
        improvement = baseline_comparison['improvement']
        
        print(f"   Baseline (30s/4s): {baseline_cost:.1f}")
        print(f"   Optimized: {optimized_cost:.1f}")
        
        if improvement['percent'] > 0:
            print(f"   ✅ Training improvement: {improvement['percent']:.1f}% better")
        else:
            print(f"   ❌ Training worse by: {abs(improvement['percent']):.1f}%")
    
    print()
    
    # ========================================================================
    # STEP 2: SAVE THE OPTIMIZED SOLUTION
    # ========================================================================
    
    print("🎓 STEP 2: Saving Solution")
    print("-" * 30)
    
    # Convert solution dict back to list format for saving
    # (The ACO optimizer returns a dict, but save function expects list)
    solution_list = []
    phase_types = []
    
    # Extract phase types from baseline comparison if available
    if baseline_comparison and 'baseline' in baseline_comparison:
        # We can derive phase types from the baseline solution structure
        baseline_sol = baseline_comparison['baseline']['solution']
        for duration in baseline_sol:
            # Baseline uses 30s for green phases, 4s for yellow phases
            phase_types.append(duration == 30)  # True if green phase
    else:
        # Fallback: extract from solution dict
        for i in range(len(best_solution_dict)):
            phase_key = f"phase_{i}"
            if phase_key in best_solution_dict:
                phase_info = best_solution_dict[phase_key]
                phase_types.append('green' in phase_info)  # True if green phase
    
    # Convert solution dict to list
    for i in range(len(best_solution_dict)):
        phase_key = f"phase_{i}"
        if phase_key in best_solution_dict:
            phase_info = best_solution_dict[phase_key]
            if 'green' in phase_info:
                solution_list.append(phase_info['green'])
            elif 'yellow' in phase_info:
                solution_list.append(phase_info['yellow'])
    
    # Create metadata
    solution_metadata = {
        'grid_size': training_config["grid_size"],
        'n_vehicles': training_config["n_vehicles"], 
        'simulation_time': training_config["simulation_time"],
        'pattern': training_config["traffic_pattern"],
        'seed': training_config["random_seed"],
        'best_cost': best_cost,
        'training_time': training_time,
        'n_ants': training_config["n_ants"],
        'n_iterations': training_config["n_iterations"],
        'alpha': training_config["alpha"],
        'evaporation_rate': training_config["evaporation_rate"],
        'phase_types': phase_types
    }
    
    # Save solution
    results_dir = "results/final_solutions"
    os.makedirs(results_dir, exist_ok=True)
    
    saved_info = save_optimized_solution(
        solution=solution_list,
        metadata=solution_metadata, 
        output_dir=results_dir
    )
    
    solution_file = saved_info['solution_file']
    print(f"💾 Solution saved: {os.path.basename(solution_file)}")
    print(f"   Training cost: {best_cost:.1f}")
    print()
    
    # ========================================================================
    # STEP 3: EVALUATE ON MULTIPLE DIFFERENT SEEDS  
    # ========================================================================
    
    print("🎓 STEP 3: Cross-Seed Evaluation")
    print("-" * 30)
    
    test_seeds = [123, 456, 789, 999]  # Different seeds for testing
    evaluation_results = []
    baseline_results = []  # Track baseline performance on each seed
    
    print(f"🔄 Testing solution on {len(test_seeds)} different traffic seeds...")
    print("   Each test compares: Optimized vs Baseline (30s/4s) vs Training performance")
    print()
    
    for i, test_seed in enumerate(test_seeds, 1):
        print(f"   Test {i}/{len(test_seeds)}: Seed {test_seed}")
        
        # Evaluate optimized solution
        eval_result = evaluate_solution_with_new_seed(
            solution_file=solution_file,
            new_seed=test_seed,
            n_vehicles=training_config["n_vehicles"],  # Keep same vehicle count
            sim_time=training_config["simulation_time"]  # Keep same sim time
        )
        
        # Evaluate baseline solution on same seed
        baseline_result = evaluate_baseline_on_seed(test_seed, training_config, phase_types)
        
        if eval_result['success'] and baseline_result['success']:
            opt_cost = eval_result.get('cost', float('inf'))
            opt_avg_time = eval_result.get('avg_travel_time', 0)
            opt_vehicles = eval_result.get('metrics', {}).get('vehicles', 0)
            
            base_cost = baseline_result['cost']
            base_avg_time = baseline_result['avg_time']
            base_vehicles = baseline_result['metrics']['vehicles']
            
            # Calculate improvement over baseline
            if base_cost > 0 and opt_cost != float('inf'):
                improvement_pct = ((base_cost - opt_cost) / base_cost) * 100
            else:
                improvement_pct = 0
            
            print(f"      ✅ Optimized: Cost {opt_cost:.1f}, Vehicles {opt_vehicles}/{training_config['n_vehicles']}")
            print(f"      📊 Baseline:  Cost {base_cost:.1f}, Vehicles {base_vehicles}/{training_config['n_vehicles']}")
            
            if improvement_pct > 0:
                print(f"      🎯 Improvement: {improvement_pct:.1f}% better than baseline")
            elif improvement_pct < 0:
                print(f"      ⚠️  Degradation: {abs(improvement_pct):.1f}% worse than baseline")
            else:
                print(f"      ➖ Similar to baseline")
            
            evaluation_results.append({
                'seed': test_seed,
                'optimized': {'cost': opt_cost, 'avg_time': opt_avg_time, 'vehicles': opt_vehicles},
                'baseline': {'cost': base_cost, 'avg_time': base_avg_time, 'vehicles': base_vehicles},
                'improvement_percent': improvement_pct,
                'result': eval_result
            })
            baseline_results.append(baseline_result)
            
        else:
            error_msg = eval_result.get('error', 'Unknown') if not eval_result['success'] else baseline_result.get('error', 'Unknown')
            print(f"      ❌ Evaluation failed: {error_msg}")
        
        print()
    
    print(f"\n✅ Completed {len(evaluation_results)}/{len(test_seeds)} evaluations")
    print()
    
    # ========================================================================
    # STEP 4: RESULTS SUMMARY
    # ========================================================================
    
    print("🎓 STEP 4: Results Summary")
    print("-" * 30)
    
    print("📊 TRAINING vs TESTING PERFORMANCE:")
    print(f"   🎯 Training Performance (Seed {training_config['random_seed']}): {best_cost:.1f}")
    print("   🔄 Testing Performance:")
    
    if evaluation_results:
        # Calculate statistics for optimized solution
        opt_costs = [r['optimized']['cost'] for r in evaluation_results if r['optimized']['cost'] != float('inf')]
        opt_times = [r['optimized']['avg_time'] for r in evaluation_results]
        
        # Calculate statistics for baseline solution
        base_costs = [r['baseline']['cost'] for r in evaluation_results if r['baseline']['cost'] != float('inf')]
        base_times = [r['baseline']['avg_time'] for r in evaluation_results]
        
        # Calculate improvement statistics
        improvements = [r['improvement_percent'] for r in evaluation_results]
        
        if opt_costs and base_costs:
            avg_opt_cost = sum(opt_costs) / len(opt_costs)
            avg_base_cost = sum(base_costs) / len(base_costs)
            avg_improvement = sum(improvements) / len(improvements)
            
            print(f"      → Average optimized cost: {avg_opt_cost:.1f}")
            print(f"      → Average baseline cost: {avg_base_cost:.1f}")
            print(f"      → Average improvement over baseline: {avg_improvement:.1f}%")
            print()
            
            # Performance comparison against training
            if avg_opt_cost > best_cost:
                diff = avg_opt_cost - best_cost
                print(f"   📈 Solution is {diff:.1f} points worse on average on new seeds")
                print("      → This is normal - training optimizes for specific seed")
            else:
                diff = best_cost - avg_opt_cost  
                print(f"   📉 Solution is {diff:.1f} points better on average on new seeds")
                print("      → Excellent generalization!")
            
            # Overall baseline comparison
            if avg_improvement > 0:
                print(f"   🎯 Overall: Optimized solution is {avg_improvement:.1f}% better than baseline across all seeds")
            else:
                print(f"   ⚠️  Overall: Optimized solution is {abs(avg_improvement):.1f}% worse than baseline")
            
            print()
            print("   📋 Detailed Results (Optimized vs Baseline):")
            for i, eval_data in enumerate(evaluation_results, 1):
                seed = eval_data['seed']
                opt_cost = eval_data['optimized']['cost']
                base_cost = eval_data['baseline']['cost']
                improvement = eval_data['improvement_percent']
                opt_vehicles = eval_data['optimized']['vehicles']
                
                status = "✅" if opt_cost != float('inf') else "❌"
                comparison = f"({improvement:+.1f}%)" if improvement != 0 else "(same)"
                print(f"      {i}. Seed {seed}: {status} Opt {opt_cost:.1f} vs Base {base_cost:.1f} {comparison}, Vehicles {opt_vehicles}")
        else:
            print("      → No successful evaluations with valid costs")
    else:
        print("      → No successful evaluations completed")
    
    print()
    print("🎉 DEMONSTRATION COMPLETED!")
    print()
    print("💡 KEY INSIGHTS:")
    print("   ✅ Solution trained on one seed and saved for reuse")
    print("   ✅ Same solution evaluated on different traffic patterns")  
    print("   ✅ Performance compared against simple baseline (30s/4s)")
    print("   ✅ This validates robustness and actual improvement over simple timing")
    print("   ✅ Critical for real-world traffic light deployment")
    
    # Add baseline insights
    if evaluation_results:
        successful_improvements = [r['improvement_percent'] for r in evaluation_results if r['improvement_percent'] > 0]
        if len(successful_improvements) >= len(evaluation_results) * 0.75:  # 75% or more improved
            print("   🎯 EXCELLENT: Optimized solution beats baseline on most test seeds!")
        elif len(successful_improvements) >= len(evaluation_results) * 0.5:  # 50% or more improved
            print("   � GOOD: Optimized solution beats baseline on about half the test seeds")
        else:
            print("   ⚠️  WARNING: Optimized solution struggles to beat simple baseline")
            print("      → Consider adjusting ACO parameters or longer optimization")
    
    print()
    print(f"�📁 Saved solution file: {solution_file}")
    print("   → Use this file to evaluate the solution anytime")
    print("   → Load with: load_solution(file_path)")
    print("   → Evaluate with: evaluate_solution_with_new_seed(file_path, new_seed)")
    print("   → Compare baselines with: evaluate_baseline_on_seed(seed, config, phase_types)")

if __name__ == "__main__":
    main()
