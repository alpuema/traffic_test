#!/usr/bin/env python3
"""
Clean Traffic Light Optimization Tool

This is the main optimization tool that combines simplified ACO algorithm
with flexible traffic pattern generation. It supports:

- Multiple traffic patterns (random, commuter, commercial, etc.)
- Seed-based reproducible optimization and evaluation
- Solution saving and loading for different scenarios
- Clean, easy-to-understand configuration

Author: Traffic Optimization System
Date: August 2025
"""

import sys
import os
import json
import argparse
from datetime import datetime

# Since we're now in src/, import directly from current directory
from optimization.simple_aco import run_simplified_aco_optimization
from simplified_traffic import (
    generate_network_and_routes, 
    save_optimized_solution, 
    load_solution,
    evaluate_solution_with_new_seed,
    list_available_patterns
)

# ============================================================================
# MAIN OPTIMIZATION INTERFACE
# ============================================================================

def run_complete_optimization(config):
    """
    Run complete optimization: generate scenario, optimize, and save results.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Results dictionary
    """
    print("🚀 TRAFFIC LIGHT OPTIMIZATION")
    print("=" * 50)
    
    # Step 1: Generate scenario with specified traffic pattern
    print("\n📋 Step 1: Generating Traffic Scenario")
    
    scenario_result = generate_network_and_routes(
        grid_size=config['grid_size'],
        n_vehicles=config['n_vehicles'], 
        sim_time=config['simulation_time'],
        pattern=config['traffic_pattern'],
        seed=config.get('seed')
    )
    
    if not scenario_result['success']:
        print(f"❌ Scenario generation failed: {scenario_result['error']}")
        return {'success': False, 'error': 'Scenario generation failed'}
    
    print(f"✅ Scenario generated with {config['traffic_pattern']} pattern")
    
    # Step 2: Run ACO optimization
    print("\n🐜 Step 2: Running ACO Optimization")
    
    aco_config = {
        'grid_size': config['grid_size'],
        'n_vehicles': config['n_vehicles'],
        'simulation_time': config['simulation_time'],
        'n_ants': config.get('n_ants', 20),
        'n_iterations': config.get('n_iterations', 10)
    }
    
    # Temporarily update paths for ACO to find generated files
    import src.optimization.simple_aco as aco
    original_values = {}
    for key, value in aco_config.items():
        if hasattr(aco, key.upper()):
            original_values[key.upper()] = getattr(aco, key.upper())
            setattr(aco, key.upper(), value)
    
    optimization_result = run_simplified_aco_optimization(aco_config)
    
    # Restore original values
    for key, value in original_values.items():
        setattr(aco, key, value)
    
    if not optimization_result['success']:
        print(f"❌ Optimization failed: {optimization_result['error']}")
        return {'success': False, 'error': 'Optimization failed'}
    
    print(f"✅ Optimization completed! Best cost: {optimization_result['best_cost']:.1f}")
    
    # Step 3: Save results
    print("\n💾 Step 3: Saving Results")
    
    # Prepare metadata
    metadata = {
        'grid_size': config['grid_size'],
        'n_vehicles': config['n_vehicles'],
        'simulation_time': config['simulation_time'],
        'traffic_pattern': config['traffic_pattern'],
        'seed': config.get('seed'),
        'n_ants': aco_config['n_ants'],
        'n_iterations': aco_config['n_iterations'],
        'optimization_duration': optimization_result['duration'],
        'best_cost': optimization_result['best_cost']
    }
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save optimized solution
    save_result = save_optimized_solution(
        optimization_result['best_solution'],
        metadata,
        results_dir
    )
    
    # Combine all results
    complete_result = {
        'success': True,
        'scenario': scenario_result,
        'optimization': optimization_result,
        'saved_solution': save_result,
        'metadata': metadata
    }
    
    print_summary(complete_result)
    
    return complete_result

def evaluate_existing_solution(solution_file, new_seed, config=None):
    """
    Evaluate an existing solution with a new traffic seed.
    
    Args:
        solution_file: Path to saved solution file
        new_seed: New random seed for traffic generation
        config: Optional configuration overrides
    
    Returns:
        Evaluation results
    """
    print("🔄 SOLUTION RE-EVALUATION")
    print("=" * 50)
    
    # Load existing solution
    solution_data = load_solution(solution_file)
    if not solution_data:
        return {'success': False, 'error': 'Could not load solution'}
    
    print(f"📂 Loaded solution from: {os.path.basename(solution_file)}")
    print(f"   Original seed: {solution_data['metadata'].get('seed')}")
    print(f"   New seed: {new_seed}")
    
    # Use configuration overrides or original values
    eval_config = solution_data['metadata'].copy()
    if config:
        eval_config.update(config)
    
    # Generate new scenario
    scenario_result = generate_network_and_routes(
        grid_size=eval_config['grid_size'],
        n_vehicles=eval_config.get('n_vehicles', 30),
        sim_time=eval_config.get('simulation_time', 600),
        pattern=eval_config.get('traffic_pattern', 'commuter'),
        seed=new_seed
    )
    
    if not scenario_result['success']:
        return {'success': False, 'error': 'New scenario generation failed'}
    
    print(f"✅ New scenario generated with seed {new_seed}")
    
    # Here you would apply the solution and evaluate performance
    # For now, return the setup information
    
    result = {
        'success': True,
        'original_solution': solution_data,
        'new_scenario': scenario_result,
        'evaluation_seed': new_seed,
        'message': 'Solution loaded and new scenario generated. Ready for evaluation.'
    }
    
    return result

def print_summary(results):
    """Print a summary of optimization results."""
    print("\n📊 OPTIMIZATION SUMMARY")
    print("=" * 50)
    
    metadata = results['metadata']
    
    print(f"🏗️  Scenario: {metadata['grid_size']}x{metadata['grid_size']} grid")
    print(f"🚗 Vehicles: {metadata['n_vehicles']}")
    print(f"⏰ Simulation: {metadata['simulation_time']}s")
    print(f"🚦 Pattern: {metadata['traffic_pattern']}")
    print(f"🎲 Seed: {metadata.get('seed', 'None')}")
    print()
    print(f"🐜 ACO Config: {metadata['n_ants']} ants × {metadata['n_iterations']} iterations")
    print(f"⏱️  Duration: {metadata['optimization_duration']:.1f} seconds")
    print(f"🎯 Best Cost: {metadata['best_cost']:.1f}")
    print()
    print(f"💾 Solution saved: {results['saved_solution']['solution_file']}")
    print()
    print("🎉 Optimization completed successfully!")

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main command line interface."""
    parser = argparse.ArgumentParser(
        description='Clean Traffic Light Optimization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize.py --grid 3 --vehicles 30 --pattern commuter
  python optimize.py --evaluate solution_commuter_20250827.json --new-seed 123
  python optimize.py --list-patterns
        """
    )
    
    # Main operation modes
    parser.add_argument('--optimize', action='store_true', help='Run optimization (default)')
    parser.add_argument('--evaluate', type=str, help='Evaluate existing solution file with new seed')
    parser.add_argument('--list-patterns', action='store_true', help='List available traffic patterns')
    
    # Scenario configuration
    parser.add_argument('--grid', type=int, default=3, help='Grid size (default: 3)')
    parser.add_argument('--vehicles', type=int, default=30, help='Number of vehicles (default: 30)')
    parser.add_argument('--time', type=int, default=600, help='Simulation time in seconds (default: 600)')
    parser.add_argument('--pattern', type=str, default='commuter', 
                       help='Traffic pattern (default: commuter)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible results')
    
    # ACO configuration
    parser.add_argument('--ants', type=int, default=20, help='Number of ants (default: 20)')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations (default: 10)')
    
    # Evaluation options
    parser.add_argument('--new-seed', type=int, help='New seed for solution evaluation')
    
    args = parser.parse_args()
    
    # Handle different operation modes
    if args.list_patterns:
        list_available_patterns()
        return
    
    elif args.evaluate:
        if not args.new_seed:
            print("❌ Error: --new-seed is required for evaluation mode")
            return
        
        config_overrides = {}
        if args.vehicles != 30:
            config_overrides['n_vehicles'] = args.vehicles
        if args.time != 600:
            config_overrides['simulation_time'] = args.time
        
        result = evaluate_existing_solution(args.evaluate, args.new_seed, config_overrides)
        
        if result['success']:
            print("✅ Evaluation setup completed successfully!")
        else:
            print(f"❌ Evaluation failed: {result['error']}")
    
    else:
        # Default: run optimization
        config = {
            'grid_size': args.grid,
            'n_vehicles': args.vehicles,
            'simulation_time': args.time,
            'traffic_pattern': args.pattern,
            'seed': args.seed,
            'n_ants': args.ants,
            'n_iterations': args.iterations
        }
        
        result = run_complete_optimization(config)
        
        if not result['success']:
            print(f"❌ Optimization failed: {result['error']}")
            sys.exit(1)

if __name__ == "__main__":
    main()
