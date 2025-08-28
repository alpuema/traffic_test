#!/usr/bin/env python3
"""
Sensitivity Analysis Wrapper for Traffic Light Optimization

This module provides easy-to-use wrapper functions for conducting sensitivity 
analysis on different optimization parameters. It supports parameter sweeps
across multiple dimensions and generates comprehensive reports.

Author: Traffic Optimization System  
Date: August 2025
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Dict, List, Any, Tuple, Optional, Union

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simplified_traffic import generate_network_and_routes
from optimization.simple_aco import run_simplified_aco_optimization


def run_sensitivity_analysis(
    parameter_ranges: Dict[str, List],
    base_config: Dict[str, Any],
    n_replications: int = 3,
    output_dir: str = None,
    parallel: bool = True,
    max_workers: int = None,
    show_individual_plots: bool = False,
    show_final_plot: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive sensitivity analysis on optimization parameters.
    
    Args:
        parameter_ranges: Dict of parameter names to lists of values to test
            Example: {'n_ants': [10, 20, 30], 'n_iterations': [5, 10, 15]}
        base_config: Base configuration dict with default values
        n_replications: Number of replications per parameter combination
        output_dir: Directory to save results (default: results/sensitivity_analysis)
        parallel: Whether to run parameter combinations in parallel
        max_workers: Maximum number of parallel workers
        show_individual_plots: Show plots for each individual optimization run
        show_final_plot: Show final summary plot after analysis completes
        
    Returns:
        Dictionary with analysis results and summary statistics
        
    Example:
        >>> base_config = {
        ...     'grid_size': 3,
        ...     'n_vehicles': 30,
        ...     'simulation_time': 600,
        ...     'traffic_pattern': 'commuter',
        ...     'n_ants': 20,
        ...     'n_iterations': 10
        ... }
        >>> param_ranges = {
        ...     'n_ants': [10, 20, 30, 40],
        ...     'n_iterations': [5, 10, 15]
        ... }
        >>> results = run_sensitivity_analysis(param_ranges, base_config)
    """
    
    print("🔬 SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("results", "sensitivity_analysis", f"analysis_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Generate all parameter combinations
    param_combinations = _generate_parameter_combinations(parameter_ranges)
    total_runs = len(param_combinations) * n_replications
    
    print(f"📊 Analysis Configuration:")
    print(f"   Parameters: {list(parameter_ranges.keys())}")
    print(f"   Combinations: {len(param_combinations)}")
    print(f"   Replications per combination: {n_replications}")
    print(f"   Total optimization runs: {total_runs}")
    print()
    
    # Run sensitivity analysis
    start_time = time.time()
    
    if parallel and len(param_combinations) > 1:
        print("🚀 Running analysis in parallel...")
        results = _run_parallel_analysis(
            param_combinations, base_config, n_replications, 
            output_dir, max_workers, show_individual_plots
        )
    else:
        print("🔄 Running analysis sequentially...")
        results = _run_sequential_analysis(
            param_combinations, base_config, n_replications, output_dir, show_individual_plots
        )
    
    analysis_time = time.time() - start_time
    
    # Generate summary statistics
    summary = _generate_analysis_summary(results, parameter_ranges, analysis_time)
    
    # Save results
    results_file = os.path.join(output_dir, "sensitivity_results.json")
    summary_file = os.path.join(output_dir, "analysis_summary.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate visualizations
    plot_files = _generate_sensitivity_plots(results, parameter_ranges, output_dir, show_final_plot)
    
    print(f"✅ Sensitivity analysis completed in {analysis_time:.1f} seconds")
    print(f"📊 Results saved to: {output_dir}")
    print(f"📈 Generated {len(plot_files)} visualization plots")
    
    return {
        'results': results,
        'summary': summary,
        'output_dir': output_dir,
        'plot_files': plot_files
    }


def run_simple_parameter_sweep(
    parameter_name: str,
    parameter_values: List,
    base_config: Dict[str, Any],
    n_replications: int = 5,
    show_individual_plots: bool = False,
    show_final_plot: bool = True
) -> Dict[str, Any]:
    """
    Simplified wrapper for single-parameter sensitivity analysis.
    
    Args:
        parameter_name: Name of parameter to vary
        parameter_values: List of values to test
        base_config: Base configuration
        n_replications: Number of replications per value
        show_individual_plots: Show plots for each individual optimization run
        show_final_plot: Show final summary plot after analysis completes
        
    Returns:
        Analysis results with statistics and plot
        
    Example:
        >>> config = {'grid_size': 3, 'n_vehicles': 30, 'simulation_time': 600}
        >>> results = run_simple_parameter_sweep('n_ants', [10, 20, 30], config)
    """
    
    return run_sensitivity_analysis(
        parameter_ranges={parameter_name: parameter_values},
        base_config=base_config,
        n_replications=n_replications,
        parallel=False,  # Simpler for single parameter
        show_individual_plots=show_individual_plots,
        show_final_plot=show_final_plot
    )


def _generate_parameter_combinations(parameter_ranges: Dict[str, List]) -> List[Dict]:
    """Generate all combinations of parameters to test."""
    import itertools
    
    param_names = list(parameter_ranges.keys())
    param_values = list(parameter_ranges.values())
    
    combinations = []
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        combinations.append(param_dict)
    
    return combinations


def _run_sequential_analysis(param_combinations, base_config, n_replications, output_dir, show_individual_plots=False):
    """Run sensitivity analysis sequentially."""
    results = []
    
    for i, params in enumerate(param_combinations):
        print(f"\n📋 Parameter combination {i+1}/{len(param_combinations)}: {params}")
        
        # Update base config with current parameters
        config = base_config.copy()
        config.update(params)
        
        combination_results = []
        
        for rep in range(n_replications):
            print(f"   Replication {rep+1}/{n_replications}...")
            
            # Add replication seed for reproducibility
            config['seed'] = hash(f"{params}_{rep}") % 10000
            
            try:
                result = _run_single_optimization(config, show_plots=show_individual_plots)
                result['replication'] = rep
                result['parameters'] = params.copy()
                combination_results.append(result)
                
            except Exception as e:
                print(f"   ❌ Replication {rep+1} failed: {e}")
                continue
        
        results.extend(combination_results)
    
    return results


def _run_parallel_analysis(param_combinations, base_config, n_replications, output_dir, max_workers, show_individual_plots=False):
    """Run sensitivity analysis in parallel."""
    
    # Prepare all individual runs
    run_configs = []
    for params in param_combinations:
        config = base_config.copy()
        config.update(params)
        
        for rep in range(n_replications):
            run_config = config.copy()
            run_config['seed'] = hash(f"{params}_{rep}") % 10000
            run_config['_meta'] = {
                'parameters': params.copy(),
                'replication': rep
            }
            run_configs.append(run_config)
    
    # Run in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_config = {
            executor.submit(_run_single_optimization, config): config 
            for config in run_configs
        }
        
        results = []
        completed = 0
        
        for future in future_to_config:
            try:
                result = future.result()
                config = future_to_config[future]
                
                result['replication'] = config['_meta']['replication']
                result['parameters'] = config['_meta']['parameters']
                results.append(result)
                
                completed += 1
                print(f"   ✅ Completed {completed}/{len(run_configs)} runs")
                
            except Exception as e:
                print(f"   ❌ Run failed: {e}")
                continue
    
    return results


def _run_single_optimization(config: Dict[str, Any], show_plots: bool = False) -> Dict[str, Any]:
    """Run a single optimization with error handling."""
    
    # Remove meta information if present
    clean_config = {k: v for k, v in config.items() if not k.startswith('_')}
    
    # Generate scenario
    scenario_result = generate_network_and_routes(
        grid_size=clean_config.get('grid_size', 3),
        n_vehicles=clean_config.get('n_vehicles', 30),
        sim_time=clean_config.get('simulation_time', 600),
        pattern=clean_config.get('traffic_pattern', 'balanced'),
        seed=clean_config.get('seed')
    )
    
    if not scenario_result['success']:
        raise Exception(f"Scenario generation failed: {scenario_result['error']}")
    
    # Run optimization with configurable plot display
    optimization_result = run_simplified_aco_optimization(clean_config, show_plots_override=show_plots)
    
    if not optimization_result['success']:
        raise Exception(f"Optimization failed: {optimization_result['error']}")
    
    # Return key metrics
    return {
        'success': True,
        'best_cost': optimization_result['best_cost'],
        'improvement_pct': optimization_result.get('improvement_pct', 0),
        'optimization_time': optimization_result.get('optimization_time', 0),
        'configuration': clean_config
    }


def _generate_analysis_summary(results: List[Dict], parameter_ranges: Dict, analysis_time: float) -> Dict:
    """Generate summary statistics from sensitivity analysis results."""
    
    if not results:
        return {'error': 'No successful results to analyze'}
    
    # Group results by parameter combination
    grouped_results = {}
    for result in results:
        param_key = str(sorted(result['parameters'].items()))
        if param_key not in grouped_results:
            grouped_results[param_key] = []
        grouped_results[param_key].append(result)
    
    # Calculate statistics for each parameter combination
    parameter_stats = {}
    for param_key, group_results in grouped_results.items():
        costs = [r['best_cost'] for r in group_results]
        improvements = [r['improvement_pct'] for r in group_results]
        times = [r['optimization_time'] for r in group_results]
        
        parameter_stats[param_key] = {
            'parameters': group_results[0]['parameters'],
            'n_replications': len(group_results),
            'cost_mean': np.mean(costs),
            'cost_std': np.std(costs),
            'cost_min': np.min(costs),
            'cost_max': np.max(costs),
            'improvement_mean': np.mean(improvements),
            'improvement_std': np.std(improvements),
            'time_mean': np.mean(times)
        }
    
    # Find best parameter combination
    best_combo = min(parameter_stats.items(), key=lambda x: x[1]['cost_mean'])
    
    summary = {
        'analysis_metadata': {
            'total_runs': len(results),
            'successful_runs': len([r for r in results if r['success']]),
            'analysis_time_seconds': analysis_time,
            'parameter_ranges': parameter_ranges
        },
        'best_configuration': {
            'parameters': best_combo[1]['parameters'],
            'mean_cost': best_combo[1]['cost_mean'],
            'std_cost': best_combo[1]['cost_std'],
            'mean_improvement': best_combo[1]['improvement_mean']
        },
        'parameter_statistics': parameter_stats,
        'overall_statistics': {
            'mean_cost_across_all': np.mean([r['best_cost'] for r in results]),
            'std_cost_across_all': np.std([r['best_cost'] for r in results]),
            'mean_improvement_across_all': np.mean([r['improvement_pct'] for r in results])
        }
    }
    
    return summary


def _generate_sensitivity_plots(results: List[Dict], parameter_ranges: Dict, output_dir: str) -> List[str]:
    """Generate visualization plots for sensitivity analysis."""
    
    plot_files = []
    
    # For each parameter, create individual plots
    for param_name in parameter_ranges.keys():
        try:
            plot_file = _create_parameter_plot(results, param_name, output_dir)
            if plot_file:
                plot_files.append(plot_file)
        except Exception as e:
            print(f"⚠️  Failed to create plot for {param_name}: {e}")
    
    # Create summary comparison plot if multiple parameters
    if len(parameter_ranges) > 1:
        try:
            summary_plot = _create_summary_plot(results, parameter_ranges, output_dir)
            if summary_plot:
                plot_files.append(summary_plot)
        except Exception as e:
            print(f"⚠️  Failed to create summary plot: {e}")
    
    return plot_files


def _create_parameter_plot(results: List[Dict], param_name: str, output_dir: str) -> str:
    """Create a plot showing the effect of a single parameter."""
    
    # Group results by parameter value
    param_groups = {}
    for result in results:
        if param_name in result['parameters']:
            param_value = result['parameters'][param_name]
            if param_value not in param_groups:
                param_groups[param_value] = []
            param_groups[param_value].append(result['best_cost'])
    
    if len(param_groups) < 2:
        return None  # Need at least 2 values to plot
    
    # Calculate statistics
    param_values = sorted(param_groups.keys())
    means = [np.mean(param_groups[val]) for val in param_values]
    stds = [np.std(param_groups[val]) for val in param_values]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(param_values, means, yerr=stds, marker='o', capsize=5, linewidth=2)
    plt.xlabel(f'{param_name.replace("_", " ").title()}')
    plt.ylabel('Average Travel Time (seconds)')
    plt.title(f'Sensitivity Analysis: {param_name.replace("_", " ").title()}')
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    best_idx = np.argmin(means)
    plt.annotate(f'Best: {param_values[best_idx]}\n({means[best_idx]:.1f}s)', 
                xy=(param_values[best_idx], means[best_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Save plot
    plot_file = os.path.join(output_dir, f'sensitivity_{param_name}.png')
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file


def _create_summary_plot(results: List[Dict], parameter_ranges: Dict, output_dir: str) -> str:
    """Create a summary plot comparing all parameters."""
    
    fig, axes = plt.subplots(1, len(parameter_ranges), figsize=(5*len(parameter_ranges), 6))
    if len(parameter_ranges) == 1:
        axes = [axes]
    
    for i, param_name in enumerate(parameter_ranges.keys()):
        # Group results by parameter value
        param_groups = {}
        for result in results:
            if param_name in result['parameters']:
                param_value = result['parameters'][param_name]
                if param_value not in param_groups:
                    param_groups[param_value] = []
                param_groups[param_value].append(result['best_cost'])
        
        if len(param_groups) >= 2:
            param_values = sorted(param_groups.keys())
            means = [np.mean(param_groups[val]) for val in param_values]
            stds = [np.std(param_groups[val]) for val in param_values]
            
            axes[i].errorbar(param_values, means, yerr=stds, marker='o', capsize=5)
            axes[i].set_xlabel(f'{param_name.replace("_", " ").title()}')
            axes[i].set_ylabel('Avg Travel Time (s)')
            axes[i].set_title(f'{param_name.replace("_", " ").title()}')
            axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Sensitivity Analysis Summary', fontsize=16)
    
    # Save plot
    plot_file = os.path.join(output_dir, 'sensitivity_summary.png')
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file


if __name__ == "__main__":
    # Example usage
    print("🔬 Sensitivity Analysis Module")
    print("Example usage:")
    print()
    
    example_code = '''
    from sensitivity_analysis import run_sensitivity_analysis, run_simple_parameter_sweep
    
    # Define base configuration
    base_config = {
        'grid_size': 3,
        'n_vehicles': 30, 
        'simulation_time': 600,
        'traffic_pattern': 'commuter'
    }
    
    # Multi-parameter analysis
    param_ranges = {
        'n_ants': [10, 20, 30],
        'n_iterations': [5, 10, 15]  
    }
    results = run_sensitivity_analysis(param_ranges, base_config, n_replications=3)
    
    # Single-parameter analysis  
    results = run_simple_parameter_sweep('n_ants', [5, 10, 20, 30], base_config)
    '''
    
    print(example_code)
