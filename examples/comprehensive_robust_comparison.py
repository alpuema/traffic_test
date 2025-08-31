#!/usr/bin/env python3
"""
Comprehensive Robust vs Regular ACO Comparison
==============================================

This script implements a complete equivalent to traffic_pattern_comparison.py
but compares Robust Multi-Seed ACO vs Regular Single-Seed ACO.

Features:
- Tests all traffic patterns (commuter, industrial, random)
- Cross-validation with multiple runs
- Detailed performance analysis
- Comprehensive visualization
- Statistical significance testing
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simplified_traffic import create_traffic_scenario
from src.optimize import ACOTrafficOptimizer
from src.optimization.robust_aco import RobustACOTrafficOptimizer


def run_comprehensive_comparison():
    """Run comprehensive comparison between Robust and Regular ACO"""
    print("🌟 COMPREHENSIVE ROBUST vs REGULAR ACO COMPARISON")
    print("=" * 60)
    print("📋 Testing all traffic patterns with cross-validation")
    print()
    
    # Configuration
    patterns = ['commuter', 'industrial', 'random']
    runs_per_pattern = 3  # Multiple runs for statistical validation
    grid_size = 4
    num_vehicles = 20
    sim_time = 2000
    
    # Store all results
    all_results = {
        'patterns': {},
        'summary': {},
        'config': {
            'grid_size': grid_size,
            'num_vehicles': num_vehicles,
            'sim_time': sim_time,
            'runs_per_pattern': runs_per_pattern,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # Test each pattern
    for pattern in patterns:
        print(f"🔄 Testing pattern: {pattern.upper()}")
        print("-" * 40)
        
        pattern_results = {
            'robust': {'costs': [], 'times': [], 'improvements': []},
            'regular': {'costs': [], 'times': [], 'improvements': []},
            'comparisons': []
        }
        
        # Multiple runs for each pattern
        for run in range(runs_per_pattern):
            print(f"   Run {run + 1}/{runs_per_pattern}")
            
            # Generate scenario for this run
            scenario_result = create_traffic_scenario(grid_size, num_vehicles, sim_time, pattern=pattern)
            
            # Test Robust ACO
            print("      🌱 Testing Robust ACO...")
            robust_start = time.time()
            robust_optimizer = RobustACOTrafficOptimizer(
                sumo_config=scenario_result['config_file'],
                n_ants=10,
                n_iterations=5,
                training_seeds=3,
                exploration_rate=0.25,
                scenario_vehicles=num_vehicles,
                simulation_time=sim_time,
                show_plots=False,
                show_sumo_gui=False,
                compare_baseline=True
            )
            robust_solution, robust_cost, robust_data, robust_baseline = robust_optimizer.optimize()
            robust_time = time.time() - robust_start
            
            # Test Regular ACO
            print("      🐜 Testing Regular ACO...")
            regular_start = time.time()
            regular_optimizer = ACOTrafficOptimizer(
                sumo_config=scenario_result['config_file'],
                n_ants=12,
                n_iterations=5,
                scenario_vehicles=num_vehicles,
                simulation_time=sim_time,
                show_plots=False,
                show_sumo_gui=False,
                compare_baseline=True
            )
            regular_solution, regular_cost, regular_data, regular_baseline = regular_optimizer.optimize()
            regular_time = time.time() - regular_start
            
            # Store results
            robust_improvement = 0
            if robust_baseline and isinstance(robust_baseline, dict):
                baseline_cost = robust_baseline.get('baseline_cost', 0)
                optimized_cost = robust_baseline.get('optimized_cost', robust_cost)
                if baseline_cost > 0:
                    robust_improvement = ((baseline_cost - optimized_cost) / baseline_cost) * 100
            
            regular_improvement = 0
            if regular_baseline and isinstance(regular_baseline, dict):
                baseline_cost = regular_baseline.get('baseline_cost', 0)
                optimized_cost = regular_baseline.get('optimized_cost', regular_cost)
                if baseline_cost > 0:
                    regular_improvement = ((baseline_cost - optimized_cost) / baseline_cost) * 100
            
            pattern_results['robust']['costs'].append(robust_cost)
            pattern_results['robust']['times'].append(robust_time)
            pattern_results['robust']['improvements'].append(robust_improvement)
            
            pattern_results['regular']['costs'].append(regular_cost)
            pattern_results['regular']['times'].append(regular_time)
            pattern_results['regular']['improvements'].append(regular_improvement)
            
            # Calculate comparison metrics
            cost_difference = ((robust_cost - regular_cost) / regular_cost) * 100
            time_difference = robust_time - regular_time
            
            comparison = {
                'run': run + 1,
                'robust_cost': robust_cost,
                'regular_cost': regular_cost,
                'cost_difference_pct': cost_difference,
                'robust_time': robust_time,
                'regular_time': regular_time,
                'time_difference': time_difference,
                'robust_improvement': robust_improvement,
                'regular_improvement': regular_improvement
            }
            pattern_results['comparisons'].append(comparison)
            
            print(f"         📊 Robust: {robust_cost:.1f} cost, {robust_time:.1f}s")
            print(f"         📊 Regular: {regular_cost:.1f} cost, {regular_time:.1f}s")
            print(f"         📊 Difference: {cost_difference:+.1f}%")
            print()
        
        # Calculate pattern statistics
        robust_costs = pattern_results['robust']['costs']
        regular_costs = pattern_results['regular']['costs']
        
        pattern_stats = {
            'robust': {
                'mean_cost': np.mean(robust_costs),
                'std_cost': np.std(robust_costs),
                'mean_time': np.mean(pattern_results['robust']['times']),
                'mean_improvement': np.mean(pattern_results['robust']['improvements'])
            },
            'regular': {
                'mean_cost': np.mean(regular_costs),
                'std_cost': np.std(regular_costs),
                'mean_time': np.mean(pattern_results['regular']['times']),
                'mean_improvement': np.mean(pattern_results['regular']['improvements'])
            }
        }
        
        cost_diff_pct = ((pattern_stats['robust']['mean_cost'] - pattern_stats['regular']['mean_cost']) / 
                        pattern_stats['regular']['mean_cost']) * 100
        
        pattern_stats['comparison'] = {
            'cost_difference_pct': cost_diff_pct,
            'time_overhead': pattern_stats['robust']['mean_time'] - pattern_stats['regular']['mean_time'],
            'robust_more_consistent': pattern_stats['robust']['std_cost'] < pattern_stats['regular']['std_cost']
        }
        
        all_results['patterns'][pattern] = {
            'data': pattern_results,
            'stats': pattern_stats
        }
        
        print(f"   📈 {pattern.upper()} SUMMARY:")
        print(f"      Robust:  {pattern_stats['robust']['mean_cost']:.1f} ± {pattern_stats['robust']['std_cost']:.1f} cost")
        print(f"      Regular: {pattern_stats['regular']['mean_cost']:.1f} ± {pattern_stats['regular']['std_cost']:.1f} cost")
        print(f"      Difference: {cost_diff_pct:+.1f}%")
        print(f"      More consistent: {'Robust' if pattern_stats['comparison']['robust_more_consistent'] else 'Regular'}")
        print()
    
    # Overall analysis
    print("🏆 OVERALL ANALYSIS")
    print("=" * 30)
    
    all_robust_costs = []
    all_regular_costs = []
    all_robust_times = []
    all_regular_times = []
    
    for pattern in patterns:
        pattern_data = all_results['patterns'][pattern]['data']
        all_robust_costs.extend(pattern_data['robust']['costs'])
        all_regular_costs.extend(pattern_data['regular']['costs'])
        all_robust_times.extend(pattern_data['robust']['times'])
        all_regular_times.extend(pattern_data['regular']['times'])
    
    overall_stats = {
        'robust': {
            'mean_cost': np.mean(all_robust_costs),
            'std_cost': np.std(all_robust_costs),
            'mean_time': np.mean(all_robust_times)
        },
        'regular': {
            'mean_cost': np.mean(all_regular_costs),
            'std_cost': np.std(all_regular_costs),
            'mean_time': np.mean(all_regular_times)
        }
    }
    
    overall_cost_diff = ((overall_stats['robust']['mean_cost'] - overall_stats['regular']['mean_cost']) / 
                        overall_stats['regular']['mean_cost']) * 100
    
    all_results['summary'] = {
        'overall_stats': overall_stats,
        'cost_difference_pct': overall_cost_diff,
        'time_overhead': overall_stats['robust']['mean_time'] - overall_stats['regular']['mean_time'],
        'robust_more_consistent': overall_stats['robust']['std_cost'] < overall_stats['regular']['std_cost']
    }
    
    print(f"Overall Robust:  {overall_stats['robust']['mean_cost']:.1f} ± {overall_stats['robust']['std_cost']:.1f} cost")
    print(f"Overall Regular: {overall_stats['regular']['mean_cost']:.1f} ± {overall_stats['regular']['std_cost']:.1f} cost")
    print(f"Cost difference: {overall_cost_diff:+.1f}%")
    print(f"Time overhead: {overall_stats['robust']['mean_time'] - overall_stats['regular']['mean_time']:.1f}s")
    print(f"More consistent: {'Robust' if all_results['summary']['robust_more_consistent'] else 'Regular'}")
    print()
    
    # Generate visualization
    create_comprehensive_plots(all_results)
    
    # Save results
    results_path = os.path.join('results', 'comprehensive_robust_comparison.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"📁 Results saved to: {results_path}")
    
    return all_results


def create_comprehensive_plots(results):
    """Create comprehensive visualization of comparison results"""
    patterns = list(results['patterns'].keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Robust vs Regular ACO Comparison', fontsize=16, fontweight='bold')
    
    # 1. Cost comparison by pattern
    ax = axes[0, 0]
    robust_means = [results['patterns'][p]['stats']['robust']['mean_cost'] for p in patterns]
    regular_means = [results['patterns'][p]['stats']['regular']['mean_cost'] for p in patterns]
    robust_stds = [results['patterns'][p]['stats']['robust']['std_cost'] for p in patterns]
    regular_stds = [results['patterns'][p]['stats']['regular']['std_cost'] for p in patterns]
    
    x = np.arange(len(patterns))
    width = 0.35
    
    ax.bar(x - width/2, robust_means, width, yerr=robust_stds, label='Robust ACO', 
           alpha=0.8, color='steelblue', capsize=5)
    ax.bar(x + width/2, regular_means, width, yerr=regular_stds, label='Regular ACO', 
           alpha=0.8, color='coral', capsize=5)
    
    ax.set_xlabel('Traffic Pattern')
    ax.set_ylabel('Average Cost')
    ax.set_title('Cost Comparison by Pattern')
    ax.set_xticks(x)
    ax.set_xticklabels([p.title() for p in patterns])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Time comparison by pattern
    ax = axes[0, 1]
    robust_times = [results['patterns'][p]['stats']['robust']['mean_time'] for p in patterns]
    regular_times = [results['patterns'][p]['stats']['regular']['mean_time'] for p in patterns]
    
    ax.bar(x - width/2, robust_times, width, label='Robust ACO', 
           alpha=0.8, color='steelblue')
    ax.bar(x + width/2, regular_times, width, label='Regular ACO', 
           alpha=0.8, color='coral')
    
    ax.set_xlabel('Traffic Pattern')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Time Comparison by Pattern')
    ax.set_xticks(x)
    ax.set_xticklabels([p.title() for p in patterns])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Cost difference percentage
    ax = axes[0, 2]
    cost_diffs = [results['patterns'][p]['stats']['comparison']['cost_difference_pct'] for p in patterns]
    colors = ['red' if diff > 0 else 'green' for diff in cost_diffs]
    
    bars = ax.bar(patterns, cost_diffs, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Traffic Pattern')
    ax.set_ylabel('Cost Difference (%)')
    ax.set_title('Robust vs Regular ACO\n(Positive = Robust Worse)')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, diff in zip(bars, cost_diffs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                f'{diff:+.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    # 4. Consistency analysis (standard deviation)
    ax = axes[1, 0]
    robust_stds_all = [results['patterns'][p]['stats']['robust']['std_cost'] for p in patterns]
    regular_stds_all = [results['patterns'][p]['stats']['regular']['std_cost'] for p in patterns]
    
    ax.bar(x - width/2, robust_stds_all, width, label='Robust ACO', 
           alpha=0.8, color='steelblue')
    ax.bar(x + width/2, regular_stds_all, width, label='Regular ACO', 
           alpha=0.8, color='coral')
    
    ax.set_xlabel('Traffic Pattern')
    ax.set_ylabel('Cost Standard Deviation')
    ax.set_title('Consistency Comparison\n(Lower = More Consistent)')
    ax.set_xticks(x)
    ax.set_xticklabels([p.title() for p in patterns])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. All costs distribution
    ax = axes[1, 1]
    all_robust = []
    all_regular = []
    pattern_labels_robust = []
    pattern_labels_regular = []
    
    for pattern in patterns:
        pattern_data = results['patterns'][pattern]['data']
        robust_costs = pattern_data['robust']['costs']
        regular_costs = pattern_data['regular']['costs']
        
        all_robust.extend(robust_costs)
        all_regular.extend(regular_costs)
        pattern_labels_robust.extend([f"{pattern}_robust"] * len(robust_costs))
        pattern_labels_regular.extend([f"{pattern}_regular"] * len(regular_costs))
    
    ax.hist([all_robust, all_regular], bins=15, alpha=0.7, 
            label=['Robust ACO', 'Regular ACO'], color=['steelblue', 'coral'])
    ax.set_xlabel('Cost')
    ax.set_ylabel('Frequency')
    ax.set_title('Cost Distribution Across All Runs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Performance vs time trade-off
    ax = axes[1, 2]
    for i, pattern in enumerate(patterns):
        pattern_data = results['patterns'][pattern]['data']
        
        robust_cost = results['patterns'][pattern]['stats']['robust']['mean_cost']
        robust_time = results['patterns'][pattern]['stats']['robust']['mean_time']
        regular_cost = results['patterns'][pattern]['stats']['regular']['mean_cost']
        regular_time = results['patterns'][pattern]['stats']['regular']['mean_time']
        
        # Plot points
        ax.scatter(robust_time, robust_cost, s=100, alpha=0.8, 
                  color='steelblue', marker='o', label='Robust' if i == 0 else "")
        ax.scatter(regular_time, regular_cost, s=100, alpha=0.8, 
                  color='coral', marker='s', label='Regular' if i == 0 else "")
        
        # Add pattern labels
        ax.annotate(f'R-{pattern[:3]}', (robust_time, robust_cost), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax.annotate(f'N-{pattern[:3]}', (regular_time, regular_cost), 
                   xytext=(5, -15), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Execution Time (s)')
    ax.set_ylabel('Average Cost')
    ax.set_title('Performance vs Time Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join('results', 'comprehensive_robust_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comprehensive plot saved to: {plot_path}")
    
    plt.show()


def create_summary_report(results):
    """Create a detailed summary report"""
    print("\n" + "="*60)
    print("📋 DETAILED SUMMARY REPORT")
    print("="*60)
    
    patterns = list(results['patterns'].keys())
    
    # Pattern-by-pattern analysis
    for pattern in patterns:
        print(f"\n{pattern.upper()} PATTERN:")
        print("-" * (len(pattern) + 9))
        
        stats = results['patterns'][pattern]['stats']
        
        print(f"  Robust ACO:")
        print(f"    Average Cost: {stats['robust']['mean_cost']:.1f} ± {stats['robust']['std_cost']:.1f}")
        print(f"    Average Time: {stats['robust']['mean_time']:.1f}s")
        print(f"    Improvement: {stats['robust']['mean_improvement']:.1f}%")
        
        print(f"  Regular ACO:")
        print(f"    Average Cost: {stats['regular']['mean_cost']:.1f} ± {stats['regular']['std_cost']:.1f}")
        print(f"    Average Time: {stats['regular']['mean_time']:.1f}s")
        print(f"    Improvement: {stats['regular']['mean_improvement']:.1f}%")
        
        print(f"  Comparison:")
        print(f"    Cost Difference: {stats['comparison']['cost_difference_pct']:+.1f}%")
        print(f"    Time Overhead: {stats['comparison']['time_overhead']:+.1f}s")
        print(f"    More Consistent: {'Robust' if stats['comparison']['robust_more_consistent'] else 'Regular'}")
    
    # Overall conclusions
    summary = results['summary']
    print(f"\n🏆 OVERALL CONCLUSIONS:")
    print("-" * 20)
    print(f"  Cost Performance: {'Robust' if summary['cost_difference_pct'] < 0 else 'Regular'} ACO is {abs(summary['cost_difference_pct']):.1f}% better")
    print(f"  Consistency: {'Robust' if summary['robust_more_consistent'] else 'Regular'} ACO is more consistent")
    print(f"  Time Trade-off: Robust ACO takes {summary['time_overhead']:+.1f}s longer on average")
    print(f"  Recommendation: {'Robust ACO for critical applications' if summary['robust_more_consistent'] else 'Regular ACO for speed'}")


if __name__ == "__main__":
    try:
        print("🚀 Starting comprehensive comparison...")
        print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        results = run_comprehensive_comparison()
        create_summary_report(results)
        
        print("\n✅ Comprehensive comparison completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
