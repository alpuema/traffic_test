#!/usr/bin/env python3
"""
Quick Load & Test Example

This example shows how to load an existing saved solution
and test it on a different traffic seed. Perfect for validating
that your optimized traffic lights work well on new traffic patterns.

Usage: python load_and_test_example.py <solution_file> <test_seed>
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simplified_traffic import load_solution, evaluate_solution_with_new_seed
from src.optimize import evaluate_existing_solution

def main():
    if len(sys.argv) < 2:
        # Find the most recent solution file
        solutions_dir = "results/final_solutions"
        if os.path.exists(solutions_dir):
            solution_files = [f for f in os.listdir(solutions_dir) if f.endswith('.json')]
            if solution_files:
                # Use the most recent solution
                solution_files.sort(reverse=True)
                solution_file = os.path.join(solutions_dir, solution_files[0])
                test_seed = 999  # Default test seed
                print(f"🔍 Using most recent solution: {os.path.basename(solution_file)}")
                print(f"🎲 Testing with seed: {test_seed}")
            else:
                print("❌ No solution files found. Run train_evaluate.py or save_and_evaluate_example.py first.")
                return
        else:
            print("❌ No solutions directory found. Run train_evaluate.py or save_and_evaluate_example.py first.")
            return
    else:
        solution_file = sys.argv[1]
        test_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 999
    
    print("🎯 QUICK LOAD & TEST DEMO")
    print("=" * 40)
    print()
    
    # Load and display solution info
    print("📂 Loading solution...")
    solution_data = load_solution(solution_file)
    if not solution_data:
        print("❌ Failed to load solution file")
        return
    
    metadata = solution_data['metadata']
    print(f"✅ Solution loaded successfully!")
    print(f"   📋 Original Training:")
    print(f"      • Pattern: {metadata.get('pattern', 'unknown')}")
    print(f"      • Training Seed: {metadata.get('seed', 'unknown')}")
    print(f"      • Grid: {metadata.get('grid_size', '?')}x{metadata.get('grid_size', '?')}")
    print(f"      • Vehicles: {metadata.get('n_vehicles', '?')}")
    print(f"      • Best Cost: {metadata.get('best_cost', '?'):.1f}")
    print()
    
    # Test on different seed
    print(f"🔄 Testing on new seed {test_seed}...")
    eval_result = evaluate_solution_with_new_seed(
        solution_file=solution_file,
        new_seed=test_seed,
        n_vehicles=metadata.get('n_vehicles'),
        sim_time=metadata.get('simulation_time')
    )
    
    if eval_result['success']:
        # Extract performance metrics
        cost = eval_result.get('cost', float('inf'))
        avg_time = eval_result.get('avg_travel_time', 0)
        metrics = eval_result.get('metrics', {})
        vehicles_completed = metrics.get('vehicles', 0)
        expected_vehicles = metadata.get('n_vehicles', 0)
        
        print("✅ Cross-seed evaluation completed!")
        print(f"   📊 Performance Results:")
        print(f"      • Test Cost: {cost:.1f}")
        print(f"      • Original Training Cost: {metadata.get('best_cost', '?'):.1f}")
        print(f"      • Average Travel Time: {avg_time:.1f}s")
        print(f"      • Vehicles Completed: {vehicles_completed}/{expected_vehicles}")
        
        # Performance comparison
        original_cost = metadata.get('best_cost', float('inf'))
        if cost != float('inf') and original_cost != float('inf'):
            if cost > original_cost:
                diff = cost - original_cost
                print(f"      • Performance: {diff:.1f} points worse than training")
                print("        (Normal - different traffic pattern)")
            else:
                diff = original_cost - cost
                print(f"      • Performance: {diff:.1f} points better than training")
                print("        (Excellent generalization!)")
        
        print()
        print("✅ Same traffic light timings work on different traffic pattern!")
    else:
        print(f"❌ Evaluation failed: {eval_result.get('error', 'Unknown error')}")
        return
    
    print()
    print("🎉 LOAD & TEST COMPLETED!")
    print()
    print("💡 What happened:")
    print("   1. Loaded optimized traffic light settings from file")
    print("   2. Generated new traffic scenario with different seed")
    print("   3. Applied same light timings to new traffic pattern")
    print("   4. Ready to compare performance across seeds")
    print()
    print("🔧 Advanced usage:")
    print("   • Use evaluate_existing_solution() for more options")
    print("   • Compare costs across multiple seeds")
    print("   • Validate solution robustness")

if __name__ == "__main__":
    main()
