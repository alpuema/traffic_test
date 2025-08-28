#!/usr/bin/env python3
"""
Traffic Light Optimization: Train and Evaluate System

This script provides a clean interface for:
1. Training: Finding optimal traffic light settings for a specific scenario/seed
2. Evaluation: Testing those settings on different scenarios/seeds
3. Comparison: Comparing optimized vs default performance

Author: Alfonso Rato
Date: August 2025
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import Config
import src.optimization.simple_aco as aco
import numpy as np

class TrafficOptimizer:
    """Main class for training and evaluating traffic light optimization."""
    
    def __init__(self, config_file=None):
        """Initialize the optimizer with configuration."""
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Default configuration
        self.config = {
            # Simulation parameters
            "grid_size": 3,
            "n_vehicles": 50,
            "simulation_time": 1200,
            "traffic_pattern": "commuter",
            
            # ACO parameters
            "n_ants": 50,
            "n_iterations": 10,
            "evaporation_rate": 0.3,
            "alpha": 30.0,
            "coordination_factor": 0.2,
            
            # System parameters
            "use_traffic_engineering": True,
            "use_default_start": True,
            "use_coordination": True,
            "random_seed": 42,
            "verbose": True
        }
        
        # Load custom config if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                custom_config = json.load(f)
                self.config.update(custom_config)
    
    def save_config(self, filename):
        """Save current configuration to file."""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"💾 Configuration saved to {filename}")
    
    def train(self, scenario_name="default", custom_seed=None):
        """
        Train: Find optimal traffic light settings for a specific scenario.
        
        Args:
            scenario_name: Name for this training scenario
            custom_seed: Optional random seed override
            
        Returns:
            dict: Training results including optimized settings
        """
        print("🎓 TRAINING PHASE")
        print("=" * 60)
        
        # Set up training configuration
        training_config = self.config.copy()
        if custom_seed is not None:
            training_config["random_seed"] = custom_seed
        
        # Configure random seed
        np.random.seed(training_config["random_seed"])
        
        print(f"📋 Training Configuration:")
        print(f"   Scenario: {scenario_name}")
        print(f"   Grid Size: {training_config['grid_size']}x{training_config['grid_size']}")
        print(f"   Vehicles: {training_config['n_vehicles']}")
        print(f"   Simulation Time: {training_config['simulation_time']}s")
        print(f"   Traffic Pattern: {training_config['traffic_pattern']}")
        print(f"   Random Seed: {training_config['random_seed']}")
        print(f"   ACO: {training_config['n_ants']} ants, {training_config['n_iterations']} iterations")
        
        # Configure system
        config = Config()
        config.grid_size = training_config["grid_size"]
        config.n_vehicles = training_config["n_vehicles"]
        config.simulation_time = training_config["simulation_time"]
        config.verbose = 2 if training_config["verbose"] else 1
        config.set_traffic_pattern(training_config["traffic_pattern"])
        
        # Apply ACO configuration
        original_values = self._backup_aco_config()
        self._apply_aco_config(training_config)
        
        try:
            # Run training optimization
            start_time = time.time()
            print(f"\n🐜 Starting ACO Training...")
            
            results = aco.run_simplified_aco_optimization()
            
            training_time = time.time() - start_time
            
            if results and results.get('success', False):
                print(f"\n🎉 Training completed successfully!")
                print(f"   Training time: {training_time:.1f} seconds")
                print(f"   Improvement over default: {results['improvement']:.1f}%")
                
                # Save training results
                training_results = self._save_training_results(scenario_name, training_config, training_time, results)
                return training_results
            else:
                print(f"\n❌ Training failed!")
                return None
                
        finally:
            # Restore original ACO configuration
            self._restore_aco_config(original_values)
    
    def evaluate(self, trained_settings_file, eval_scenarios=None, eval_seeds=None):
        """
        Evaluate: Test trained settings on different scenarios/seeds.
        
        Args:
            trained_settings_file: Path to saved training results
            eval_scenarios: List of scenario names to test (default: same as training)
            eval_seeds: List of seeds to test with (default: [1, 2, 3, 4, 5])
            
        Returns:
            dict: Evaluation results
        """
        print("\n🔬 EVALUATION PHASE")
        print("=" * 60)
        
        # Load trained settings
        if not os.path.exists(trained_settings_file):
            print(f"❌ Training results file not found: {trained_settings_file}")
            return None
        
        with open(trained_settings_file, 'r') as f:
            training_results = json.load(f)
        
        optimized_settings = training_results["optimized_settings"]
        training_config = training_results["training_config"]
        
        print(f"📂 Loaded trained settings from: {trained_settings_file}")
        print(f"   Trained on: {training_results['scenario_name']}")
        print(f"   Training seed: {training_config['random_seed']}")
        print(f"   Optimized phases: {len(optimized_settings)} durations")
        
        # Default evaluation scenarios and seeds
        if eval_scenarios is None:
            eval_scenarios = [training_config["traffic_pattern"]]
        if eval_seeds is None:
            eval_seeds = [1, 2, 3, 4, 5]
        
        print(f"\n🎯 Evaluation Plan:")
        print(f"   Scenarios: {eval_scenarios}")
        print(f"   Seeds: {eval_seeds}")
        
        evaluation_results = {
            "training_file": trained_settings_file,
            "evaluation_timestamp": datetime.now().isoformat(),
            "scenarios": {},
            "summary": {}
        }
        
        # Test each scenario/seed combination
        for scenario in eval_scenarios:
            print(f"\n📊 Evaluating scenario: {scenario}")
            scenario_results = {"seeds": {}, "stats": {}}
            
            for seed in eval_seeds:
                print(f"   🎲 Testing seed {seed}...")
                
                # Set random seed
                np.random.seed(seed)
                
                # Configure system
                config = Config()
                config.grid_size = training_config["grid_size"]
                config.n_vehicles = training_config["n_vehicles"]
                config.simulation_time = training_config["simulation_time"]
                config.verbose = 1  # Quiet mode for evaluation
                config.set_traffic_pattern(scenario)
                
                # Test optimized settings
                optimized_perf = self._evaluate_settings(optimized_settings, config, f"eval_opt_{scenario}_{seed}")
                
                # Test default settings  
                default_settings = aco.get_default_durations(aco.get_file_paths()['net_file'])
                default_perf = self._evaluate_settings(default_settings, config, f"eval_def_{scenario}_{seed}")
                
                # Calculate improvement
                improvement = ((default_perf["total_time"] - optimized_perf["total_time"]) / 
                              default_perf["total_time"] * 100) if default_perf["total_time"] > 0 else 0
                
                seed_results = {
                    "seed": seed,
                    "optimized": optimized_perf,
                    "default": default_perf,
                    "improvement_percent": improvement
                }
                
                scenario_results["seeds"][str(seed)] = seed_results
                
                print(f"      Optimized: {optimized_perf['total_time']:.1f}s total, {optimized_perf['max_stop']:.1f}s max stop")
                print(f"      Default:   {default_perf['total_time']:.1f}s total, {default_perf['max_stop']:.1f}s max stop")
                print(f"      Improvement: {improvement:+.1f}%")
            
            # Calculate scenario statistics
            improvements = [result["improvement_percent"] for result in scenario_results["seeds"].values()]
            scenario_results["stats"] = {
                "mean_improvement": np.mean(improvements),
                "std_improvement": np.std(improvements),
                "min_improvement": np.min(improvements),
                "max_improvement": np.max(improvements),
                "positive_results": sum(1 for imp in improvements if imp > 0)
            }
            
            evaluation_results["scenarios"][scenario] = scenario_results
            
            print(f"   📈 Scenario Summary:")
            print(f"      Mean improvement: {scenario_results['stats']['mean_improvement']:.2f}% ± {scenario_results['stats']['std_improvement']:.2f}%")
            print(f"      Range: {scenario_results['stats']['min_improvement']:.1f}% to {scenario_results['stats']['max_improvement']:.1f}%")
            print(f"      Positive results: {scenario_results['stats']['positive_results']}/{len(eval_seeds)}")
        
        # Overall summary
        all_improvements = []
        for scenario_results in evaluation_results["scenarios"].values():
            all_improvements.extend([result["improvement_percent"] for result in scenario_results["seeds"].values()])
        
        evaluation_results["summary"] = {
            "total_tests": len(all_improvements),
            "overall_mean_improvement": np.mean(all_improvements),
            "overall_std_improvement": np.std(all_improvements),
            "success_rate": sum(1 for imp in all_improvements if imp > 0) / len(all_improvements) * 100
        }
        
        # Save evaluation results
        eval_file = self.results_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"\n📊 OVERALL EVALUATION SUMMARY:")
        print(f"   Total tests: {evaluation_results['summary']['total_tests']}")
        print(f"   Mean improvement: {evaluation_results['summary']['overall_mean_improvement']:.2f}% ± {evaluation_results['summary']['overall_std_improvement']:.2f}%")
        print(f"   Success rate: {evaluation_results['summary']['success_rate']:.1f}%")
        print(f"   Results saved to: {eval_file}")
        
        return evaluation_results
    
    def _backup_aco_config(self):
        """Backup current ACO configuration."""
        return {
            'GRID_SIZE': aco.GRID_SIZE,
            'N_VEHICLES': aco.N_VEHICLES,
            'SIMULATION_TIME': aco.SIMULATION_TIME,
            'N_ANTS': aco.N_ANTS,
            'N_ITERATIONS': aco.N_ITERATIONS,
            'EVAPORATION': aco.EVAPORATION,
            'ALPHA': aco.ALPHA,
            'USE_DEFAULT_STARTING_POINT': aco.USE_DEFAULT_STARTING_POINT,
            'USE_COORDINATION': aco.USE_COORDINATION,
            'COORDINATION_FACTOR': aco.COORDINATION_FACTOR,
            'USE_TRAFFIC_ENGINEERING_RULES': aco.USE_TRAFFIC_ENGINEERING_RULES,
            'SHOW_PROGRESS': aco.SHOW_PROGRESS,
            'SHOW_PLOTS': aco.SHOW_PLOTS,
            'LAUNCH_SUMO_GUI': aco.LAUNCH_SUMO_GUI
        }
    
    def _apply_aco_config(self, config):
        """Apply configuration to ACO module."""
        aco.GRID_SIZE = config["grid_size"]
        aco.N_VEHICLES = config["n_vehicles"]
        aco.SIMULATION_TIME = config["simulation_time"]
        aco.N_ANTS = config["n_ants"]
        aco.N_ITERATIONS = config["n_iterations"]
        aco.EVAPORATION = config["evaporation_rate"]
        aco.ALPHA = config["alpha"]
        aco.USE_DEFAULT_STARTING_POINT = config["use_default_start"]
        aco.USE_COORDINATION = config["use_coordination"]
        aco.COORDINATION_FACTOR = config["coordination_factor"]
        aco.USE_TRAFFIC_ENGINEERING_RULES = config["use_traffic_engineering"]
        aco.SHOW_PROGRESS = config["verbose"]
        aco.SHOW_PLOTS = config["verbose"]
        aco.LAUNCH_SUMO_GUI = False  # Never launch GUI during training
    
    def _restore_aco_config(self, original_values):
        """Restore original ACO configuration."""
        for key, value in original_values.items():
            setattr(aco, key, value)
    
    def _evaluate_settings(self, settings, config, tag):
        """Evaluate specific traffic light settings."""
        paths = aco.get_file_paths()
        
        # Ensure scenario files exist
        if not aco.setup_scenario_files(paths):
            raise RuntimeError("Failed to setup scenario files")
        
        # Evaluate settings
        total_time, max_stop, vehicle_count = aco.evaluate_tls_settings(
            paths['net_file'], paths['sumocfg_file'], settings, paths, tag
        )
        
        return {
            "total_time": total_time,
            "max_stop": max_stop,
            "vehicle_count": vehicle_count,
            "cost": total_time + config.alpha * max_stop
        }
    
    def _save_training_results(self, scenario_name, training_config, training_time, aco_results):
        """Save training results to file."""
        results = {
            "scenario_name": scenario_name,
            "training_timestamp": datetime.now().isoformat(),
            "training_config": training_config,
            "training_time_seconds": training_time,
            "optimized_settings": aco_results['optimized_settings'],
            "training_performance": {
                "best_cost": aco_results['best_cost'],
                "best_time": aco_results['best_time'],
                "best_max_stop": aco_results['best_max_stop'],
                "improvement_percent": aco_results['improvement']
            },
            "default_performance": {
                "default_time": aco_results['default_time'],
                "default_max_stop": aco_results['default_max_stop']
            },
            "results_directory": str(self.results_dir)
        }
        
        # Save to file
        results_file = self.results_dir / f"training_{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Training results saved to: {results_file}")
        return results


def main():
    """Main function for train/evaluate interface."""
    
    print("🚦 Traffic Light Optimization: Train & Evaluate System")
    print("=" * 70)
    print("Choose an option:")
    print("1. Train: Find optimal settings for a scenario")
    print("2. Evaluate: Test trained settings on different scenarios/seeds")
    print("3. Quick demo: Train on commuter pattern, evaluate on multiple seeds")
    print("4. Show configuration")
    print("0. Exit")
    
    optimizer = TrafficOptimizer()
    
    while True:
        try:
            choice = input("\nEnter your choice (0-4): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
                
            elif choice == "1":
                # Training mode
                scenario = input("Enter scenario name (default='commuter_test'): ").strip() or "commuter_test"
                seed_input = input("Enter random seed (default=42): ").strip()
                seed = int(seed_input) if seed_input else 42
                
                print(f"\n🎓 Starting training for scenario '{scenario}' with seed {seed}")
                results = optimizer.train(scenario_name=scenario, custom_seed=seed)
                
                if results:
                    print("\n✅ Training completed successfully!")
                    print("   Use option 2 to evaluate these results.")
                    
            elif choice == "2":
                # Evaluation mode
                results_files = list(optimizer.results_dir.glob("training_*.json"))
                if not results_files:
                    print("❌ No training results found. Run training first (option 1).")
                    continue
                
                print("\nAvailable training results:")
                for i, file in enumerate(results_files, 1):
                    print(f"   {i}. {file.name}")
                
                file_choice = input(f"Select file (1-{len(results_files)}): ").strip()
                try:
                    file_idx = int(file_choice) - 1
                    training_file = results_files[file_idx]
                    
                    scenarios = input("Enter scenarios to test (comma-separated, default='commuter,realistic'): ").strip()
                    eval_scenarios = [s.strip() for s in scenarios.split(',')] if scenarios else ["commuter", "realistic"]
                    
                    seeds_input = input("Enter seeds to test (comma-separated, default='1,2,3,4,5'): ").strip()
                    eval_seeds = [int(s.strip()) for s in seeds_input.split(',')] if seeds_input else [1, 2, 3, 4, 5]
                    
                    print(f"\n🔬 Starting evaluation...")
                    results = optimizer.evaluate(training_file, eval_scenarios, eval_seeds)
                    
                except (ValueError, IndexError):
                    print("❌ Invalid selection")
                    
            elif choice == "3":
                # Quick demo
                print("\n🎮 Quick Demo: Train on commuter pattern, evaluate on multiple seeds")
                print("   This will train on seed 42 and evaluate on seeds 1-5")
                
                # Train
                results = optimizer.train(scenario_name="demo_commuter", custom_seed=42)
                
                if results:
                    # Find the training file we just created
                    training_files = list(optimizer.results_dir.glob("training_demo_commuter_*.json"))
                    if training_files:
                        latest_file = max(training_files, key=os.path.getctime)
                        
                        # Evaluate
                        optimizer.evaluate(latest_file, ["commuter"], [1, 2, 3, 4, 5])
                        
            elif choice == "4":
                # Show configuration
                print("\n📋 Current Configuration:")
                for key, value in optimizer.config.items():
                    print(f"   {key}: {value}")
                    
            else:
                print("❌ Invalid choice. Please enter 0-4.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
