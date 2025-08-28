#!/usr/bin/env python3
"""
User-Friendly Traffic Optimization Menu

This example provides a guided menu system for new users to easily
explore traffic light optimization without needing to know all the
technical details upfront.

Usage:
    python examples/menu_example.py

Author: Traffic Optimization System
Date: August 2025
"""

import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def show_welcome():
    """Show welcome message and overview."""
    print("🚦 TRAFFIC LIGHT OPTIMIZATION SYSTEM")
    print("=" * 60)
    print("Welcome! This system uses Ant Colony Optimization (ACO) to find")
    print("better traffic light timings that reduce vehicle waiting times.")
    print()
    print("🎯 What the system does:")
    print("   • Creates virtual traffic scenarios")
    print("   • Tests different signal timing combinations")
    print("   • Finds settings that minimize traffic delays")
    print("   • Shows you the improvement compared to defaults")
    print()

def show_main_menu():
    """Show main menu options."""
    print("📋 CHOOSE YOUR EXPERIENCE LEVEL")
    print("=" * 40)
    print()
    print("1. 🚀 Quick Demo (Beginner)")
    print("   → See the system in action with preset values")
    print("   → Takes ~2-3 minutes")
    print("   → Great for first-time users")
    print()
    print("2. ⚙️  Custom Configuration (Intermediate)")
    print("   → Choose your own settings with guidance")
    print("   → Flexible optimization parameters")
    print("   → Learn about different options")
    print()
    print("3. 🔬 Sensitivity Analysis (Advanced)")
    print("   → Find optimal parameter combinations")
    print("   → Compare multiple configurations")
    print("   → Statistical analysis and plots")
    print()
    print("4. 📚 Learn More")
    print("   → Understand the technology")
    print("   → See all available options")
    print("   → Get optimization tips")
    print()
    print("5. 👋 Exit")
    print()

def run_quick_demo():
    """Run a quick demonstration with preset values."""
    print("🚀 QUICK DEMO MODE")
    print("=" * 50)
    print("Perfect for first-time users! Using proven settings.")
    print()
    
    print("⚙️  Demo Settings:")
    print("   🏗️  Network: 3x3 grid (9 intersections)")
    print("   🚗 Traffic: 30 vehicles, commuter pattern")
    print("   ⏱️  Time: 600 seconds simulation")
    print("   🐜 Algorithm: 20 ants, 10 iterations")
    print("   📊 Output: Progress plots enabled")
    print()
    
    confirm = input("Start demo? (y/n) [default: y]: ").strip().lower()
    if confirm == 'n':
        return False
    
    try:
        print("\n🔄 Running optimization...")
        
        # Import and run with demo config
        from examples.quick_start import main as quick_start_main
        success = quick_start_main()
        
        if success:
            print("\n🎉 Demo completed successfully!")
            print("\n💡 Next Steps:")
            print("   • Try option 2 to customize your own settings")
            print("   • Experiment with different traffic patterns")
            print("   • Use option 3 for parameter optimization")
        
        return success
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def run_custom_configuration():
    """Run optimization with custom user configuration."""
    print("⚙️  CUSTOM CONFIGURATION MODE")
    print("=" * 50)
    print("Configure your optimization with helpful guidance!")
    print()
    
    try:
        # Import and run interactive configuration
        from examples.simple_aco_optimization import get_user_configuration, run_optimization_with_config
        
        print("Let's configure your optimization step by step...")
        print()
        
        config = get_user_configuration()
        
        print(f"\n🚀 Starting Your Custom Optimization")
        print("-" * 50)
        
        success = run_optimization_with_config(config)
        
        if success:
            print("\n🎉 Custom optimization completed!")
            print("\n💡 What to try next:")
            print("   • Run again with different settings")
            print("   • Try sensitivity analysis (option 3)")
            print("   • Experiment with other traffic patterns")
        
        return success
        
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False

def run_sensitivity_analysis():
    """Run sensitivity analysis to find optimal parameters."""
    print("🔬 SENSITIVITY ANALYSIS MODE")
    print("=" * 50)
    print("Find the best parameter combinations systematically!")
    print()
    
    try:
        from examples.sensitivity_example import run_quick_sensitivity_demo
        
        print("This will help you discover optimal settings...")
        print()
        
        success = run_quick_sensitivity_demo()
        
        if success:
            print("\n🎉 Sensitivity analysis completed!")
            print("\n💡 Advanced Options:")
            print("   • Modify src/sensitivity_analysis.py for custom tests")
            print("   • Test multiple parameters simultaneously")
            print("   • Run longer analyses with more replications")
        
        return success
        
    except Exception as e:
        print(f"❌ Sensitivity analysis failed: {e}")
        return False

def show_learning_resources():
    """Show learning resources and tips."""
    print("📚 LEARNING RESOURCES")
    print("=" * 50)
    
    print("\n🎓 Understanding the System:")
    print("   • ACO (Ant Colony Optimization): Bio-inspired algorithm")
    print("   • Ants explore different signal timing combinations")
    print("   • Better solutions are reinforced over iterations")
    print("   • Gradual convergence to optimal or near-optimal settings")
    
    print("\n⚙️  Key Parameters Explained:")
    print("   • Grid Size: Larger = more complex, takes longer")
    print("   • Vehicles: More vehicles = more realistic, slower simulation") 
    print("   • Simulation Time: Longer = more accurate results")
    print("   • Ants: More ants = better exploration, longer runtime")
    print("   • Iterations: More iterations = better convergence")
    
    print("\n🚦 Traffic Patterns:")
    print("   • Commuter: Suburbs to downtown (realistic rush hour)")
    print("   • Industrial: Horizontal corridor flow")
    print("   • Random: Unpredictable origins and destinations")
    
    print("\n💡 Optimization Tips:")
    print("   • Start with smaller scenarios (2x2 or 3x3 grids)")
    print("   • Use 20-30 ants and 10-15 iterations for good results")
    print("   • Try different traffic patterns to test robustness")
    print("   • Use sensitivity analysis to find optimal settings")
    print("   • Longer simulations give more stable results")
    
    print("\n📁 File Locations:")
    print("   • Results: saved to 'results/' directory")
    print("   • Plots: optimization progress and comparisons")
    print("   • Configurations: JSON files with settings")
    print("   • SUMO files: network and traffic scenario data")
    
    print("\n🔧 Advanced Usage:")
    print("   • main.py: Command-line interface with options")
    print("   • src/: Core modules for custom development")
    print("   • Import modules to build custom optimization tools")
    
    input("\nPress Enter to return to main menu...")

def main():
    """Main menu loop."""
    
    while True:
        try:
            show_welcome()
            show_main_menu()
            
            choice = input("Choose an option (1-5) [default: 1]: ").strip()
            if not choice:
                choice = '1'
            
            print("\n" + "=" * 60)
            
            if choice == '1':
                run_quick_demo()
                
            elif choice == '2':
                run_custom_configuration()
                
            elif choice == '3':
                run_sensitivity_analysis()
                
            elif choice == '4':
                show_learning_resources()
                
            elif choice == '5':
                print("👋 Thanks for using the Traffic Optimization System!")
                print("Feel free to run again anytime to explore more options.")
                break
                
            else:
                print("❌ Please choose option 1, 2, 3, 4, or 5")
                input("\nPress Enter to continue...")
                continue
            
            # Ask if user wants to continue
            print("\n" + "=" * 60)
            continue_choice = input("Return to main menu? (y/n) [default: y]: ").strip().lower()
            if continue_choice == 'n':
                print("👋 Thanks for using the system!")
                break
            
            print("\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for trying the system.")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Let's return to the main menu...")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()
