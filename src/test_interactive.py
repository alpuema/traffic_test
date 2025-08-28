#!/usr/bin/env python3
"""
Test the interactive ACO example with automated inputs.
This demonstrates all the features without requiring manual input.
"""

import subprocess
import sys
import os

def test_interactive_demo():
    """Test the interactive demo with simulated user inputs."""
    
    print("🧪 Testing Interactive ACO Example")
    print("=" * 50)
    
    # Test inputs for a quick interactive demo
    # Choice 1 (Interactive mode), then reasonable defaults
    test_inputs = [
        "1",    # Interactive mode
        "3",    # Grid size 3x3
        "25",   # 25 vehicles
        "500",  # 500 seconds simulation
        "1",    # Balanced traffic pattern
        "15",   # 15 ants
        "6",    # 6 iterations
        "y",    # Show plots
        "n",    # Don't launch GUI
        "y"     # Show verbose output
    ]
    
    # Prepare input string
    input_string = "\n".join(test_inputs) + "\n"
    
    try:
        # Change to project directory
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(project_dir)
        
        # Run the example with simulated inputs
        process = subprocess.Popen(
            [sys.executable, "examples/simple_aco_optimization.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input=input_string, timeout=300)
        
        print("📋 Test Results:")
        print("-" * 30)
        print(stdout)
        
        if stderr:
            print("⚠️  Stderr:")
            print(stderr)
        
        print(f"✅ Process completed with return code: {process.returncode}")
        
        return process.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_interactive_demo()
    if success:
        print("\n🎉 Interactive demo test completed successfully!")
    else:
        print("\n❌ Interactive demo test failed.")
