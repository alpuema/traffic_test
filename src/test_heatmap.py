#!/usr/bin/env python3
"""
Simple test for traffic pattern heatmap functionality.
"""

import sys
import os

def test_heatmap_basics():
    """Test basic heatmap functionality."""
    
    print("🧪 Testing heatmap basics...")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        print("✅ Matplotlib imported successfully")
        
        # Test basic plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        data = np.random.rand(3, 3) * 10
        im = ax.imshow(data, cmap='Reds')
        ax.set_title("Test Heatmap")
        plt.colorbar(im, ax=ax)
        
        # Save test plot
        test_path = os.path.join('results', 'heatmap_test.png')
        os.makedirs('results', exist_ok=True)
        plt.savefig(test_path)
        plt.close()
        
        print(f"✅ Test heatmap saved to: {test_path}")
        
        # Test traffic generation
        from simplified_traffic import generate_network_and_routes
        print("✅ Traffic generation module imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_heatmap_basics()
    if success:
        print("✅ All basic tests passed!")
    else:
        print("❌ Some tests failed.")
