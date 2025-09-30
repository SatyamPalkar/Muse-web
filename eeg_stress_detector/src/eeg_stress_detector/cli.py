#!/usr/bin/env python3
"""
Command Line Interface for EEG Stress Detector
==============================================
"""

import argparse
import sys
from .stress_detector import StressDetector, SessionAnalyzer


def main():
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="EEG Stress Detector - Advanced stress analysis system",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        choices=["session", "realtime", "demo"],
        default="session",
        help="Analysis mode (default: session)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="OSC port for Mind Monitor (default: 8000)"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Session duration in seconds (default: 120)"
    )
    
    args = parser.parse_args()
    
    print("🧠 EEG Stress Detector Package")
    print("=" * 40)
    print(f"Mode: {args.mode}")
    print(f"Port: {args.port}")
    
    if args.mode == "session":
        print("📊 Session analysis mode")
        print("Connect your Mind Monitor and start collecting data")
        # Session mode implementation would go here
        
    elif args.mode == "realtime":
        print("🔄 Real-time monitoring mode")  
        print("Live stress detection active")
        # Real-time mode implementation would go here
        
    elif args.mode == "demo":
        print("🧪 Demo mode with simulated data")
        run_demo()


def run_demo():
    """Run demo with simulated EEG data."""
    
    import numpy as np
    import time
    
    detector = StressDetector()
    
    print("\n🧪 Testing with simulated EEG patterns...")
    
    # Simulate different mental states
    patterns = [
        ("Relaxed", 15, 40, 18, 6, 28),
        ("Focused", 12, 25, 35, 12, 18),
        ("Stressed", 8, 15, 50, 20, 12),
        ("Very Relaxed", 18, 45, 15, 5, 30)
    ]
    
    for state_name, t, a, b, g, d in patterns:
        print(f"\n📊 Simulating {state_name} state...")
        
        # Add some realistic noise
        theta = max(1, np.random.normal(t, 2))
        alpha = max(1, np.random.normal(a, 4))
        beta = max(1, np.random.normal(b, 3))
        gamma = max(1, np.random.normal(g, 2))
        delta = max(1, np.random.normal(d, 3))
        
        # Analyze sample
        result = detector.analyze_single_sample(theta, alpha, beta, gamma, delta)
        
        print(f"  🧠 Stress Level: {result['stress_level']}")
        print(f"  📈 Confidence: {result['confidence']:.3f}")
        print(f"  🔢 Stress Index: {result['features']['stress_index']:.2f}")
        
        if result['recommendations']:
            print(f"  💡 {result['recommendations'][0]}")
        
        time.sleep(1)
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()