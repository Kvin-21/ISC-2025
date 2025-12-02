"""Quick system check running all AI modules."""
import sys
import os

sys.path.insert(0, 'ai_modules/navigation')
sys.path.insert(0, 'ai_modules/detection')
sys.path.insert(0, 'ai_modules/manipulation')
sys.path.insert(0, 'ai_modules/predictive')
sys.path.insert(0, 'ai_modules/damage_assessment')

import nav
import detect
import manipulate
import predictive
import damage_scan


def main():
    print("=" * 70)
    print(" ISC-2025 SATELLITE AI SYSTEM")
    print("=" * 70)
    print("\nInitialising all modules...\n")
    
    # Navigation
    print("[1/5] Navigation AI")
    print("      Running collision avoidance simulation...")
    try:
        output = os.path.join(os.path.expanduser("~"), "Downloads", "nav_demo.gif")
        result = nav.run_navigation_demo(save_path=output)
        print(f"      ✓ Complete - Trajectory: {result['trajectory_length']} points")
        print(f"      ✓ Saved to: {output}")
    except Exception as err:
        print(f"      ✗ Error: {err}")
    
    # Detection
    print("\n[2/5] Debris Detection AI")
    print("      Checking for trained model...")
    try:
        model = detect.load_model()
        if model:
            print("      ✓ Model loaded successfully")
        else:
            print("      ⚠ Model not found - run detect.py to train")
    except Exception:
        print("      ⚠ Detection module requires setup")
    
    # Damage Scanning
    print("\n[3/5] Damage Scanning AI")
    print("      Scanning target debris...")
    try:
        scan_result = damage_scan.run_damage_scan_demo()
        print(f"      ✓ Found {len(scan_result['damage_report'])} damaged components")
    except Exception as err:
        print(f"      ✗ Error: {err}")
    
    # Manipulation
    print("\n[4/5] Manipulation & Resource AI")
    print("      Training decision agent (100 episodes)...")
    try:
        agent = manipulate.train_agent(episodes=100)
        print("      ✓ Trained")
        manipulate.evaluate_agent(agent, runs=3)
    except Exception as err:
        print(f"      ✗ Error: {err}")
    
    print("\n" + "=" * 70)
    print(" CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
