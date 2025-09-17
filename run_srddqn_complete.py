#!/usr/bin/env python3
"""
Example usage of the fixed SRDDQN implementation with two-phase training.
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='SRDDQN Two-Phase Training Example')
    parser.add_argument('--config', type=str, default='configs/srddqn_config.yaml', 
                        help='Configuration file path')
    parser.add_argument('--download_data', action='store_true', 
                        help='Download fresh data')
    
    args = parser.parse_args()
    
    print("SRDDQN Two-Phase Training System")
    print("=" * 50)
    
    print("\nStep 1: Data Preparation")
    if args.download_data:
        os.system(f"python main.py --download_data --config {args.config}")
    else:
        print("Using existing data (use --download_data to refresh)")
    
    print("\nStep 2: Phase 1 - Pre-training Reward Network")
    print("This trains the reward network to predict expert rewards using supervised learning...")
    os.system(f"python main.py --pretrain --config {args.config}")
    
    print("\nStep 3: Phase 2 - SRDDQN Training")
    print("This trains the DQN agent using self-rewarding mechanism (max of self and expert rewards)...")
    os.system(f"python main.py --train --config {args.config}")
    
    print("\nStep 4: Evaluation")
    print("Evaluating the trained SRDDQN agent...")
    os.system(f"python main.py --evaluate --config {args.config}")
    
    print("\n" + "=" * 50)
    print("✅ Two-Phase SRDDQN Training Completed!")
    print("\nKey Components:")
    print("  Phase 1: models/saved/pretrained_reward_network.pth")
    print("  Phase 2: models/saved/srddqn_model.pth")
    print("  Results: results/ directory")
    
    print("\nWhat's been fixed:")
    print("  ✅ Proper two-phase training architecture")
    print("  ✅ Self-rewarding mechanism: reward = max(self_reward, expert_reward)")
    print("  ✅ Pre-trained reward network using supervised learning")
    print("  ✅ Enhanced replay buffers for both DQN and reward network training")
    print("  ✅ Correct integration of expert reward dictionaries")

if __name__ == "__main__":
    main()