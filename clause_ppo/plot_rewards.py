#!/usr/bin/env python3
"""
Real-time PPO reward monitoring script
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
import sys

def load_training_log(log_path):
    """Load training log and extract rewards"""
    if not Path(log_path).exists():
        return [], [], [], []
    
    episodes, terminals, prm_scores, rewards = [], [], [], []
    
    with open(log_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                episodes.append(entry['episode'])
                terminals.append(entry['terminal'])
                prm_scores.append(entry['prm_score'])
                rewards.append(entry['reward'])
            except (json.JSONDecodeError, KeyError):
                continue
    
    return episodes, terminals, prm_scores, rewards

def plot_rewards(log_path, save_plot=False):
    """Plot reward progression"""
    episodes, terminals, prm_scores, rewards = load_training_log(log_path)
    
    if not episodes:
        print(f"No data found in {log_path}")
        return
    
    # Calculate running averages
    window = min(50, len(rewards) // 4) if len(rewards) > 10 else len(rewards)
    if window > 1:
        reward_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        terminal_avg = np.convolve(terminals, np.ones(window)/window, mode='valid')
        episodes_avg = episodes[window-1:]
    else:
        reward_avg = rewards
        terminal_avg = terminals
        episodes_avg = episodes
    
    # Create plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Total reward plot
    ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw Reward')
    if len(reward_avg) > 1:
        ax1.plot(episodes_avg, reward_avg, color='red', linewidth=2, label=f'Avg (window={window})')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('PPO Training Progress: Total Reward (Execution + PRM)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Execution success rate
    ax2.plot(episodes, terminals, alpha=0.3, color='green', label='Raw Execution')
    if len(terminal_avg) > 1:
        ax2.plot(episodes_avg, terminal_avg, color='orange', linewidth=2, label=f'Success Rate (window={window})')
    ax2.set_ylabel('Execution Success (-1/+1)')
    ax2.set_title('Execution Success Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # PRM scores
    ax3.plot(episodes, prm_scores, alpha=0.6, color='purple', label='PRM Score')
    ax3.set_ylabel('PRM Score')
    ax3.set_xlabel('Episode')
    ax3.set_title('Process Reward Model Scores')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('ppo_reward_progress.png', dpi=150, bbox_inches='tight')
        print("Saved plot: ppo_reward_progress.png")
    
    # Print current statistics
    if len(rewards) > 0:
        print(f"\n📊 Current Training Statistics (Episode {episodes[-1]}):")
        print(f"   Latest Reward: {rewards[-1]:+.3f}")
        print(f"   Latest Execution: {'+SUCCESS' if terminals[-1] > 0 else 'FAILED'}")
        print(f"   Latest PRM: {prm_scores[-1]:.3f}")
        if len(rewards) >= 10:
            recent_success_rate = sum(1 for t in terminals[-10:] if t > 0) / 10
            recent_avg_reward = np.mean(rewards[-10:])
            print(f"   Recent Success Rate (last 10): {recent_success_rate:.1%}")
            print(f"   Recent Avg Reward (last 10): {recent_avg_reward:+.3f}")
    
    return fig

if __name__ == "__main__":
    log_path = "results/ppo_training_log.jsonl"
    
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        # Real-time monitoring mode
        print("🎯 Real-time PPO reward monitoring (Ctrl+C to stop)")
        print(f"📁 Watching: {log_path}")
        try:
            while True:
                plt.clf()
                fig = plot_rewards(log_path, save_plot=False)
                if fig:
                    plt.pause(5)  # Update every 5 seconds
                else:
                    time.sleep(5)
        except KeyboardInterrupt:
            print("\n✅ Monitoring stopped")
    else:
        # One-time plot
        plot_rewards(log_path, save_plot=True)
        plt.show()