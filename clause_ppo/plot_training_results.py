#!/usr/bin/env python3
"""
Plot PPO training results: rolling accuracy and rewards vs episodes
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def load_training_data(log_path):
    """Load and parse PPO training log data."""
    data = []
    
    with open(log_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    # Skip config and completion entries
                    if 'episode' in entry:
                        data.append(entry)
                except json.JSONDecodeError:
                    continue
    
    return data

def calculate_rolling_stats(data, window=10):
    """Calculate rolling statistics for plotting."""
    df = pd.DataFrame(data)
    
    # Convert success_rate from rolling window to binary success for each episode
    df['success'] = (df['terminal'] > 0).astype(int)
    
    # Calculate rolling averages
    df['rolling_success_rate'] = df['success'].rolling(window=window, min_periods=1).mean()
    df['rolling_reward'] = df['reward'].rolling(window=window, min_periods=1).mean()
    
    return df

def plot_training_results(log_path, output_path=None, window=10):
    """Create plots of PPO training progression."""
    print(f"Loading training data from {log_path}")
    
    # Load data
    data = load_training_data(log_path)
    if not data:
        print("No valid training data found!")
        return
    
    print(f"Found {len(data)} training episodes")
    
    # Calculate rolling stats
    df = calculate_rolling_stats(data, window=window)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PPO Training Results: Clause-Level SQL Repair', fontsize=16, fontweight='bold')
    
    # Plot 1: Rolling Success Rate
    ax1.plot(df['episode'], df['rolling_success_rate'], 'b-', linewidth=2, alpha=0.8, label=f'Rolling Success Rate (window={window})')
    ax1.scatter(df['episode'], df['success'], alpha=0.3, s=10, c=['red' if x == 0 else 'green' for x in df['success']], label='Individual Episodes')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Training Success Rate Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(-0.05, 1.05)
    
    # Add trend line
    z = np.polyfit(df['episode'], df['rolling_success_rate'], 1)
    trend_line = np.poly1d(z)
    ax1.plot(df['episode'], trend_line(df['episode']), "r--", alpha=0.8, 
             label=f'Trend (slope={z[0]:.4f})')
    ax1.legend()
    
    # Plot 2: Rolling Reward
    ax2.plot(df['episode'], df['rolling_reward'], 'g-', linewidth=2, alpha=0.8, label=f'Rolling Reward (window={window})')
    ax2.scatter(df['episode'], df['reward'], alpha=0.3, s=10, c=['red' if x < 0 else 'green' for x in df['reward']], label='Individual Episodes')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.set_title('Training Reward Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Plot 3: Success Rate Distribution by Phase
    phase_size = len(df) // 4
    phases = []
    phase_labels = ['Early (1-25%)', 'Mid-Early (25-50%)', 'Mid-Late (50-75%)', 'Late (75-100%)']
    
    for i in range(4):
        start_idx = i * phase_size
        end_idx = (i + 1) * phase_size if i < 3 else len(df)
        phase_data = df.iloc[start_idx:end_idx]
        phases.append(phase_data['success'].mean())
    
    bars = ax3.bar(phase_labels, phases, color=['lightgreen', 'green', 'orange', 'red'], alpha=0.7)
    ax3.set_ylabel('Average Success Rate')
    ax3.set_title('Success Rate by Training Phase')
    ax3.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, phases):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Episode Duration (if available)
    if 'elapsed_time' in df.columns:
        # Calculate time per episode
        df['time_diff'] = df['elapsed_time'].diff().fillna(df['elapsed_time'].iloc[0])
        ax4.plot(df['episode'], df['time_diff'], 'purple', alpha=0.7, label='Time per Episode')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_title('Episode Duration Over Time')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Highlight generation speed collapse
        if df['time_diff'].max() > df['time_diff'].median() * 3:
            ax4.axhline(y=df['time_diff'].median() * 3, color='red', linestyle='--', alpha=0.7, 
                       label='3x Median Threshold')
            ax4.legend()
    else:
        # Plot reward vs success rate correlation
        ax4.scatter(df['reward'], df['success'], alpha=0.6, c=df['episode'], cmap='viridis')
        ax4.set_xlabel('Reward')
        ax4.set_ylabel('Success (0/1)')
        ax4.set_title('Reward vs Success Correlation')
        ax4.grid(True, alpha=0.3)
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('Episode')
    
    plt.tight_layout()
    
    # Print summary statistics
    print(f"\n📊 TRAINING SUMMARY:")
    print(f"   Episodes: {len(df)}")
    print(f"   Overall Success Rate: {df['success'].mean():.1%}")
    print(f"   Early Phase (first 25%): {phases[0]:.1%}")
    print(f"   Late Phase (last 25%): {phases[-1]:.1%}")
    print(f"   Performance Change: {phases[-1] - phases[0]:+.1%}")
    print(f"   Average Reward: {df['reward'].mean():.2f}")
    print(f"   Reward Std: {df['reward'].std():.2f}")
    
    if 'time_diff' in df.columns:
        print(f"   Avg Episode Time: {df['time_diff'].mean():.1f}s")
        print(f"   Max Episode Time: {df['time_diff'].max():.1f}s")
        if df['time_diff'].max() > df['time_diff'].median() * 5:
            print(f"   ⚠️  Generation speed collapse detected!")
    
    # Save plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {output_path}")
    else:
        plt.savefig('ppo_training_results.png', dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to ppo_training_results.png")
    
    plt.show()
    
    return df

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', default='results/ppo_minimal_training_log.jsonl')
    parser.add_argument('--output', default='ppo_training_analysis.png')
    parser.add_argument('--window', type=int, default=10)
    args = parser.parse_args()
    
    # Check if log file exists
    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        
        # Try alternative paths
        alternative_paths = [
            'results/ppo_training_log.jsonl',
            '../results/ppo_minimal_training_log.jsonl',
            'clause_ppo/results/ppo_training_log.jsonl'
        ]
        
        for alt_path in alternative_paths:
            if Path(alt_path).exists():
                log_path = Path(alt_path)
                print(f"Found alternative log: {log_path}")
                break
        else:
            print("No training log found!")
            return
    
    # Create plot
    df = plot_training_results(str(log_path), args.output, args.window)
    
    # Save summary stats
    summary = {
        'total_episodes': len(df),
        'overall_success_rate': float(df['success'].mean()),
        'final_rolling_rate': float(df['rolling_success_rate'].iloc[-1]),
        'average_reward': float(df['reward'].mean()),
        'reward_variance': float(df['reward'].var()),
        'trend_slope': float(np.polyfit(df['episode'], df['rolling_success_rate'], 1)[0]),
    }
    
    with open('ppo_training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📁 Summary saved to ppo_training_summary.json")

if __name__ == '__main__':
    main()