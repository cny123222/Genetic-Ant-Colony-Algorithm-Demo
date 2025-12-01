#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize BipedalWalker Training Progress
Extracts training data from log file and creates a fitness curve plot
"""

import re
import matplotlib.pyplot as plt
import numpy as np

# Set up better fonts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

# Chinese font for labels
from matplotlib.font_manager import FontProperties
chinese_font = None
for font_name in ['Arial Unicode MS', 'SimHei', 'Songti SC', 'STSong']:
    try:
        chinese_font = FontProperties(fname=None, family=font_name)
        break
    except:
        continue

if chinese_font is None:
    chinese_font = FontProperties()

def parse_log_file(log_file_path, max_gen=600):
    """Parse training log file and extract best fitness, mean fitness, and std per generation"""
    generations = []
    best_fitness = []
    mean_fitness = []
    std_fitness = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match generation lines with Best, Mean, and Std
    # Example: [Gen 1/600] ... Best: -91.19, Mean: -92.35, Std: 1.79 ...
    pattern = r'\[Gen (\d+)/\d+\].*?Best: ([-\d.]+), Mean: ([-\d.]+), Std: ([-\d.]+)'
    
    matches = re.findall(pattern, content)
    
    for gen_str, best_str, mean_str, std_str in matches:
        gen = int(gen_str)
        if gen <= max_gen:
            generations.append(gen)
            best_fitness.append(float(best_str))
            mean_fitness.append(float(mean_str))
            std_fitness.append(float(std_str))
    
    return np.array(generations), np.array(best_fitness), np.array(mean_fitness), np.array(std_fitness)

def visualize_training(log_file_path, output_path='walker_training_progress.png', max_gen=600):
    """Create training progress visualization (matching CartPole style)"""
    
    # Parse log file
    generations, best_fitness, mean_fitness, std_fitness = parse_log_file(log_file_path, max_gen)
    
    if len(generations) == 0:
        print("No training data found in log file!")
        return
    
    print(f"Loaded {len(generations)} generations of data")
    print(f"Best fitness range: {best_fitness.min():.2f} to {best_fitness.max():.2f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot best fitness (blue line)
    ax.plot(generations, best_fitness, 'b-', label='最佳适应度', linewidth=2)
    
    # Plot mean fitness (green dashed line)
    ax.plot(generations, mean_fitness, 'g--', label='平均适应度', linewidth=1.5)
    
    # Plot std range (green shaded area)
    ax.fill_between(
        generations,
        mean_fitness - std_fitness,
        mean_fitness + std_fitness,
        alpha=0.3,
        color='green',
        label='标准差范围'
    )
    
    # Set labels with larger font
    ax.set_xlabel('代数', fontproperties=chinese_font, fontsize=18)
    ax.set_ylabel('适应度', fontproperties=chinese_font, fontsize=18)
    
    # Set axis limits
    ax.set_xlim(0, max_gen)
    
    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Add legend in lower right
    ax.legend(loc='lower right', fontsize=14, prop=chinese_font)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training progress plot saved to: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    log_file = "training_round7_r4params_600gen_20251127_145807.txt"
    output_file = "walker_training_progress.png"
    
    print("=" * 60)
    print("BipedalWalker Training Progress Visualization")
    print("=" * 60)
    print("Using Round 7 data (Best fitness: 173.26)")
    print()
    
    visualize_training(log_file, output_file, max_gen=600)
    
    print("\nDone!")

