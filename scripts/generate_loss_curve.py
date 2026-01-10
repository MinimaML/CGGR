import json
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_editorial_plot():
    results_path = r'c:\Users\wrc02\Desktop\CGGR\benchmark_results\benchmark_results.json'
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    # Aesthetics to match dashboard
    bg_color = '#FDFCF0'  # Off-white / Cream
    text_color = '#1A1A1A' # Deep Black
    color_std = '#D32F2F'  # Editorial Red
    color_cggr = '#2E7D32' # Editorial Green

    plt.rcParams['text.color'] = text_color
    plt.rcParams['axes.labelcolor'] = text_color
    plt.rcParams['xtick.color'] = text_color
    plt.rcParams['ytick.color'] = text_color

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    for quality in data['quality']:
        name = quality['name']
        history = np.array(quality['loss_history'])
        
        # Apply smoothing
        window = 5
        smoothed = np.convolve(history, np.ones(window)/window, mode='valid')
        x_axis = np.arange(len(smoothed))
        
        color = color_cggr if 'CGGR' in name else color_std
        
        # Plot raw data with low alpha
        ax.plot(history, color=color, alpha=0.1, linewidth=1)
        # Plot smoothed data
        ax.plot(x_axis + window//2, smoothed, label=name, color=color, linewidth=2.5)

    ax.set_title('CONVERGENCE RACE: LOSS DEPTH', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    
    ax.grid(True, linestyle='-', alpha=0.1, color=text_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    ax.legend(frameon=False, fontsize=11)

    output_path = r'c:\Users\wrc02\Desktop\CGGR\ignore\loss_curve.png'
    plt.savefig(output_path, dpi=300, facecolor=bg_color, bbox_inches='tight')
    print(f"Editorial loss curve saved to {output_path}")

if __name__ == "__main__":
    generate_editorial_plot()
