import matplotlib.pyplot as plt
import numpy as np
import os

def create_editorial_dashboard():
    # Data from README and Reports
    std_throughput = 14368 / 6 
    cggr_throughput = 58716 / 6
    throughput_gain = ((cggr_throughput / std_throughput) - 1) * 100

    std_acc = 8.0
    cggr_acc = 9.5
    acc_rel_improvement = ((cggr_acc / std_acc) - 1) * 100

    std_loss = 0.36
    cggr_loss = 0.10
    loss_reduction = ((std_loss - cggr_loss) / std_loss) * 100

    # Aesthetics
    bg_color = '#FDFCF0'  # Off-white / Cream
    text_color = '#1A1A1A' # Deep Black
    color_std = '#D32F2F'  # Editorial Red
    color_cggr = '#2E7D32' # Editorial Green

    plt.rcParams['text.color'] = text_color
    plt.rcParams['axes.labelcolor'] = text_color
    plt.rcParams['xtick.color'] = text_color
    plt.rcParams['ytick.color'] = text_color

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(bg_color)
    
    # 1. Throughput
    ax = axs[0, 0]
    ax.set_facecolor(bg_color)
    ax.bar(['Standard (BS=1)', 'CGGR (BS=4)'], [std_throughput, cggr_throughput], 
            color=[color_std, color_cggr], width=0.5, edgecolor=text_color, linewidth=1.5)
    ax.set_title('THROUGHPUT (Samples/Hour)', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Samples / Hour', fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle='-', alpha=0.1, color=text_color)
    ax.text(1, cggr_throughput + 100, f'+{throughput_gain:.0f}%', ha='center', va='bottom', color=color_cggr, fontweight='black', fontsize=20)

    # 2. Accuracy
    ax = axs[0, 1]
    ax.set_facecolor(bg_color)
    ax.bar(['Standard (BS=1)', 'CGGR (BS=4)'], [std_acc, cggr_acc], color=[color_std, color_cggr], width=0.5, edgecolor=text_color, linewidth=1.5)
    ax.set_title('SOLVING ACCURACY (AIME 2024)', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Accuracy %', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 15)
    ax.grid(axis='y', linestyle='-', alpha=0.1, color=text_color)
    ax.text(1, cggr_acc + 0.5, f'+{acc_rel_improvement:.1f}%', ha='center', va='bottom', color=color_cggr, fontweight='black', fontsize=20)

    # 3. Objective Convergence (Loss)
    ax = axs[1, 0]
    ax.set_facecolor(bg_color)
    ax.bar(['Standard', 'CGGR'], [std_loss, cggr_loss], color=[color_std, color_cggr], width=0.5, edgecolor=text_color, linewidth=1.5)
    ax.set_title('ERROR LEVEL (Final Loss)', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle='-', alpha=0.1, color=text_color)
    ax.text(1, cggr_loss + 0.02, f'-{loss_reduction:.0f}%', ha='center', va='bottom', color=color_cggr, fontweight='black', fontsize=20)

    # 4. Memory Efficiency (Same Scale: BS=16)
    ax = axs[1, 1]
    ax.set_facecolor(bg_color)
    vram_std_16 = 19.7
    vram_cggr_16 = 6.5
    
    ax.bar(['Standard (BS=16)', 'CGGR (BS=16)'], [vram_std_16, vram_cggr_16], 
            color=[color_std, color_cggr], width=0.5, edgecolor=text_color, linewidth=1.5)
    
    ax.set_title('MEMORY EFFICIENCY (VRAM @ BS=16)', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Usage (GB)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 25)
    
    # RTX 3060 Limit line
    ax.axhline(y=12, color=text_color, linestyle='--', alpha=0.6, linewidth=2)
    ax.text(0.85, 12.5, 'RTX 3060 LIMIT (12GB)', color=text_color, ha='center', fontweight='bold', fontsize=12)
    
    # Annotations
    ax.text(1, vram_cggr_16 + 1, '-66% SAVINGS', ha='center', va='bottom', color=color_cggr, fontweight='black', fontsize=16)
    ax.text(0, vram_std_16 + 1, 'OOM (FAIL)', ha='center', va='bottom', color=color_std, fontweight='bold', fontsize=14)

    # Cleanup frames
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

    # Global Title
    fig.suptitle('CGGR: CONFIDENCE-GATED GRADIENT ROUTING\nPerformance Comparison Benchmark', 
                 fontsize=28, fontweight='black', color=text_color, y=0.98)
    
    plt.figtext(0.5, 0.88, 'Standard Training vs. Active Routing Strategy on Consumer-Grade Hardware (NVIDIA RTX 3060)', 
                ha='center', fontsize=14, color=text_color, style='italic')

    plt.tight_layout(rect=[0, 0, 1, 0.86])
    
    # Save output
    output_path = r'c:\Users\wrc02\Desktop\CGGR\ignore\benchmark_dashboard.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
    print(f"Editorial dashboard saved to {output_path}")

if __name__ == "__main__":
    create_editorial_dashboard()
