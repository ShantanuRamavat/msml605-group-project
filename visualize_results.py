import matplotlib.pyplot as plt
import numpy as np

# ── Real-time latency data ──────────────────────────────────────────
rt_labels = ['Mean', 'P50', 'P95', 'P99', 'Max']
rt_values = [15.58, 15.17, 17.16, 18.77, 39.17]

# ── Batch data ──────────────────────────────────────────────────────
batch_sizes        = [1, 10, 50, 100, 500]
avg_per_house_ms   = [0.261, 0.046, 0.013, 0.008, 0.007]
end_to_end_ms      = [43.332, 8.986, 16.407, 8.418, 15.303]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('House Price API — Performance Analysis on AWS EKS', fontsize=15, fontweight='bold')

# ── Plot 1: Real-time latency bar chart ─────────────────────────────
ax1 = axes[0, 0]
colors = ['#2196F3', '#2196F3', '#FF9800', '#F44336', '#F44336']
bars = ax1.bar(rt_labels, rt_values, color=colors, edgecolor='white', width=0.5)
ax1.set_title('Real-Time API Latency (100 requests)', fontweight='bold')
ax1.set_ylabel('Latency (ms)')
ax1.set_ylim(0, 45)
for bar, val in zip(bars, rt_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val}ms', ha='center', va='bottom', fontsize=9)
ax1.axhline(y=15.58, color='blue', linestyle='--', alpha=0.4, label='Mean')
ax1.legend()

# ── Plot 2: Batch avg per house ─────────────────────────────────────
ax2 = axes[0, 1]
ax2.plot(batch_sizes, avg_per_house_ms, marker='o', color='#4CAF50',
         linewidth=2, markersize=8, label='Avg per house')
ax2.axhline(y=15.58, color='#2196F3', linestyle='--', linewidth=1.5,
            label=f'Real-time mean (15.58ms)')
ax2.set_title('Batch — Avg Latency per House vs Batch Size', fontweight='bold')
ax2.set_xlabel('Batch Size')
ax2.set_ylabel('Avg Latency per House (ms)')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)
for x, y in zip(batch_sizes, avg_per_house_ms):
    ax2.annotate(f'{y}ms', (x, y), textcoords='offset points',
                 xytext=(5, 5), fontsize=8)

# ── Plot 3: End-to-end batch latency ────────────────────────────────
ax3 = axes[1, 0]
bars3 = ax3.bar([str(s) for s in batch_sizes], end_to_end_ms,
                color='#9C27B0', edgecolor='white', width=0.5)
ax3.set_title('Batch — End-to-End Latency per Request', fontweight='bold')
ax3.set_xlabel('Batch Size')
ax3.set_ylabel('End-to-End Latency (ms)')
for bar, val in zip(bars3, end_to_end_ms):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}ms', ha='center', va='bottom', fontsize=9)

# ── Plot 4: Throughput comparison ───────────────────────────────────
ax4 = axes[1, 1]
rt_throughput    = 1000 / 15.58          # houses per second, real-time
batch_throughput = [s / (e/1000) for s, e in zip(batch_sizes, end_to_end_ms)]
all_labels  = ['Real-Time\n(1 house)'] + [f'Batch\n({s})' for s in batch_sizes]
all_values  = [rt_throughput] + batch_throughput
bar_colors  = ['#2196F3'] + ['#4CAF50'] * len(batch_sizes)
bars4 = ax4.bar(all_labels, all_values, color=bar_colors, edgecolor='white')
ax4.set_title('Throughput — Houses Predicted per Second', fontweight='bold')
ax4.set_ylabel('Houses / second')
for bar, val in zip(bars4, all_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f'{val:.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('performance_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved as performance_results.png")