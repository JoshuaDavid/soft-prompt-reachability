"""
Additional analysis plots for deeper understanding of results.
"""

import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

RESULTS_DIR = Path("results")

def load_json(path):
    with open(path) as f:
        return json.load(f)

# Load all metrics
full = load_json(RESULTS_DIR / "experiment1" / "metrics.json")
noffn = load_json(RESULTS_DIR / "experiment4" / "metrics_no_ffn.json")
noattn = load_json(RESULTS_DIR / "experiment4" / "metrics_no_attn.json")
interp = load_json(RESULTS_DIR / "experiment3" / "metrics.json")
emb = load_json(RESULTS_DIR / "experiment1b" / "metrics.json")
exp2a = load_json(RESULTS_DIR / "experiment2a" / "metrics.json")
exp2b = load_json(RESULTS_DIR / "experiment2b" / "metrics.json")
corr = load_json(RESULTS_DIR / "experiment5" / "metrics.json")

# ── Plot 1: Cross-ablation correlation ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

full_cos = np.array(full['final_cosines'])
noffn_cos = np.array(noffn['final_cosines'])
noattn_cos = np.array(noattn['final_cosines'])

for ax, (x, y, xl, yl, title) in zip(axes, [
    (full_cos, noffn_cos, 'Full Model', 'No FFN', 'Full vs No-FFN'),
    (full_cos, noattn_cos, 'Full Model', 'No Attn', 'Full vs No-Attn'),
    (noffn_cos, noattn_cos, 'No FFN', 'No Attn', 'No-FFN vs No-Attn'),
]):
    ax.scatter(x, y, alpha=0.6, s=30)
    r, p = pearsonr(x, y)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(f'{title}\nr={r:.3f}, p={p:.2e}')
    # Fit line
    z = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, np.polyval(z, xline), 'r--', alpha=0.5)
    ax.grid(True, alpha=0.2)

fig.suptitle('Target Difficulty Correlations Across Model Variants', fontsize=14)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "difficulty_correlations.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved difficulty_correlations.png")

# ── Plot 2: Grand comparison - all experiments on one axis ──
fig, ax = plt.subplots(figsize=(16, 8))

# Collect all data points
categories = []
values = []
colors = []
positions = []

def add_category(name, data, color, pos):
    categories.append(name)
    values.append(data)
    colors.append(color)
    positions.append(pos)

add_category('Exp 1:\nReal Targets', full['final_cosines'], '#2196F3', 0)
add_category('Exp 4a:\nNo FFN', noffn['final_cosines'], '#F44336', 1)
add_category('Exp 4b:\nNo Attn', noattn['final_cosines'], '#FF5722', 2)
add_category('Exp 2a:\nDist-Matched', exp2a['final_cosines'], '#4CAF50', 3.5)
add_category('Exp 2b:\nRaw Random', exp2b['final_cosines'], '#FF9800', 4.5)

# Interpolation alphas
for alpha in ["-0.5", "0.5", "2.0"]:
    if alpha in interp['all_cosines_by_alpha']:
        add_category(f'Exp 3:\nα={alpha}', interp['all_cosines_by_alpha'][alpha], '#9C27B0',
                     6 + ["-0.5", "0.5", "2.0"].index(alpha))

# Corruption fractions
for frac in ["0.0", "0.1", "0.3", "0.7", "1.0"]:
    if frac in corr['all_cosines_by_fraction']:
        add_category(f'Exp 5:\nf={frac}', corr['all_cosines_by_fraction'][frac], '#607D8B',
                     9.5 + ["0.0", "0.1", "0.3", "0.7", "1.0"].index(frac))

bp = ax.boxplot(values, positions=positions, widths=0.7, patch_artist=True,
                showfliers=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_xticks(positions)
ax.set_xticklabels(categories, fontsize=8, rotation=45, ha='right')
ax.set_ylabel('Final Cosine Similarity', fontsize=12)
ax.set_title('Complete Comparison: All Experiments and Conditions', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

# Add group separators
for x in [3, 5.5, 9]:
    ax.axvline(x, color='gray', linestyle=':', alpha=0.3)

fig.tight_layout()
fig.savefig(RESULTS_DIR / "complete_comparison.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved complete_comparison.png")

# ── Plot 3: Convergence comparison across experiments ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# We need to reconstruct convergence curves from different experiments
# Use the median cosine history from the first experiment's results

# Exp 1 convergence
cos_histories = [r['best']['cosine_history'] for r in [
    {"best": {"cosine_history": full['all_restart_cosines'][i]}}
    for i in range(len(full['all_restart_cosines']))
] if 'cosine_history' in r.get('best', {})]

# Since we don't have full convergence curves in the JSON,
# let's do a different analysis: final MSE vs final cosine
ax = axes[0, 0]
for name, data, color in [
    ("Real Targets", full, '#2196F3'),
    ("Dist-Matched", exp2a, '#4CAF50'),
    ("Raw Random", exp2b, '#FF9800'),
]:
    cos = np.array(data['final_cosines'])
    mse = np.array(data['final_mses'])
    ax.scatter(cos, mse, alpha=0.5, s=20, color=color, label=name)
ax.set_xlabel('Final Cosine Similarity')
ax.set_ylabel('Final MSE')
ax.set_title('Cosine vs MSE Tradeoff')
ax.legend()
ax.set_yscale('log')
ax.grid(True, alpha=0.2)

# Restart variance distribution
ax = axes[0, 1]
for name, data, color in [
    ("Real", full, '#2196F3'),
    ("No-FFN", noffn, '#F44336'),
    ("No-Attn", noattn, '#FF5722'),
]:
    vars = np.array(data['restart_variances'])
    ax.hist(vars, bins=20, alpha=0.5, color=color, label=f'{name} (max={vars.max():.2e})')
ax.set_xlabel('Restart Variance')
ax.set_ylabel('Count')
ax.set_title('Optimization Landscape: Restart Variance')
ax.legend(fontsize=8)
ax.set_xscale('log')

# Interpolation: MSE by alpha
ax = axes[1, 0]
alphas = sorted(interp['median_cosine_by_alpha'].keys(), key=float)
mse_by_alpha = {}
# Use MSE from metrics if available, otherwise use cosine data
cos_vals = [interp['median_cosine_by_alpha'][a] for a in alphas]
ax.plot([float(a) for a in alphas], cos_vals, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Alpha')
ax.set_ylabel('Median Cosine Similarity')
ax.set_title('Exp 3: Interpolation/Extrapolation')
ax.axvspan(0, 1, alpha=0.08, color='green')
ax.grid(True, alpha=0.3)

# Corruption: all individual points
ax = axes[1, 1]
fracs = sorted(corr['all_cosines_by_fraction'].keys(), key=float)
for fi, frac in enumerate(fracs):
    vals = corr['all_cosines_by_fraction'][frac]
    x = [float(frac)] * len(vals)
    ax.scatter(x, vals, alpha=0.3, s=10, color='blue')
medians = [np.median(corr['all_cosines_by_fraction'][f]) for f in fracs]
ax.plot([float(f) for f in fracs], medians, 'ro-', linewidth=2, markersize=8, zorder=5, label='Median')
ax.set_xlabel('Fraction PCA Dirs Replaced')
ax.set_ylabel('Final Cosine Similarity')
ax.set_title('Exp 5: Corruption Sweep (all points)')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle('Additional Diagnostic Plots', fontsize=14)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "additional_diagnostics.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved additional_diagnostics.png")

print("\nDone!")
