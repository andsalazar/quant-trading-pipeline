"""
Double Descent Analysis
========================
Reads all experiment results and generates comprehensive analysis plots.

Produces:
  1. Model-wise double descent (sweep): val_loss & val_auc vs params
  2. Epoch-wise double descent: loss curves for 200ep and 500ep runs
  3. Summary table of all experiments
  4. Combined epoch curves coloured by model size

Run: python analyze_results.py
"""

import os
import json
import glob
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis_plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_all_results():
    """Load all config + history pairs."""
    configs = sorted(glob.glob(os.path.join(RESULTS_DIR, '*_config.json')))
    results = []
    for cfg_path in configs:
        base = cfg_path.replace('_config.json', '')
        hist_path = base + '_history.csv'
        if not os.path.exists(hist_path):
            continue
        with open(cfg_path) as f:
            config = json.load(f)
        history = pd.read_csv(hist_path)
        results.append({
            'config': config,
            'history': history,
            'label': config.get('label', ''),
            'params': config.get('total_params', 0),
            'epochs': config.get('epochs', 0),
            'train_samples': config.get('train_samples', 0),
        })
    return results


def categorize_results(results):
    """Split into sweep, 200ep, 500ep groups."""
    sweep = [r for r in results if r['label'].startswith('mlp_w') and r['label'].endswith('_d3')]
    mlp_200 = [r for r in results if r['label'].startswith('MLP_') and r['epochs'] == 200]
    mlp_500 = [r for r in results if r['label'].startswith('MLP_') and r['epochs'] == 500]
    # Sort sweep by params
    sweep.sort(key=lambda r: r['params'])
    mlp_200.sort(key=lambda r: r['params'])
    mlp_500.sort(key=lambda r: r['params'])
    return sweep, mlp_200, mlp_500


def fmt_params(x, pos):
    if x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    return f'{x:.0f}'


# ════════════════════════════════════════════════════════════════════════
# PLOT 1: MODEL-WISE DOUBLE DESCENT (SWEEP)
# ════════════════════════════════════════════════════════════════════════

def plot_model_wise_dd(sweep):
    """The key double-descent plot: val metrics vs model size."""
    params = [r['params'] for r in sweep]
    train_samples = sweep[0]['train_samples'] if sweep else 1e6

    # Extract metrics at various checkpoints
    final_val_loss = [r['history']['val_loss'].iloc[-1] for r in sweep]
    final_val_auc = [r['history']['val_auc'].iloc[-1] for r in sweep]
    best_val_loss = [r['history']['val_loss'].min() for r in sweep]
    best_val_auc = [r['history']['val_auc'].max() for r in sweep]
    final_train_loss = [r['history']['loss'].iloc[-1] for r in sweep]
    final_train_auc = [r['history']['auc'].iloc[-1] for r in sweep]
    labels = [r['label'] for r in sweep]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Model-Wise Double Descent — MLP Width Sweep (depth=3, 300 epochs)',
                 fontsize=16, fontweight='bold')

    # Val Loss vs Params
    ax = axes[0, 0]
    ax.plot(params, final_val_loss, 'r-o', label='Final val loss', markersize=8, linewidth=2)
    ax.plot(params, best_val_loss, 'b-s', label='Best val loss', markersize=7, linewidth=2)
    ax.plot(params, final_train_loss, 'g--^', label='Final train loss', markersize=6, alpha=0.7)
    ax.axvline(train_samples, color='gray', linestyle=':', alpha=0.7, label=f'N_train = {train_samples:,.0f}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Model Parameters', fontsize=12)
    ax.set_ylabel('Loss (BCE)', fontsize=12)
    ax.set_title('Loss vs Model Size')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_params))
    for i, lbl in enumerate(labels):
        w = lbl.split('_')[1]
        ax.annotate(w, (params[i], final_val_loss[i]), fontsize=7,
                    textcoords='offset points', xytext=(5, 5))

    # Val AUC vs Params
    ax = axes[0, 1]
    ax.plot(params, final_val_auc, 'r-o', label='Final val AUC', markersize=8, linewidth=2)
    ax.plot(params, best_val_auc, 'b-s', label='Best val AUC', markersize=7, linewidth=2)
    ax.plot(params, final_train_auc, 'g--^', label='Final train AUC', markersize=6, alpha=0.7)
    ax.axvline(train_samples, color='gray', linestyle=':', alpha=0.7, label=f'N_train = {train_samples:,.0f}')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.set_xscale('log')
    ax.set_xlabel('Model Parameters', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('AUC vs Model Size')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_params))

    # Overparameterisation ratio
    ax = axes[1, 0]
    ratios = [p / train_samples for p in params]
    ax.plot(ratios, final_val_loss, 'r-o', label='Final val loss', markersize=8, linewidth=2)
    ax.plot(ratios, best_val_loss, 'b-s', label='Best val loss', markersize=7, linewidth=2)
    ax.axvline(1.0, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Interpolation threshold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Params / Train Samples (overparameterisation ratio)', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Val Loss vs Overparameterisation Ratio')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Val accuracy vs params
    ax = axes[1, 1]
    final_val_acc = [r['history']['val_accuracy'].iloc[-1] for r in sweep]
    best_val_acc = [r['history']['val_accuracy'].max() for r in sweep]
    ax.plot(params, final_val_acc, 'r-o', label='Final val accuracy', markersize=8, linewidth=2)
    ax.plot(params, best_val_acc, 'b-s', label='Best val accuracy', markersize=7, linewidth=2)
    ax.axvline(train_samples, color='gray', linestyle=':', alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('Model Parameters', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Model Size')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_params))

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '01_model_wise_double_descent.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# PLOT 2: EPOCH-WISE CURVES (SWEEP)
# ════════════════════════════════════════════════════════════════════════

def plot_sweep_epoch_curves(sweep):
    """All 9 sweep models' loss/AUC over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Epoch-Wise Training Curves — Width Sweep (depth=3, 300 epochs)',
                 fontsize=14, fontweight='bold')

    cmap = plt.cm.viridis(np.linspace(0, 1, len(sweep)))

    for i, r in enumerate(sweep):
        h = r['history']
        label = r['label'].replace('mlp_', '').replace('_d3', '') + f' ({r["params"]:,}p)'
        axes[0].plot(h['epoch'], h['val_loss'], color=cmap[i], label=label, linewidth=1.5)
        axes[1].plot(h['epoch'], h['val_auc'], color=cmap[i], label=label, linewidth=1.5)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title('Val Loss over Epochs')
    axes[0].legend(fontsize=7, loc='upper left')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation AUC')
    axes[1].set_title('Val AUC over Epochs')
    axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1].legend(fontsize=7, loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '02_sweep_epoch_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# PLOT 3: 200-EPOCH vs 500-EPOCH COMPARISON
# ════════════════════════════════════════════════════════════════════════

def plot_epoch_comparison(mlp_200, mlp_500):
    """Compare 200ep and 500ep runs for the same architectures."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Epoch-Wise Double Descent — 200 vs 500 Epochs',
                 fontsize=14, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 200ep curves
    for i, r in enumerate(mlp_200):
        h = r['history']
        c = colors[i % len(colors)]
        label = f"{r['label']} ({r['params']:,}p)"
        axes[0, 0].plot(h['epoch'], h['loss'], '--', color=c, alpha=0.5)
        axes[0, 0].plot(h['epoch'], h['val_loss'], '-', color=c, label=label, linewidth=1.5)
        axes[0, 1].plot(h['epoch'], h['auc'], '--', color=c, alpha=0.5)
        axes[0, 1].plot(h['epoch'], h['val_auc'], '-', color=c, label=label, linewidth=1.5)

    axes[0, 0].set_title('200 Epochs — Loss (dashed=train, solid=val)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('200 Epochs — AUC')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)

    # 500ep curves
    for i, r in enumerate(mlp_500):
        h = r['history']
        c = colors[i % len(colors)]
        label = f"{r['label']} ({r['params']:,}p)"
        axes[1, 0].plot(h['epoch'], h['loss'], '--', color=c, alpha=0.5)
        axes[1, 0].plot(h['epoch'], h['val_loss'], '-', color=c, label=label, linewidth=1.5)
        axes[1, 1].plot(h['epoch'], h['auc'], '--', color=c, alpha=0.5)
        axes[1, 1].plot(h['epoch'], h['val_auc'], '-', color=c, label=label, linewidth=1.5)

    axes[1, 0].set_title('500 Epochs — Loss (dashed=train, solid=val)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('500 Epochs — AUC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '03_epoch_comparison_200v500.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# PLOT 4: TRAIN vs VAL DIVERGENCE (MEMORISATION MAP)
# ════════════════════════════════════════════════════════════════════════

def plot_memorisation_map(sweep):
    """Heatmap-style: train loss approaching 0, val loss exploding."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Memorisation Dynamics — Width Sweep',
                 fontsize=14, fontweight='bold')

    cmap = plt.cm.plasma(np.linspace(0, 1, len(sweep)))

    for i, r in enumerate(sweep):
        h = r['history']
        label = r['label'].replace('mlp_', '').replace('_d3', '') + f' ({r["params"]:,}p)'
        axes[0].plot(h['epoch'], h['loss'], color=cmap[i], label=label, linewidth=1.5)
        axes[1].plot(h['loss'], h['val_loss'], color=cmap[i], label=label, linewidth=1.5, alpha=0.8)
        # Mark start and end
        axes[1].plot(h['loss'].iloc[0], h['val_loss'].iloc[0], 'o', color=cmap[i], markersize=6)
        axes[1].plot(h['loss'].iloc[-1], h['val_loss'].iloc[-1], 's', color=cmap[i], markersize=8)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_title('Train Loss Convergence')
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=6)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Train Loss')
    axes[1].set_ylabel('Val Loss')
    axes[1].set_title('Train vs Val Loss Trajectory (o=start, ■=end)')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=6)
    axes[1].grid(True, alpha=0.3)
    # Perfect generalisation line
    lims = [1e-3, 5]
    axes[1].plot(lims, lims, 'k--', alpha=0.3, label='train=val')

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '04_memorisation_map.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════

def print_summary(sweep, mlp_200, mlp_500):
    """Print comprehensive summary table."""
    print("\n" + "=" * 100)
    print("DOUBLE DESCENT EXPERIMENT — COMPLETE RESULTS SUMMARY")
    print("=" * 100)

    def print_group(title, group):
        print(f"\n{'─' * 100}")
        print(f"  {title}")
        print(f"{'─' * 100}")
        print(f"  {'Label':<20} {'Params':>10} {'Ratio':>8} {'Epochs':>7}  "
              f"{'Train Loss':>11} {'Val Loss':>10} {'Train AUC':>10} {'Val AUC':>10} "
              f"{'Best ValAUC':>11} {'BestEp':>7}")
        print(f"  {'-'*18:<20} {'-'*10:>10} {'-'*8:>8} {'-'*7:>7}  "
              f"{'-'*11:>11} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10} "
              f"{'-'*11:>11} {'-'*7:>7}")
        for r in group:
            h = r['history']
            ratio = r['params'] / r['train_samples'] if r['train_samples'] > 0 else 0
            best_val_auc = h['val_auc'].max()
            best_ep = h.loc[h['val_auc'].idxmax(), 'epoch']
            print(f"  {r['label']:<20} {r['params']:>10,} {ratio:>8.3f} {r['epochs']:>7}  "
                  f"{h['loss'].iloc[-1]:>11.4f} {h['val_loss'].iloc[-1]:>10.4f} "
                  f"{h['auc'].iloc[-1]:>10.4f} {h['val_auc'].iloc[-1]:>10.4f} "
                  f"{best_val_auc:>11.4f} {best_ep:>7}")

    print_group("MODEL-SIZE SWEEP (depth=3, 300 epochs)", sweep)
    print_group("MLP ARCHITECTURES (200 epochs)", mlp_200)
    print_group("MLP ARCHITECTURES (500 epochs)", mlp_500)

    # Double descent verdict
    print(f"\n{'=' * 100}")
    print("DOUBLE DESCENT ANALYSIS")
    print(f"{'=' * 100}")

    if sweep:
        val_aucs = [(r['params'], r['history']['val_auc'].max(), r['label']) for r in sweep]
        val_losses = [(r['params'], r['history']['val_loss'].min(), r['label']) for r in sweep]

        # Find the worst point (valley) and check if later models recover
        worst_auc = min(val_aucs, key=lambda x: x[1])
        best_auc = max(val_aucs, key=lambda x: x[1])
        worst_loss = max(val_losses, key=lambda x: x[1])

        print(f"\n  Sweep results (best val AUC per model):")
        for p, auc, lbl in val_aucs:
            marker = " <-- WORST" if auc == worst_auc[1] else (" <-- BEST" if auc == best_auc[1] else "")
            ratio = p / sweep[0]['train_samples']
            print(f"    {lbl:<18} params={p:>12,}  ratio={ratio:>8.4f}  best_val_AUC={auc:.4f}{marker}")

        # Check for U-shape (non-monotonic)
        aucs_only = [x[1] for x in val_aucs]
        found_decrease = False
        found_recovery = False
        min_idx = aucs_only.index(min(aucs_only))
        if min_idx > 0:
            found_decrease = True
        if min_idx < len(aucs_only) - 1:
            if max(aucs_only[min_idx+1:]) > aucs_only[min_idx]:
                found_recovery = True

        if found_decrease and found_recovery:
            recovery = max(aucs_only[min_idx+1:]) - aucs_only[min_idx]
            print(f"\n  VERDICT: EVIDENCE OF DOUBLE DESCENT DETECTED")
            print(f"    Val AUC valley at {val_aucs[min_idx][2]} ({val_aucs[min_idx][0]:,} params)")
            print(f"    Recovery: +{recovery:.4f} AUC after the valley")
        elif found_decrease and not found_recovery:
            print(f"\n  VERDICT: MONOTONIC DEGRADATION — no recovery observed")
            print(f"    Larger models did not recover. May need more params or epochs.")
        else:
            print(f"\n  VERDICT: NO CLEAR OVERFITTING VALLEY — smallest models perform worst")

    return sweep, mlp_200, mlp_500


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    print("Loading results...")
    results = load_all_results()
    print(f"  Found {len(results)} experiment runs")

    sweep, mlp_200, mlp_500 = categorize_results(results)
    print(f"  Sweep: {len(sweep)} | 200ep: {len(mlp_200)} | 500ep: {len(mlp_500)}")

    # Summary table
    print_summary(sweep, mlp_200, mlp_500)

    # Generate plots
    print(f"\nGenerating plots in {PLOTS_DIR}...")
    if sweep:
        plot_model_wise_dd(sweep)
        plot_sweep_epoch_curves(sweep)
        plot_memorisation_map(sweep)
    if mlp_200 or mlp_500:
        plot_epoch_comparison(mlp_200, mlp_500)

    print(f"\nDone! All plots saved to: {PLOTS_DIR}")


if __name__ == '__main__':
    main()
