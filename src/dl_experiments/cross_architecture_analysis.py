"""
Cross-Architecture Double Descent Analysis
============================================
Comprehensive comparison of MLP, CNN, and LSTM experiments.

Analyses:
  1. Epoch-wise training curves (all 3 architectures, 6 panels)
  2. Model-wise double descent (val metrics vs params, all archs)
  3. Memorisation dynamics (train/val divergence trajectories)
  4. Architecture behaviour comparison (val loss stability)
  5. Best-epoch analysis (when each model peaks)
  6. Complete summary table + final verdict

Run:  conda activate dl_experiments
      python cross_architecture_analysis.py
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
from matplotlib.lines import Line2D

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis_plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════

# Canonical 500-epoch runs only (skip smoke tests, 200ep, sweep)
CANONICAL_RUNS = {
    # MLP 500ep
    'MLP_small':   'MLP_small_20260304_230955',
    'MLP_medium':  'MLP_medium_20260305_020255',
    'MLP_large':   'MLP_large_20260305_052037',
    'MLP_xlarge':  'MLP_xlarge_20260305_085756',
    'MLP_xxlarge': 'MLP_xxlarge_20260305_125314',
    # CNN 500ep
    'CNN_small':   'CNN_small_20260305_154215',
    'CNN_medium':  'CNN_medium_20260306_045016',
    'CNN_large':   'CNN_large_20260306_181400',
    'CNN_xlarge':  'CNN_xlarge_20260307_072412',
    # LSTM 500ep
    'LSTM_small':  'LSTM_small_20260307_214430',
    'LSTM_medium': 'LSTM_medium_20260308_130942',
    'LSTM_large':  'LSTM_large_20260309_031237',
    'LSTM_xlarge': 'LSTM_xlarge_20260309_171333',
}

# Also load the MLP width sweep (300 epochs)
SWEEP_RUNS = {
    f'mlp_w{w}_d3': None  # will be auto-detected
    for w in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
}


def load_run(base_name):
    """Load a single run by its base filename."""
    cfg_path = os.path.join(RESULTS_DIR, f'{base_name}_config.json')
    hist_path = os.path.join(RESULTS_DIR, f'{base_name}_history.csv')
    if not os.path.exists(cfg_path) or not os.path.exists(hist_path):
        return None
    with open(cfg_path) as f:
        config = json.load(f)
    history = pd.read_csv(hist_path)
    return {
        'config': config,
        'history': history,
        'label': config.get('label', base_name),
        'params': config.get('total_params', 0),
        'epochs': config.get('epochs', 0),
        'train_samples': config.get('train_samples', 0),
    }


def load_all():
    """Load canonical runs + sweep."""
    runs = {}
    for key, base in CANONICAL_RUNS.items():
        r = load_run(base)
        if r:
            runs[key] = r
            print(f"  Loaded: {key:<16} | {r['params']:>10,} params | {len(r['history'])} epochs")
        else:
            print(f"  MISSING: {key}")

    # Auto-detect sweep runs
    sweep = {}
    for pattern_key in SWEEP_RUNS:
        matches = glob.glob(os.path.join(RESULTS_DIR, f'{pattern_key}_*_config.json'))
        if matches:
            base = matches[0].replace('_config.json', '')
            base_name = os.path.basename(base)
            r = load_run(base_name)
            if r:
                sweep[pattern_key] = r

    sweep_sorted = dict(sorted(sweep.items(), key=lambda x: x[1]['params']))
    return runs, sweep_sorted


def get_arch(key):
    """Extract architecture family from key."""
    return key.split('_')[0]


def fmt_params(x, pos=None):
    if x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    return f'{x:.0f}'


# Consistent colours per architecture
ARCH_COLORS = {'MLP': '#1f77b4', 'CNN': '#ff7f0e', 'LSTM': '#2ca02c'}
SIZE_MARKERS = {'small': 'o', 'medium': 's', 'large': 'D', 'xlarge': '^', 'xxlarge': 'P'}
SIZE_ORDER = ['small', 'medium', 'large', 'xlarge', 'xxlarge']


# ════════════════════════════════════════════════════════════════════════
# PLOT 1: EPOCH-WISE CURVES — ALL ARCHITECTURES (3 × 2)
# ════════════════════════════════════════════════════════════════════════

def plot_epoch_curves_all(runs):
    """6-panel: Val Loss + Val AUC columns, MLP/CNN/LSTM rows."""
    archs = ['MLP', 'CNN', 'LSTM']
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    fig.suptitle('Epoch-Wise Training Curves — All Architectures (500 Epochs, Fixed LR=0.001)',
                 fontsize=16, fontweight='bold', y=0.98)

    size_colors = {
        'small': '#4daf4a', 'medium': '#377eb8', 'large': '#ff7f00',
        'xlarge': '#e41a1c', 'xxlarge': '#984ea3'
    }

    for row, arch in enumerate(archs):
        arch_runs = {k: v for k, v in runs.items() if get_arch(k) == arch}
        arch_runs_sorted = sorted(arch_runs.items(), key=lambda x: x[1]['params'])

        for key, r in arch_runs_sorted:
            h = r['history']
            size = key.split('_')[1]
            color = size_colors.get(size, 'gray')
            label = f"{size} ({fmt_params(r['params'])})"

            # Val loss
            axes[row, 0].plot(h['epoch'], h['val_loss'], color=color, linewidth=1.8,
                              label=label, alpha=0.9)
            axes[row, 0].plot(h['epoch'], h['loss'], color=color, linewidth=0.8,
                              linestyle='--', alpha=0.4)

            # Val AUC
            axes[row, 1].plot(h['epoch'], h['val_auc'], color=color, linewidth=1.8,
                              label=label, alpha=0.9)
            axes[row, 1].plot(h['epoch'], h['auc'], color=color, linewidth=0.8,
                              linestyle='--', alpha=0.4)

        # Formatting
        axes[row, 0].set_ylabel(f'{arch}\nLoss', fontsize=12, fontweight='bold')
        axes[row, 0].set_title(f'{arch} — Loss (solid=val, dashed=train)')
        axes[row, 0].legend(fontsize=8, loc='upper left')
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 0].set_xlim(0, 500)

        axes[row, 1].set_ylabel('AUC', fontsize=12)
        axes[row, 1].set_title(f'{arch} — AUC (solid=val, dashed=train)')
        axes[row, 1].axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        axes[row, 1].legend(fontsize=8, loc='upper right')
        axes[row, 1].grid(True, alpha=0.3)
        axes[row, 1].set_xlim(0, 500)
        axes[row, 1].set_ylim(0.45, 1.05)

    axes[2, 0].set_xlabel('Epoch', fontsize=12)
    axes[2, 1].set_xlabel('Epoch', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(PLOTS_DIR, '05_cross_arch_epoch_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# PLOT 2: MODEL-WISE DOUBLE DESCENT — ALL ARCHITECTURES
# ════════════════════════════════════════════════════════════════════════

def plot_model_wise_all(runs):
    """Val loss and val AUC vs params, coloured by architecture."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Model-Wise Double Descent — Across All Architectures (500 Epochs)',
                 fontsize=16, fontweight='bold')

    for arch in ['MLP', 'CNN', 'LSTM']:
        arch_runs = sorted(
            [(k, v) for k, v in runs.items() if get_arch(k) == arch],
            key=lambda x: x[1]['params']
        )
        if not arch_runs:
            continue

        params = [v['params'] for _, v in arch_runs]
        train_n = arch_runs[0][1]['train_samples']
        color = ARCH_COLORS[arch]

        best_val_auc = [v['history']['val_auc'].max() for _, v in arch_runs]
        final_val_auc = [v['history']['val_auc'].iloc[-1] for _, v in arch_runs]
        best_val_loss = [v['history']['val_loss'].min() for _, v in arch_runs]
        final_val_loss = [v['history']['val_loss'].iloc[-1] for _, v in arch_runs]
        final_train_loss = [v['history']['loss'].iloc[-1] for _, v in arch_runs]

        # (0,0) Best Val AUC vs Params
        axes[0, 0].plot(params, best_val_auc, '-o', color=color, label=arch,
                        markersize=8, linewidth=2.5)
        for i, (k, _) in enumerate(arch_runs):
            axes[0, 0].annotate(k.split('_')[1], (params[i], best_val_auc[i]),
                                fontsize=7, textcoords='offset points', xytext=(5, 5))

        # (0,1) Final Val AUC vs Params
        axes[0, 1].plot(params, final_val_auc, '-o', color=color, label=arch,
                        markersize=8, linewidth=2.5)

        # (1,0) Best Val Loss vs Params
        axes[1, 0].plot(params, best_val_loss, '-s', color=color, label=f'{arch} best',
                        markersize=7, linewidth=2)
        axes[1, 0].plot(params, final_val_loss, '--^', color=color, label=f'{arch} final',
                        markersize=6, linewidth=1.5, alpha=0.7)

        # (1,1) Final Train Loss vs Params (memorisation completeness)
        axes[1, 1].plot(params, final_train_loss, '-o', color=color, label=arch,
                        markersize=8, linewidth=2.5)

    # Formatting
    ax = axes[0, 0]
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_params))
    ax.set_xlabel('Model Parameters')
    ax.set_ylabel('Best Val AUC')
    ax.set_title('Best Validation AUC vs Model Size')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_params))
    ax.set_xlabel('Model Parameters')
    ax.set_ylabel('Final Val AUC (ep 500)')
    ax.set_title('Final Validation AUC vs Model Size')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    for _, v in runs.items():
        ax.annotate(f"  {v['label'].split('_')[1]}", 
                    (v['params'], v['history']['val_auc'].iloc[-1]),
                    fontsize=6, alpha=0.7)

    ax = axes[1, 0]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_params))
    ax.set_xlabel('Model Parameters')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs Model Size')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_params))
    ax.set_xlabel('Model Parameters')
    ax.set_ylabel('Final Train Loss')
    ax.set_title('Final Train Loss vs Model Size (Memorisation)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '06_cross_arch_model_wise_dd.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# PLOT 3: VAL LOSS STABILITY COMPARISON
# ════════════════════════════════════════════════════════════════════════

def plot_val_loss_stability(runs):
    """Key finding: LSTM val loss stable vs CNN/MLP exploding.
    Plot val loss for the 'large' variant of each architecture."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle('Val Loss Stability — Large Models Compared (500 Epochs)',
                 fontsize=14, fontweight='bold')

    targets = ['MLP_xlarge', 'CNN_xlarge', 'LSTM_xlarge']
    titles = ['MLP xlarge (872K params)', 'CNN xlarge (1.22M params)', 'LSTM xlarge (743K params)']

    for i, (key, title) in enumerate(zip(targets, titles)):
        if key not in runs:
            continue
        h = runs[key]['history']
        ax = axes[i]

        # Plot val loss with moving average
        ax.plot(h['epoch'], h['val_loss'], alpha=0.3, color=ARCH_COLORS[get_arch(key)],
                linewidth=0.8, label='Val loss (raw)')
        # Rolling average
        window = 20
        if len(h) > window:
            rolling = h['val_loss'].rolling(window=window, center=True).mean()
            ax.plot(h['epoch'], rolling, color=ARCH_COLORS[get_arch(key)],
                    linewidth=2.5, label=f'Val loss ({window}-ep avg)')

        ax.plot(h['epoch'], h['loss'], color='gray', linewidth=1, alpha=0.5,
                label='Train loss')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 500)

        # Annotate final values
        final_val = h['val_loss'].iloc[-1]
        final_train = h['loss'].iloc[-1]
        ax.annotate(f'Final val: {final_val:.3f}\nFinal train: {final_train:.4f}',
                    xy=(0.98, 0.95), xycoords='axes fraction', fontsize=9,
                    ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3',
                    facecolor='white', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '07_val_loss_stability.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# PLOT 4: MEMORISATION SPEED (Train AUC → 1.0)
# ════════════════════════════════════════════════════════════════════════

def plot_memorisation_speed(runs):
    """How quickly each architecture memorises training data."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle('Memorisation Speed — Train AUC Convergence to 1.0 (500 Epochs)',
                 fontsize=14, fontweight='bold')

    size_colors = {
        'small': '#4daf4a', 'medium': '#377eb8', 'large': '#ff7f00',
        'xlarge': '#e41a1c', 'xxlarge': '#984ea3'
    }

    for col, arch in enumerate(['MLP', 'CNN', 'LSTM']):
        ax = axes[col]
        arch_runs = sorted(
            [(k, v) for k, v in runs.items() if get_arch(k) == arch],
            key=lambda x: x[1]['params']
        )
        for key, r in arch_runs:
            h = r['history']
            size = key.split('_')[1]
            ax.plot(h['epoch'], h['auc'], color=size_colors.get(size, 'gray'),
                    linewidth=2, label=f"{size} ({fmt_params(r['params'])})")

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Train AUC', fontsize=11)
        ax.set_title(f'{arch}', fontsize=13, fontweight='bold')
        ax.set_ylim(0.5, 1.02)
        ax.set_xlim(0, 500)
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.4)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '08_memorisation_speed.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# PLOT 5: BEST EPOCH ANALYSIS
# ════════════════════════════════════════════════════════════════════════

def plot_best_epoch_analysis(runs):
    """When does each model achieve its best val AUC?"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Best Validation AUC — When and How Much',
                 fontsize=14, fontweight='bold')

    data = []
    for key, r in runs.items():
        h = r['history']
        best_idx = h['val_auc'].idxmax()
        data.append({
            'model': key,
            'arch': get_arch(key),
            'params': r['params'],
            'best_val_auc': h['val_auc'].max(),
            'best_epoch': h.loc[best_idx, 'epoch'],
            'final_val_auc': h['val_auc'].iloc[-1],
            'degradation': h['val_auc'].max() - h['val_auc'].iloc[-1],
        })

    df = pd.DataFrame(data).sort_values('params')

    # (0) Best val AUC with best epoch annotation  
    ax = axes[0]
    for arch in ['MLP', 'CNN', 'LSTM']:
        sub = df[df['arch'] == arch]
        ax.barh(sub['model'], sub['best_val_auc'], color=ARCH_COLORS[arch],
                alpha=0.8, label=arch, edgecolor='white', linewidth=0.5)
        for _, row in sub.iterrows():
            ax.annotate(f"ep {int(row['best_epoch'])}",
                        xy=(row['best_val_auc'], row['model']),
                        fontsize=8, va='center', ha='left',
                        xytext=(5, 0), textcoords='offset points')

    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('Best Val AUC', fontsize=12)
    ax.set_title('Best Val AUC by Model (annotated with best epoch)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0.5, 0.7)

    # (1) Degradation: best → final
    ax = axes[1]
    for arch in ['MLP', 'CNN', 'LSTM']:
        sub = df[df['arch'] == arch]
        ax.barh(sub['model'], sub['degradation'], color=ARCH_COLORS[arch],
                alpha=0.8, label=arch, edgecolor='white', linewidth=0.5)
        for _, row in sub.iterrows():
            ax.annotate(f"{row['final_val_auc']:.3f}",
                        xy=(row['degradation'], row['model']),
                        fontsize=8, va='center', ha='left',
                        xytext=(5, 0), textcoords='offset points')

    ax.set_xlabel('AUC Degradation (Best − Final)', fontsize=12)
    ax.set_title('Val AUC Degradation After Overfitting (annotated: final AUC)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '09_best_epoch_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return df


# ════════════════════════════════════════════════════════════════════════
# PLOT 6: OVERPARAMETERISATION RATIO — UNIFIED VIEW
# ════════════════════════════════════════════════════════════════════════

def plot_overparameterisation(runs, sweep):
    """Val metrics vs overparameterisation ratio for all archs + sweep."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Double Descent Check — Val Metrics vs Overparameterisation Ratio',
                 fontsize=14, fontweight='bold')

    # Sweep (light gray background)
    if sweep:
        s_sorted = sorted(sweep.values(), key=lambda x: x['params'])
        s_ratios = [r['params'] / r['train_samples'] for r in s_sorted]
        s_best_auc = [r['history']['val_auc'].max() for r in s_sorted]
        s_best_loss = [r['history']['val_loss'].min() for r in s_sorted]
        axes[0].plot(s_ratios, s_best_loss, 'o-', color='gray', alpha=0.5,
                     linewidth=1.5, markersize=5, label='MLP sweep (300ep)')
        axes[1].plot(s_ratios, s_best_auc, 'o-', color='gray', alpha=0.5,
                     linewidth=1.5, markersize=5, label='MLP sweep (300ep)')

    # Main runs
    for arch in ['MLP', 'CNN', 'LSTM']:
        arch_runs = sorted(
            [(k, v) for k, v in runs.items() if get_arch(k) == arch],
            key=lambda x: x[1]['params']
        )
        if not arch_runs:
            continue

        ratios = [v['params'] / v['train_samples'] for _, v in arch_runs]
        best_auc = [v['history']['val_auc'].max() for _, v in arch_runs]
        best_loss = [v['history']['val_loss'].min() for _, v in arch_runs]

        axes[0].plot(ratios, best_loss, '-o', color=ARCH_COLORS[arch],
                     markersize=9, linewidth=2.5, label=f'{arch} (500ep)')
        axes[1].plot(ratios, best_auc, '-o', color=ARCH_COLORS[arch],
                     markersize=9, linewidth=2.5, label=f'{arch} (500ep)')

    # Interpolation threshold
    for ax in axes:
        ax.axvline(1.0, color='red', linestyle=':', linewidth=2, alpha=0.6,
                   label='Interpolation threshold (p/n=1)')
        ax.set_xscale('log')
        ax.set_xlabel('Params / Train Samples', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Best Val Loss', fontsize=12)
    axes[0].set_title('Best Validation Loss vs Overparameterisation')
    axes[0].set_yscale('log')

    axes[1].set_ylabel('Best Val AUC', fontsize=12)
    axes[1].set_title('Best Validation AUC vs Overparameterisation')
    axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.4)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '10_overparameterisation_ratio.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# PLOT 7: COMBINED HERO PLOT (val AUC all models on one chart)
# ════════════════════════════════════════════════════════════════════════

def plot_hero_val_auc(runs):
    """Single chart: val AUC over epochs for all 13 models."""
    fig, ax = plt.subplots(figsize=(22, 10))
    ax.set_title('Validation AUC — All 13 Models × 500 Epochs\n'
                 '(No regularisation, no LR scheduling, fixed LR=0.001)',
                 fontsize=14, fontweight='bold')

    line_styles = {'MLP': '-', 'CNN': '--', 'LSTM': '-.'}
    size_alphas = {'small': 0.5, 'medium': 0.65, 'large': 0.8, 'xlarge': 0.95, 'xxlarge': 1.0}
    size_widths = {'small': 1.2, 'medium': 1.5, 'large': 2.0, 'xlarge': 2.5, 'xxlarge': 2.8}

    for key in sorted(runs.keys(), key=lambda k: (get_arch(k), runs[k]['params'])):
        r = runs[key]
        h = r['history']
        arch = get_arch(key)
        size = key.split('_')[1]

        ax.plot(h['epoch'], h['val_auc'],
                color=ARCH_COLORS[arch],
                linestyle=line_styles[arch],
                alpha=size_alphas.get(size, 0.7),
                linewidth=size_widths.get(size, 1.5),
                label=f"{key} ({fmt_params(r['params'])})")

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation AUC', fontsize=12)
    ax.set_xlim(0, 500)
    ax.set_ylim(0.48, 0.70)
    ax.legend(fontsize=8, loc='upper right', ncol=3, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Add architecture legend separately
    custom_lines = [
        Line2D([0], [0], color=ARCH_COLORS['MLP'], linestyle='-', linewidth=3),
        Line2D([0], [0], color=ARCH_COLORS['CNN'], linestyle='--', linewidth=3),
        Line2D([0], [0], color=ARCH_COLORS['LSTM'], linestyle='-.', linewidth=3),
    ]
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.legend(custom_lines, ['MLP (flat)', 'CNN (seq)', 'LSTM (seq)'],
               fontsize=11, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '11_hero_val_auc_all_models.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════

def print_summary_table(runs):
    """Comprehensive summary of all 13 models."""
    print("\n" + "=" * 130)
    print("CROSS-ARCHITECTURE DOUBLE DESCENT EXPERIMENT — COMPLETE RESULTS")
    print("=" * 130)
    print(f"  Total compute time: ~112.5 hours (4.7 days)")
    print(f"  GPU: NVIDIA RTX 4060 Laptop 8GB | PyTorch 2.6.0+cu124")
    print(f"  Config: 500 epochs, LR=0.001 (fixed), no dropout, no batch norm, no L2")
    print(f"  Data: 178 features, binary target (outperform_1d > 2%), 10.4% positive rate")

    rows = []
    for key, r in sorted(runs.items(), key=lambda x: (get_arch(x[0]), x[1]['params'])):
        h = r['history']
        best_idx = h['val_auc'].idxmax()
        ratio = r['params'] / r['train_samples'] if r['train_samples'] else 0
        rows.append({
            'Model': key,
            'Arch': get_arch(key),
            'Params': r['params'],
            'Ratio': ratio,
            'Train Loss': h['loss'].iloc[-1],
            'Val Loss': h['val_loss'].iloc[-1],
            'Train AUC': h['auc'].iloc[-1],
            'Final Val AUC': h['val_auc'].iloc[-1],
            'Best Val AUC': h['val_auc'].max(),
            'Best Epoch': int(h.loc[best_idx, 'epoch']),
            'Degradation': h['val_auc'].max() - h['val_auc'].iloc[-1],
        })

    df = pd.DataFrame(rows)

    for arch in ['MLP', 'CNN', 'LSTM']:
        sub = df[df['Arch'] == arch]
        print(f"\n{'─' * 130}")
        data_type = 'flat' if arch == 'MLP' else 'sequence (60-day windows)'
        n_train = sub.iloc[0]['Params']  # just for header
        print(f"  {arch} — {data_type}")
        print(f"{'─' * 130}")
        print(f"  {'Model':<16} {'Params':>10} {'P/N Ratio':>10} "
              f"{'Train Loss':>11} {'Val Loss':>10} {'Train AUC':>10} "
              f"{'Val AUC':>9} {'Best AUC':>9} {'Best Ep':>8} {'Degrad':>8}")
        print(f"  {'-'*14:<16} {'-'*10:>10} {'-'*10:>10} "
              f"{'-'*11:>11} {'-'*10:>10} {'-'*10:>10} "
              f"{'-'*9:>9} {'-'*9:>9} {'-'*8:>8} {'-'*8:>8}")
        for _, row in sub.iterrows():
            print(f"  {row['Model']:<16} {row['Params']:>10,} {row['Ratio']:>10.4f} "
                  f"{row['Train Loss']:>11.4f} {row['Val Loss']:>10.4f} "
                  f"{row['Train AUC']:>10.4f} {row['Final Val AUC']:>9.4f} "
                  f"{row['Best Val AUC']:>9.4f} {row['Best Epoch']:>8} "
                  f"{row['Degradation']:>8.4f}")

    # Cross-architecture comparison
    print(f"\n{'=' * 130}")
    print("CROSS-ARCHITECTURE COMPARISON")
    print(f"{'=' * 130}")

    for arch in ['MLP', 'CNN', 'LSTM']:
        sub = df[df['Arch'] == arch]
        best_row = sub.loc[sub['Best Val AUC'].idxmax()]
        print(f"\n  {arch} best: {best_row['Model']} — Val AUC {best_row['Best Val AUC']:.4f} "
              f"at epoch {best_row['Best Epoch']}")
        print(f"    Final val AUC at ep 500: {best_row['Final Val AUC']:.4f} "
              f"(degradation: -{best_row['Degradation']:.4f})")

    # Overall best
    overall_best = df.loc[df['Best Val AUC'].idxmax()]
    print(f"\n  OVERALL BEST: {overall_best['Model']} — Val AUC {overall_best['Best Val AUC']:.4f} "
          f"at epoch {overall_best['Best Epoch']}")

    return df


def print_verdict(df):
    """Final double descent verdict across all architectures."""
    print(f"\n{'=' * 130}")
    print("DOUBLE DESCENT VERDICT")
    print(f"{'=' * 130}")

    print("""
  EPOCH-WISE DOUBLE DESCENT:
  ─────────────────────────
  Classic double descent predicts: val loss ↑ (overfitting) → peak → ↓ (second descent).
  
  MLP:  Val loss rises monotonically after epoch ~5. No recovery at 500 epochs.
        Val AUC peaks at epoch 1-4, then degrades steadily. NO double descent.
  
  CNN:  Nearly identical to MLP. Val loss explodes, val AUC peaks at epoch 1.
        The convolutional structure adds no benefit on this tabular data. NO double descent.
  
  LSTM: DIFFERENT BEHAVIOUR — val loss remarkably stable (doesn't explode like MLP/CNN).
        Val AUC still peaks at epoch 1, but degradation is slower and gentler.
        LSTM_small val AUC actually rises slightly from ep 250-500 (0.575 → 0.582).
        However, this is NOT classic double descent — it's regularisation from
        recurrent structure (vanishing gradients act as implicit regularisation).
        NO classic double descent, but LSTM shows unique stability.

  MODEL-WISE DOUBLE DESCENT:
  ──────────────────────────
  Classic double descent predicts: val loss ↑ as params approach n_train → then ↓ past it.
  
  No architecture shows the U-shaped curve. For all three:
    - Best val AUC is essentially flat across model sizes (0.62-0.65 range)
    - Larger models don't improve AND don't show the characteristic valley-then-recovery
    - CNN_xlarge (1.22M params, ratio 1.22×) is past the interpolation threshold
      but shows no improvement over CNN_small
  
  WHY NO DOUBLE DESCENT:
  ─────────────────────
  1. SIGNAL-TO-NOISE RATIO: Financial return prediction has extremely low SNR.
     The "true signal" is only ~0.62-0.65 AUC worth of information — additional 
     model capacity memorises noise, not structure.
  
  2. INSUFFICIENT OVERPARAMETERISATION: The largest models (MLP_xxlarge at 3.1M
     params vs 1.03M training samples, ratio 3.0×) may still not be sufficiently
     overparameterised. Double descent in vision/NLP often requires 10-100× ratios.
  
  3. DATA TYPE: Double descent is most reliably observed with structured data
     (images, text) where models can discover hierarchical features. Financial
     tabular data may lack this hierarchical structure.
  
  4. BINARY TARGET: Our highly imbalanced target (10.4% positive) with a 2% threshold
     may create a decision boundary that doesn't benefit from overparameterisation.

  PRACTICAL FINDINGS:
  ──────────────────
  • ALL architectures achieve their best val AUC at epochs 1-4
  • More epochs and bigger models do NOT help — early stopping is essential
  • Best achievable: ~0.64 AUC (LSTM_large) — consistent across all architectures
  • LSTM's stable val loss is interesting but doesn't translate to better AUC
  • For this dataset, a simple small model trained for 5 epochs ≈ a large model trained for 500
""")

    # Save summary CSV
    path = os.path.join(PLOTS_DIR, 'cross_architecture_summary.csv')
    df.to_csv(path, index=False)
    print(f"  Summary CSV saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("CROSS-ARCHITECTURE DOUBLE DESCENT ANALYSIS")
    print("=" * 80)
    print("\nLoading results...")

    runs, sweep = load_all()
    print(f"\n  Loaded {len(runs)} canonical 500-epoch runs")
    print(f"  Loaded {len(sweep)} MLP sweep runs")

    # Summary table
    df = print_summary_table(runs)

    # Generate plots
    print(f"\nGenerating analysis plots in {PLOTS_DIR}...")
    plot_epoch_curves_all(runs)       # Plot 5: 3×2 epoch curves
    plot_model_wise_all(runs)          # Plot 6: model-wise DD
    plot_val_loss_stability(runs)      # Plot 7: val loss stability
    plot_memorisation_speed(runs)      # Plot 8: train AUC convergence
    best_df = plot_best_epoch_analysis(runs)  # Plot 9: best epoch
    plot_overparameterisation(runs, sweep)     # Plot 10: ratio view
    plot_hero_val_auc(runs)            # Plot 11: hero chart

    # Final verdict
    print_verdict(df)

    print(f"\n{'=' * 80}")
    print(f"ANALYSIS COMPLETE — {7} plots saved to {PLOTS_DIR}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
