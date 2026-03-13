"""
Run Deep Learning Double-Descent Experiments (PyTorch + CUDA)
==============================================================

Entry point for all DL experiments. Orchestrates data loading, model
building, training on GPU, and result saving.

Usage (run from dl_experiments conda env):
  python run_dl_experiment.py                  # Quick smoke test (5 epochs)
  python run_dl_experiment.py --mode mlp       # MLP experiments
  python run_dl_experiment.py --mode cnn       # 1D-CNN experiments
  python run_dl_experiment.py --mode lstm      # LSTM experiments
  python run_dl_experiment.py --mode sweep     # Model-size sweep (MLP)
  python run_dl_experiment.py --mode all       # All architectures
  python run_dl_experiment.py --mode quick     # Quick smoke test

All results saved to #_dl_experiments/results/

Created: March 2026
"""

import os
import sys
import argparse
import time
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dl_data_loader import DLDataLoader
from dl_double_descent_trainer import (
    DoubleDescentTrainer,
    build_mlp, build_cnn, build_lstm,
    plot_loss_curves, plot_model_wise_dd,
)


# ════════════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIGS
# ════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    'epochs': 500,
    'batch_size': 1024,
    'learning_rate': 0.001,
    'dropout': 0.0,
    'l2_reg': 0.0,
    'use_batch_norm': False,
    'use_ticker_embedding': False,
    'target_days': 1,
    'target_threshold': 0.02,
    'train_ratio': 0.80,
    'sequence_length': 60,
}

QUICK_CONFIG = {**DEFAULT_CONFIG, 'epochs': 5, 'batch_size': 2048}


# ════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNERS
# ════════════════════════════════════════════════════════════════════════

def run_mlp_experiment(data_flat, n_features, config, trainer):
    print("\n" + "#" * 70)
    print("EXPERIMENT: MLP (Fully Connected) -- Double Descent")
    print("#" * 70)

    architectures = [
        ('MLP_small',   [64, 32]),
        ('MLP_medium',  [256, 128, 64]),
        ('MLP_large',   [512, 256, 128]),
        ('MLP_xlarge',  [1024, 512, 256, 128]),
        ('MLP_xxlarge', [2048, 1024, 512, 256]),
    ]

    results = []
    for label, hidden in architectures:
        model = build_mlp(
            n_features=n_features, hidden_layers=hidden,
            dropout=config['dropout'],
            use_batch_norm=config['use_batch_norm'],
        )
        result = trainer.train(
            model=model, data=data_flat,
            epochs=config['epochs'], batch_size=config['batch_size'],
            learning_rate=config['learning_rate'], label=label,
        )
        results.append(result)
        del model
        torch.cuda.empty_cache()

    plot_loss_curves(results, trainer.output_dir, title='MLP_Epoch_Descent')
    plot_model_wise_dd(results, trainer.output_dir, title='MLP_Model_Descent')
    return results


def run_cnn_experiment(n_features, seq_len, config, trainer,
                       train_loader=None, test_loader=None, seq_info=None):
    print("\n" + "#" * 70)
    print("EXPERIMENT: 1D-CNN -- Double Descent")
    print("#" * 70)

    architectures = [
        ('CNN_small',  [32, 64],           [64]),
        ('CNN_medium', [64, 128, 64],      [128]),
        ('CNN_large',  [128, 256, 128],    [256, 128]),
        ('CNN_xlarge', [256, 512, 256, 128], [512, 256]),
    ]

    results = []
    for label, filters, dense in architectures:
        model = build_cnn(
            n_features=n_features, seq_len=seq_len,
            filters=filters, dense_units=dense,
            dropout=config['dropout'],
            use_batch_norm=config['use_batch_norm'],
        )
        result = trainer.train(
            model=model, epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'], label=label,
            is_sequence=True,
            train_loader=train_loader, test_loader=test_loader,
            n_train=seq_info.get('n_train') if seq_info else None,
            n_test=seq_info.get('n_test') if seq_info else None,
        )
        results.append(result)
        del model
        torch.cuda.empty_cache()

    plot_loss_curves(results, trainer.output_dir, title='CNN_Epoch_Descent')
    plot_model_wise_dd(results, trainer.output_dir, title='CNN_Model_Descent')
    return results


def run_lstm_experiment(n_features, seq_len, config, trainer,
                        train_loader=None, test_loader=None, seq_info=None):
    print("\n" + "#" * 70)
    print("EXPERIMENT: LSTM -- Double Descent")
    print("#" * 70)

    architectures = [
        ('LSTM_small',  [32, 16],       [32]),
        ('LSTM_medium', [64, 32],       [64]),
        ('LSTM_large',  [128, 64, 32],  [128]),
        ('LSTM_xlarge', [256, 128, 64], [256, 128]),
    ]

    results = []
    for label, lstm_units, dense in architectures:
        model = build_lstm(
            n_features=n_features, lstm_units=lstm_units,
            dense_units=dense, dropout=config['dropout'],
            use_batch_norm=config['use_batch_norm'],
        )
        result = trainer.train(
            model=model, epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'], label=label,
            is_sequence=True,
            train_loader=train_loader, test_loader=test_loader,
            n_train=seq_info.get('n_train') if seq_info else None,
            n_test=seq_info.get('n_test') if seq_info else None,
        )
        results.append(result)
        del model
        torch.cuda.empty_cache()

    plot_loss_curves(results, trainer.output_dir, title='LSTM_Epoch_Descent')
    plot_model_wise_dd(results, trainer.output_dir, title='LSTM_Model_Descent')
    return results


def run_sweep_experiment(data_flat, n_features, config, trainer):
    print("\n" + "#" * 70)
    print("EXPERIMENT: MODEL-SIZE SWEEP (MLP)")
    print("#" * 70)

    results = trainer.run_model_size_sweep(
        arch='mlp', data=data_flat,
        widths=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        depth=3, epochs=config['epochs'],
        batch_size=config['batch_size'], learning_rate=config['learning_rate'],
    )

    plot_loss_curves(results, trainer.output_dir, title='MLP_Width_Sweep')
    plot_model_wise_dd(results, trainer.output_dir, title='MLP_Width_Sweep')
    return results


def run_quick_test(data_flat, n_features, config, trainer):
    print("\n" + "#" * 70)
    print("QUICK TEST -- 5 epochs, small model, GPU verification")
    print("#" * 70)

    model = build_mlp(n_features=n_features, hidden_layers=[64, 32])
    result = trainer.train(
        model=model, data=data_flat,
        epochs=config['epochs'], batch_size=config['batch_size'],
        learning_rate=config['learning_rate'], label='quick_test_gpu',
    )
    return [result]


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='DL Double-Descent Experiments')
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['mlp', 'cnn', 'lstm', 'sweep', 'all', 'quick'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--target_days', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=None)

    args = parser.parse_args()

    config = QUICK_CONFIG.copy() if args.mode == 'quick' else DEFAULT_CONFIG.copy()
    if args.epochs is not None: config['epochs'] = args.epochs
    if args.batch_size is not None: config['batch_size'] = args.batch_size
    if args.lr is not None: config['learning_rate'] = args.lr
    if args.seq_len is not None: config['sequence_length'] = args.seq_len
    if args.dropout is not None: config['dropout'] = args.dropout
    if args.target_days is not None: config['target_days'] = args.target_days
    if args.threshold is not None: config['target_threshold'] = args.threshold

    # ── Header ──────────────────────────────────────────────────────
    print("=" * 70)
    print("DEEP LEARNING DOUBLE-DESCENT EXPERIMENT (PyTorch + CUDA)")
    print("=" * 70)
    print(f"  Mode:       {args.mode}")
    print(f"  Epochs:     {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  LR:         {config['learning_rate']}")
    print(f"  Dropout:    {config['dropout']}")
    print(f"  Target:     outperform_{config['target_days']}d > {config['target_threshold']*100:.1f}%")
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  CUDA:       {torch.cuda.is_available()}", end='')
    if torch.cuda.is_available():
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()
    print(f"  Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    t0 = time.time()

    # ── Load data ───────────────────────────────────────────────────
    loader = DLDataLoader(
        train_ratio=config['train_ratio'],
        target_days=config['target_days'],
        target_threshold=config['target_threshold'],
        sequence_length=config['sequence_length'],
        verbose=True,
    )
    loader.load()

    data_flat = loader.get_flat_data()
    n_features = data_flat['X_train'].shape[1]

    # Build lazy sequence loaders for CNN/LSTM (memory-efficient)
    seq_train_loader, seq_test_loader, seq_info = None, None, None
    need_seq = args.mode in ('cnn', 'lstm', 'all')
    if need_seq:
        print("\nBuilding lazy sequence index (memory-efficient)...")
        seq_train_loader, seq_test_loader, seq_info = (
            loader.get_lazy_sequence_dataloaders(
                batch_size=config['batch_size'],
                seq_len=config['sequence_length'],
            )
        )

    # ── Trainer ─────────────────────────────────────────────────────
    trainer = DoubleDescentTrainer(experiment_name=f'dd_{args.mode}', verbose=True)

    # ── Run experiments ─────────────────────────────────────────────
    all_results = []

    if args.mode == 'quick':
        all_results = run_quick_test(data_flat, n_features, config, trainer)
    elif args.mode == 'mlp':
        all_results = run_mlp_experiment(data_flat, n_features, config, trainer)
    elif args.mode == 'cnn':
        all_results = run_cnn_experiment(
            n_features, config['sequence_length'], config, trainer,
            train_loader=seq_train_loader, test_loader=seq_test_loader,
            seq_info=seq_info)
    elif args.mode == 'lstm':
        all_results = run_lstm_experiment(
            n_features, config['sequence_length'], config, trainer,
            train_loader=seq_train_loader, test_loader=seq_test_loader,
            seq_info=seq_info)
    elif args.mode == 'sweep':
        all_results = run_sweep_experiment(data_flat, n_features, config, trainer)
    elif args.mode == 'all':
        r1 = run_mlp_experiment(data_flat, n_features, config, trainer)
        r2 = run_cnn_experiment(
            n_features, config['sequence_length'], config, trainer,
            train_loader=seq_train_loader, test_loader=seq_test_loader,
            seq_info=seq_info)
        r3 = run_lstm_experiment(
            n_features, config['sequence_length'], config, trainer,
            train_loader=seq_train_loader, test_loader=seq_test_loader,
            seq_info=seq_info)
        all_results = r1 + r2 + r3
        plot_loss_curves(all_results, trainer.output_dir, title='All_Architectures')

    # ── Summary ─────────────────────────────────────────────────────
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"  Mode:           {args.mode}")
    print(f"  Models trained: {len(all_results)}")
    print(f"  Total time:     {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Results dir:    {trainer.output_dir}")
    print("=" * 70)

    print(f"\n{'Model':<22} {'Params':>10} {'Train Loss':>12} {'Val Loss':>12} "
          f"{'Train AUC':>11} {'Val AUC':>11}")
    print("-" * 80)
    for r in all_results:
        last = r['history'][-1]
        print(f"{r['config']['label']:<22} {r['config']['total_params']:>10,} "
              f"{last['loss']:>12.4f} {last['val_loss']:>12.4f} "
              f"{last['auc']:>11.4f} {last['val_auc']:>11.4f}")

    print(f"\nNext steps:")
    print(f"  1. Check results in {trainer.output_dir}")
    print(f"  2. Look for the U-shaped val_loss curve (double descent)")
    print(f"  3. Try: --mode mlp --epochs 200 (first real experiment)")
    print(f"  4. Then: --mode sweep --epochs 300 (model-size double descent)")


if __name__ == '__main__':
    main()
