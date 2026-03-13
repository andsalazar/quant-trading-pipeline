"""
Deep Learning Double-Descent Trainer (PyTorch + CUDA)
======================================================

Implements MLP, 1D-CNN, and LSTM architectures designed to explore the
double-descent phenomenon on financial time-series data.

Key design principles:
  1. NO early stopping by default — train to interpolation and beyond
  2. Configurable regularisation (off by default for pure double-descent)
  3. Full epoch-by-epoch logging of train AND test metrics
  4. Systematic model-size sweeps for model-wise double descent
  5. GPU-accelerated via PyTorch CUDA on RTX 4060

Created: March 2026
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from dl_data_loader import FlatDataset, SequenceDataset


# ════════════════════════════════════════════════════════════════════════
# DEVICE SETUP
# ════════════════════════════════════════════════════════════════════════

def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"  Device: CPU")
    return device


# ════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """Fully-connected MLP with configurable depth/width."""

    def __init__(self, n_features, hidden_layers=(512, 256, 128),
                 dropout=0.0, use_batch_norm=False,
                 n_symbols=0, embedding_dim=16):
        super().__init__()

        self.use_embedding = n_symbols > 0
        if self.use_embedding:
            self.embedding = nn.Embedding(n_symbols, embedding_dim)
            input_dim = n_features + embedding_dim
        else:
            input_dim = n_features

        layers = []
        prev_dim = input_dim
        for units in hidden_layers:
            layers.append(nn.Linear(prev_dim, units))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = units

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features, ticker_id=None):
        x = features
        if self.use_embedding and ticker_id is not None:
            emb = self.embedding(ticker_id.squeeze(-1))
            x = torch.cat([x, emb], dim=-1)
        return self.network(x).squeeze(-1)


class CNN1D(nn.Module):
    """1-D CNN for temporal sequences."""

    def __init__(self, n_features, seq_len, filters=(64, 128, 64),
                 kernel_size=3, dense_units=(128,), dropout=0.0,
                 use_batch_norm=False, n_symbols=0, embedding_dim=16):
        super().__init__()

        self.use_embedding = n_symbols > 0
        if self.use_embedding:
            self.embedding = nn.Embedding(n_symbols, embedding_dim)

        # Conv layers: input is (batch, channels=n_features, seq_len)
        conv_layers = []
        in_ch = n_features
        for f in filters:
            conv_layers.append(nn.Conv1d(in_ch, f, kernel_size, padding='same'))
            if use_batch_norm:
                conv_layers.append(nn.BatchNorm1d(f))
            conv_layers.append(nn.ReLU())
            if dropout > 0:
                conv_layers.append(nn.Dropout(dropout))
            conv_layers.append(nn.MaxPool1d(2))
            in_ch = f
        self.conv = nn.Sequential(*conv_layers)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Dense head
        dense_input = in_ch + (embedding_dim if self.use_embedding else 0)
        dense_layers = []
        prev = dense_input
        for units in dense_units:
            dense_layers.append(nn.Linear(prev, units))
            if use_batch_norm:
                dense_layers.append(nn.BatchNorm1d(units))
            dense_layers.append(nn.ReLU())
            if dropout > 0:
                dense_layers.append(nn.Dropout(dropout))
            prev = units
        dense_layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*dense_layers)

    def forward(self, features, ticker_id=None):
        # features: (batch, seq_len, n_features) -> Conv1D needs (batch, n_features, seq_len)
        x = features.permute(0, 2, 1)
        x = self.conv(x)
        x = self.gap(x).squeeze(-1)  # (batch, channels)

        if self.use_embedding and ticker_id is not None:
            emb = self.embedding(ticker_id.squeeze(-1))
            x = torch.cat([x, emb], dim=-1)

        return self.head(x).squeeze(-1)


class LSTMModel(nn.Module):
    """Stacked LSTM for temporal sequences."""

    def __init__(self, n_features, lstm_units=(128, 64), dense_units=(64,),
                 dropout=0.0, use_batch_norm=False,
                 n_symbols=0, embedding_dim=16):
        super().__init__()

        self.use_embedding = n_symbols > 0
        if self.use_embedding:
            self.embedding = nn.Embedding(n_symbols, embedding_dim)

        # Stacked LSTM
        self.lstm_layers = nn.ModuleList()
        self.lstm_bns = nn.ModuleList() if use_batch_norm else None
        self.lstm_drops = nn.ModuleList() if dropout > 0 else None

        input_size = n_features
        for i, units in enumerate(lstm_units):
            self.lstm_layers.append(
                nn.LSTM(input_size, units, batch_first=True)
            )
            if use_batch_norm:
                self.lstm_bns.append(nn.BatchNorm1d(units))
            if dropout > 0:
                self.lstm_drops.append(nn.Dropout(dropout))
            input_size = units

        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout

        # Dense head
        dense_input = lstm_units[-1] + (embedding_dim if self.use_embedding else 0)
        dense_layers = []
        prev = dense_input
        for units in dense_units:
            dense_layers.append(nn.Linear(prev, units))
            if use_batch_norm:
                dense_layers.append(nn.BatchNorm1d(units))
            dense_layers.append(nn.ReLU())
            if dropout > 0:
                dense_layers.append(nn.Dropout(dropout))
            prev = units
        dense_layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*dense_layers)

    def forward(self, features, ticker_id=None):
        # features: (batch, seq_len, n_features)
        x = features
        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)
            if i < len(self.lstm_layers) - 1:
                # Pass full sequence to next layer
                if self.use_batch_norm and self.lstm_bns is not None:
                    # BN on (batch, features) for each timestep
                    x = x.permute(0, 2, 1)
                    x = self.lstm_bns[i](x)
                    x = x.permute(0, 2, 1)
                if self.lstm_drops is not None:
                    x = self.lstm_drops[i](x)
            else:
                # Last layer: take final hidden state
                x = x[:, -1, :]
                if self.use_batch_norm and self.lstm_bns is not None:
                    x = self.lstm_bns[i](x)
                if self.lstm_drops is not None:
                    x = self.lstm_drops[i](x)

        if self.use_embedding and ticker_id is not None:
            emb = self.embedding(ticker_id.squeeze(-1))
            x = torch.cat([x, emb], dim=-1)

        return self.head(x).squeeze(-1)


# ════════════════════════════════════════════════════════════════════════
# MODEL BUILDERS (convenience functions)
# ════════════════════════════════════════════════════════════════════════

def build_mlp(n_features, hidden_layers=(512, 256, 128), **kwargs):
    return MLP(n_features, hidden_layers=hidden_layers, **kwargs)


def build_cnn(n_features, seq_len, filters=(64, 128, 64), **kwargs):
    return CNN1D(n_features, seq_len, filters=filters, **kwargs)


def build_lstm(n_features, lstm_units=(128, 64), **kwargs):
    return LSTMModel(n_features, lstm_units=lstm_units, **kwargs)


# ════════════════════════════════════════════════════════════════════════
# TRAINER
# ════════════════════════════════════════════════════════════════════════

class DoubleDescentTrainer:
    """Train deep learning models and log full loss curves."""

    def __init__(self, output_dir=None, experiment_name='experiment', verbose=True):
        self.base_path = CONFIG_BASE_PATH  # Set in config.py
        self.output_dir = output_dir or os.path.join(
            self.base_path, '#_dl_experiments', 'results'
        )
        self.experiment_name = experiment_name
        self.verbose = verbose
        self.device = get_device()

        os.makedirs(self.output_dir, exist_ok=True)

    def train(self, model, data=None, epochs=500, batch_size=1024,
              learning_rate=0.001, use_ticker_embedding=False,
              label='', is_sequence=False,
              train_loader=None, test_loader=None,
              n_train=None, n_test=None):
        """Train a model and return full epoch-by-epoch history.

        Args:
            model: PyTorch nn.Module.
            data: Dict with X_train, y_train, X_test, y_test, tid_train, tid_test.
                  Can be None if train_loader/test_loader are provided.
            epochs: Number of epochs (NO early stopping).
            batch_size: Mini-batch size.
            learning_rate: Adam LR.
            use_ticker_embedding: Feed ticker IDs to model.
            label: Run label.
            is_sequence: True for CNN/LSTM (3-D input).
            train_loader: Pre-built DataLoader (overrides data-based creation).
            test_loader: Pre-built DataLoader (overrides data-based creation).
            n_train: Number of training samples (for logging when using loaders).
            n_test: Number of test samples (for logging when using loaders).

        Returns:
            Dict with 'history', 'model', 'config', 'elapsed'.
        """
        run_label = label or self.experiment_name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Move model to device
        model = model.to(self.device)
        total_params = sum(p.numel() for p in model.parameters())

        # Determine sample counts
        _n_train = n_train or (len(data['y_train']) if data else len(train_loader.dataset))
        _n_test = n_test or (len(data['y_test']) if data else len(test_loader.dataset))

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"TRAINING: {run_label}")
            print(f"{'='*70}")
            print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {learning_rate}")
            print(f"  Params: {total_params:,}")
            print(f"  Train samples: {_n_train:,}")
            print(f"  Test samples: {_n_test:,}")
            ratio = total_params / _n_train
            print(f"  Overparameterisation ratio: {ratio:.2f}x "
                  f"({'INTERPOLATING' if ratio > 1 else 'UNDERPARAMETERISED'})")
            print(f"  Device: {self.device}")
            print(f"{'='*70}")

        # Create DataLoaders (skip if pre-built loaders provided)
        if train_loader is None:
            DatasetClass = SequenceDataset if is_sequence else FlatDataset
            train_ds = DatasetClass(
                data['X_train'], data['y_train'],
                data['tid_train'] if use_ticker_embedding else None,
            )
            test_ds = DatasetClass(
                data['X_test'], data['y_test'],
                data['tid_test'] if use_ticker_embedding else None,
            )
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      num_workers=0, pin_memory=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                     num_workers=0, pin_memory=True)

        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        history = []
        t0 = time.time()

        for epoch in range(epochs):
            # ── Train ───────────────────────────────────────────────
            model.train()
            train_loss_sum = 0.0
            train_preds_all = []
            train_labels_all = []
            n_batches = 0

            for batch in train_loader:
                features = batch['features'].to(self.device)
                targets = batch['target'].to(self.device)
                tid = batch.get('ticker_id')
                if tid is not None:
                    tid = tid.to(self.device)

                optimizer.zero_grad()
                logits = model(features, ticker_id=tid)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                n_batches += 1

                with torch.no_grad():
                    probs = torch.sigmoid(logits).cpu().numpy()
                    train_preds_all.append(probs)
                    train_labels_all.append(targets.cpu().numpy())

            train_loss = train_loss_sum / n_batches
            train_preds = np.concatenate(train_preds_all)
            train_labels = np.concatenate(train_labels_all)

            try:
                train_auc = roc_auc_score(train_labels, train_preds)
            except ValueError:
                train_auc = 0.5

            train_acc = ((train_preds > 0.5) == train_labels).mean()

            # ── Evaluate ────────────────────────────────────────────
            model.eval()
            val_loss_sum = 0.0
            val_preds_all = []
            val_labels_all = []
            n_val_batches = 0

            with torch.no_grad():
                for batch in test_loader:
                    features = batch['features'].to(self.device)
                    targets = batch['target'].to(self.device)
                    tid = batch.get('ticker_id')
                    if tid is not None:
                        tid = tid.to(self.device)

                    logits = model(features, ticker_id=tid)
                    loss = criterion(logits, targets)

                    val_loss_sum += loss.item()
                    n_val_batches += 1

                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds_all.append(probs)
                    val_labels_all.append(targets.cpu().numpy())

            val_loss = val_loss_sum / n_val_batches
            val_preds = np.concatenate(val_preds_all)
            val_labels = np.concatenate(val_labels_all)

            try:
                val_auc = roc_auc_score(val_labels, val_preds)
            except ValueError:
                val_auc = 0.5

            val_acc = ((val_preds > 0.5) == val_labels).mean()

            # Record
            record = {
                'epoch': epoch + 1,
                'loss': train_loss, 'val_loss': val_loss,
                'auc': train_auc, 'val_auc': val_auc,
                'accuracy': train_acc, 'val_accuracy': val_acc,
            }
            history.append(record)

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:4d} | "
                      f"loss {train_loss:.4f}/{val_loss:.4f} | "
                      f"AUC {train_auc:.4f}/{val_auc:.4f} | "
                      f"Acc {train_acc:.4f}/{val_acc:.4f}")

            # Periodic checkpoint — save history every 25 epochs so a
            # crash/restart loses at most 25 epochs of one model.
            if (epoch + 1) % 25 == 0 and (epoch + 1) < epochs:
                ckpt_path = os.path.join(
                    self.output_dir,
                    f'{run_label}_{timestamp}_checkpoint_ep{epoch+1}.csv'
                )
                pd.DataFrame(history).to_csv(ckpt_path, index=False)
                # Also save model weights so training can resume
                ckpt_model_path = os.path.join(
                    self.output_dir,
                    f'{run_label}_{timestamp}_checkpoint_ep{epoch+1}_model.pt'
                )
                try:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, ckpt_model_path)
                except Exception:
                    pass
                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"    [checkpoint saved @ epoch {epoch+1}]")

        elapsed = time.time() - t0

        if self.verbose:
            print(f"\n  Training complete in {elapsed:.1f}s ({elapsed/epochs:.2f}s/epoch)")
            last = history[-1]
            print(f"  Final -> loss: {last['loss']:.4f}/{last['val_loss']:.4f}  "
                  f"AUC: {last['auc']:.4f}/{last['val_auc']:.4f}  "
                  f"Acc: {last['accuracy']:.4f}/{last['val_accuracy']:.4f}")

        # Config
        config = {
            'label': run_label, 'timestamp': timestamp,
            'epochs': epochs, 'batch_size': batch_size,
            'learning_rate': learning_rate,
            'total_params': total_params,
            'train_samples': _n_train,
            'test_samples': _n_test,
            'elapsed_seconds': elapsed,
            'device': str(self.device),
        }

        result = {'history': history, 'model': model, 'config': config, 'elapsed': elapsed}
        self._save_results(result, run_label, timestamp)
        return result

    def _save_results(self, result, label, timestamp):
        safe_label = label.replace(' ', '_').replace('/', '_')

        # History CSV
        hist_path = os.path.join(self.output_dir, f'{safe_label}_{timestamp}_history.csv')
        pd.DataFrame(result['history']).to_csv(hist_path, index=False)

        # Config JSON
        config_path = os.path.join(self.output_dir, f'{safe_label}_{timestamp}_config.json')
        with open(config_path, 'w') as f:
            json.dump(result['config'], f, indent=2)

        # Model weights
        model_path = os.path.join(self.output_dir, f'{safe_label}_{timestamp}_model.pt')
        try:
            torch.save(result['model'].state_dict(), model_path)
        except Exception:
            pass

        if self.verbose:
            print(f"  Saved: {os.path.basename(hist_path)}")
            print(f"  Saved: {os.path.basename(config_path)}")

    def run_model_size_sweep(self, arch, data, widths=(32, 64, 128, 256, 512, 1024, 2048),
                             depth=3, epochs=300, batch_size=1024, learning_rate=0.001,
                             seq_len=None, is_sequence=False):
        """Sweep model width at fixed depth to visualise model-wise double descent."""
        results = []
        n_features = data['X_train'].shape[-1]

        print(f"\n{'#'*70}")
        print(f"MODEL-SIZE SWEEP: {arch.upper()} -- depth={depth}, "
              f"widths={list(widths)}, epochs={epochs}")
        print(f"{'#'*70}")

        for width in widths:
            hidden = [width] * depth
            label = f'{arch}_w{width}_d{depth}'

            if arch == 'mlp':
                model = build_mlp(n_features=n_features, hidden_layers=hidden)
            elif arch == 'cnn':
                sl = seq_len or data['X_train'].shape[1]
                model = build_cnn(n_features=n_features, seq_len=sl,
                                  filters=hidden, dense_units=(width,))
            elif arch == 'lstm':
                model = build_lstm(n_features=n_features, lstm_units=hidden,
                                   dense_units=(width,))
            else:
                raise ValueError(f"Unknown arch: {arch}")

            result = self.train(
                model=model, data=data, epochs=epochs,
                batch_size=batch_size, learning_rate=learning_rate,
                label=label, is_sequence=is_sequence,
            )
            results.append(result)
            del model
            torch.cuda.empty_cache()

        # Save sweep summary
        summary = []
        for r in results:
            last = r['history'][-1]
            summary.append({
                'label': r['config']['label'],
                'params': r['config']['total_params'],
                'final_train_loss': last['loss'],
                'final_val_loss': last['val_loss'],
                'final_train_auc': last['auc'],
                'final_val_auc': last['val_auc'],
                'best_val_auc': max(h['val_auc'] for h in r['history']),
                'best_val_loss': min(h['val_loss'] for h in r['history']),
                'elapsed': r['elapsed'],
            })

        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(
            self.output_dir,
            f'sweep_{arch}_d{depth}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        summary_df.to_csv(summary_path, index=False)

        if self.verbose:
            print(f"\n{'='*70}")
            print("SWEEP SUMMARY")
            print(f"{'='*70}")
            print(summary_df.to_string(index=False))
            print(f"\nSaved: {os.path.basename(summary_path)}")

        return results


# ════════════════════════════════════════════════════════════════════════
# PLOT UTILITIES
# ════════════════════════════════════════════════════════════════════════

def plot_loss_curves(results, output_dir=None, title=''):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available -- skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for r in results:
        label = r['config']['label']
        epochs = [h['epoch'] for h in r['history']]
        axes[0].plot(epochs, [h['loss'] for h in r['history']], '--', alpha=0.4)
        axes[0].plot(epochs, [h['val_loss'] for h in r['history']], label=label)
        axes[1].plot(epochs, [h['auc'] for h in r['history']], '--', alpha=0.4)
        axes[1].plot(epochs, [h['val_auc'] for h in r['history']], label=label)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (BCE)')
    axes[0].set_title(f'{title} -- Loss Curves')
    axes[0].legend(fontsize=7)
    axes[0].set_yscale('log')

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].set_title(f'{title} -- AUC Curves')
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    if output_dir:
        path = os.path.join(output_dir, f'{title.replace(" ", "_")}_curves.png')
        plt.savefig(path, dpi=150)
        print(f"  Plot saved: {os.path.basename(path)}")
    plt.close()


def plot_model_wise_dd(results, output_dir=None, title=''):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    params = [r['config']['total_params'] for r in results]
    best_val_loss = [min(h['val_loss'] for h in r['history']) for r in results]
    final_val_loss = [r['history'][-1]['val_loss'] for r in results]
    labels = [r['config']['label'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(params, best_val_loss, 'b-o', label='Best val loss', markersize=8)
    ax.plot(params, final_val_loss, 'r--s', label='Final val loss', markersize=6)

    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (params[i], best_val_loss[i]), fontsize=7,
                    textcoords='offset points', xytext=(5, 5))

    ax.set_xlabel('Model Parameters')
    ax.set_ylabel('Validation Loss')
    ax.set_title(f'{title} -- Model-Wise Double Descent')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        path = os.path.join(output_dir, f'{title.replace(" ", "_")}_model_dd.png')
        plt.savefig(path, dpi=150)
        print(f"  Plot saved: {os.path.basename(path)}")
    plt.close()
