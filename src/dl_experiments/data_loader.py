"""
Deep Learning Data Loader — Universal Model (PyTorch)
======================================================

Loads ml_features_master.csv (336 columns, ~1.3M rows) and prepares data
for deep learning double-descent experiments.

Key design decisions:
  - UNIVERSAL model (all tickers pooled) — NOT per-ticker
  - Temporal train/test split (no data leakage)
  - Ticker ID encoded as integer for optional embedding layer
  - Returns both flat (MLP) and sequenced (CNN/LSTM) formats
  - Normalization fit on training data only
  - Returns PyTorch tensors, GPU-ready

Created: March 2026
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler, LabelEncoder

import torch
from torch.utils.data import Dataset, DataLoader


# ════════════════════════════════════════════════════════════════════════
# PyTorch Datasets
# ════════════════════════════════════════════════════════════════════════

class FlatDataset(Dataset):
    """Flat tabular dataset for MLP models."""

    def __init__(self, X, y, ticker_ids=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.ticker_ids = (
            torch.tensor(ticker_ids, dtype=torch.long) if ticker_ids is not None else None
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        item = {'features': self.X[idx], 'target': self.y[idx]}
        if self.ticker_ids is not None:
            item['ticker_id'] = self.ticker_ids[idx]
        return item


class SequenceDataset(Dataset):
    """Sequence dataset for CNN/LSTM models (pre-materialised)."""

    def __init__(self, X, y, ticker_ids=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.ticker_ids = (
            torch.tensor(ticker_ids, dtype=torch.long) if ticker_ids is not None else None
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        item = {'features': self.X[idx], 'target': self.y[idx]}
        if self.ticker_ids is not None:
            item['ticker_id'] = self.ticker_ids[idx]
        return item


class LazySequenceDataset(Dataset):
    """Memory-efficient sequence dataset — builds windows on-the-fly.

    Instead of pre-allocating a giant (N, seq_len, n_features) array,
    this stores per-symbol feature matrices and an index of valid
    (symbol_idx, row) pairs.  Each __getitem__ slices one window.

    RAM footprint ≈ base DataFrame (already in memory) + index list.
    """

    def __init__(self, symbol_arrays, symbol_targets, symbol_tids,
                 seq_len, max_nan_frac=0.10):
        """
        Args:
            symbol_arrays: list of np.ndarray (rows, n_features) per symbol
            symbol_targets: list of np.ndarray (rows,) per symbol
            symbol_tids: list of int (symbol id per symbol)
            seq_len: window length
            max_nan_frac: skip windows with more NaN than this
        """
        self.symbol_arrays = symbol_arrays
        self.symbol_targets = symbol_targets
        self.symbol_tids = symbol_tids
        self.seq_len = seq_len

        # Build index of valid (sym_list_idx, row_within_symbol) pairs
        self.index = []
        for s_idx in range(len(symbol_arrays)):
            feats = symbol_arrays[s_idx]
            labels = symbol_targets[s_idx]
            n_rows = len(labels)
            for i in range(seq_len, n_rows):
                if np.isnan(labels[i]):
                    continue
                window = feats[i - seq_len:i]
                nan_frac = np.isnan(window).mean()
                if nan_frac > max_nan_frac:
                    continue
                self.index.append((s_idx, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s_idx, row = self.index[idx]
        window = self.symbol_arrays[s_idx][row - self.seq_len:row].copy()
        np.nan_to_num(window, nan=0.0, copy=False)
        label = self.symbol_targets[s_idx][row]
        item = {
            'features': torch.tensor(window, dtype=torch.float32),
            'target': torch.tensor(label, dtype=torch.float32),
        }
        return item


# ════════════════════════════════════════════════════════════════════════
# Data Loader
# ════════════════════════════════════════════════════════════════════════

class DLDataLoader:
    """Universal data loader for deep learning experiments."""

    META_COLS = ['date', 'symbol']

    PRICE_LEVEL_COLS = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP',
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
        'BB_Upper', 'BB_Lower', 'OBV', 'Volume_SMA_20', 'ATR_14',
        'SPY_Close', 'SPY_Volume', 'QQQ_Close', 'QQQ_Volume',
        'IWM_Close', 'IWM_Volume', 'VTI_Close', 'VTI_Volume',
        'GLD_Close', 'GLD_Volume', 'TLT_Close', 'TLT_Volume',
        'VXX_Close', 'VXX_Volume', 'USO_Close', 'USO_Volume',
        'ES_Close', 'SPY_Close_fut', 'QQQ_Close_fut', 'TY_Close',
        'FV_Close', 'TLT_Close_fut', 'CL_Close', 'NG_Close',
        'GLD_Close_fut', 'DX_Close', 'VXX_Close_fut', 'HG_Close',
        'VXX_MA_20', 'atr_14d', 'atr_28d',
        'Market_Volume_Total', 'Transactions',
        'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'USD_CAD',
        'AUD_USD', 'NZD_USD', 'EUR_GBP', 'EUR_JPY', 'GBP_JPY',
        'EUR_USD_ma_60d', 'GBP_USD_ma_60d', 'USD_JPY_ma_60d',
        'USD_CHF_ma_60d', 'USD_CAD_ma_60d', 'AUD_USD_ma_60d',
        'NZD_USD_ma_60d', 'EUR_GBP_ma_60d', 'EUR_JPY_ma_60d',
        'GBP_JPY_ma_60d',
        'Treasury_3M', 'Treasury_6M', 'Treasury_1Y', 'Treasury_2Y',
        'Treasury_5Y', 'Treasury_10Y', 'Treasury_30Y', 'TIPS_10Y',
        'Breakeven_5Y', 'Breakeven_10Y',
    ]

    def __init__(
        self,
        csv_path: str = None,
        train_ratio: float = 0.80,
        target_days: int = 1,
        target_threshold: float = 0.02,
        sequence_length: int = 60,
        drop_price_levels: bool = True,
        min_rows_per_symbol: int = 120,
        verbose: bool = True,
    ):
        self.base_path = CONFIG_BASE_PATH  # Set in config.py
        self.csv_path = csv_path or os.path.join(
            self.base_path, r'#_feature_engineering\ml_features_master.csv'
        )
        self.train_ratio = train_ratio
        self.target_days = target_days
        self.target_threshold = target_threshold
        self.sequence_length = sequence_length
        self.drop_price_levels = drop_price_levels
        self.min_rows_per_symbol = min_rows_per_symbol
        self.verbose = verbose

        self.df = None
        self.feature_cols = None
        self.scaler = None
        self.label_enc = None
        self.split_date = None
        self.n_features = None
        self.n_symbols = None

    # ── PUBLIC API ──────────────────────────────────────────────────

    def load(self):
        """Load CSV, clean, create targets, normalise, split."""
        self._load_csv()
        self._encode_symbols()
        self._create_targets()
        self._select_features()
        self._temporal_split()
        self._normalise()
        self._print_summary()
        return self

    def get_flat_data(self):
        """Return numpy arrays for MLP training."""
        train = self.df[self.df['_split'] == 'train']
        test = self.df[self.df['_split'] == 'test']

        result = {
            'X_train': train[self.feature_cols].values.astype(np.float32),
            'y_train': train['_target'].values.astype(np.float32),
            'X_test': test[self.feature_cols].values.astype(np.float32),
            'y_test': test['_target'].values.astype(np.float32),
            'tid_train': train['_symbol_id'].values.astype(np.int64),
            'tid_test': test['_symbol_id'].values.astype(np.int64),
        }

        if self.verbose:
            print(f"\n  Flat data -> train {result['X_train'].shape}, "
                  f"test {result['X_test'].shape}")
            print(f"  Class balance -> train {result['y_train'].mean()*100:.1f}% pos, "
                  f"test {result['y_test'].mean()*100:.1f}% pos")
        return result

    def get_flat_dataloaders(self, batch_size=1024, num_workers=0,
                             use_ticker_ids=False):
        """Return PyTorch DataLoaders for flat data."""
        data = self.get_flat_data()
        train_ds = FlatDataset(
            data['X_train'], data['y_train'],
            data['tid_train'] if use_ticker_ids else None,
        )
        test_ds = FlatDataset(
            data['X_test'], data['y_test'],
            data['tid_test'] if use_ticker_ids else None,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
        return train_loader, test_loader, data

    def get_sequence_data(self, seq_len=None):
        """Return 3-D sequences (samples, timesteps, features) for CNN/LSTM."""
        seq_len = seq_len or self.sequence_length

        X_train_seq, y_train_seq, tid_train_seq = [], [], []
        X_test_seq, y_test_seq, tid_test_seq = [], [], []

        for sid, grp in self.df.groupby('_symbol_id'):
            grp = grp.sort_values('date')
            feats = grp[self.feature_cols].values.astype(np.float32)
            labels = grp['_target'].values.astype(np.float32)
            splits = grp['_split'].values

            for i in range(seq_len, len(grp)):
                window = feats[i - seq_len:i]
                label = labels[i]
                split = splits[i]

                if np.isnan(label):
                    continue
                nan_frac = np.isnan(window).mean()
                if nan_frac > 0.10:
                    continue
                window = np.nan_to_num(window, nan=0.0)

                if split == 'train':
                    X_train_seq.append(window)
                    y_train_seq.append(label)
                    tid_train_seq.append(sid)
                else:
                    X_test_seq.append(window)
                    y_test_seq.append(label)
                    tid_test_seq.append(sid)

        result = {
            'X_train': np.array(X_train_seq, dtype=np.float32),
            'y_train': np.array(y_train_seq, dtype=np.float32),
            'X_test': np.array(X_test_seq, dtype=np.float32),
            'y_test': np.array(y_test_seq, dtype=np.float32),
            'tid_train': np.array(tid_train_seq, dtype=np.int64),
            'tid_test': np.array(tid_test_seq, dtype=np.int64),
        }

        if self.verbose:
            print(f"\n  Sequence data (seq_len={seq_len}) ->")
            print(f"    train {result['X_train'].shape}, test {result['X_test'].shape}")
            print(f"    Class balance -> train {result['y_train'].mean()*100:.1f}%, "
                  f"test {result['y_test'].mean()*100:.1f}% pos")
        return result

    def get_sequence_dataloaders(self, batch_size=512, seq_len=None,
                                 num_workers=0, use_ticker_ids=False):
        """Return PyTorch DataLoaders for sequence data (pre-materialised)."""
        data = self.get_sequence_data(seq_len=seq_len)
        train_ds = SequenceDataset(
            data['X_train'], data['y_train'],
            data['tid_train'] if use_ticker_ids else None,
        )
        test_ds = SequenceDataset(
            data['X_test'], data['y_test'],
            data['tid_test'] if use_ticker_ids else None,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
        return train_loader, test_loader, data

    def get_lazy_sequence_dataloaders(self, batch_size=512, seq_len=None,
                                      num_workers=0):
        """Return memory-efficient DataLoaders for CNN/LSTM.

        Builds an index (~few MB) instead of materialising all windows
        (~40 GiB).  Each batch is assembled on-the-fly.
        """
        seq_len = seq_len or self.sequence_length

        # Separate per-symbol arrays by split
        train_arrays, train_targets, train_tids = [], [], []
        test_arrays, test_targets, test_tids = [], [], []

        for sid, grp in self.df.groupby('_symbol_id'):
            grp = grp.sort_values('date')
            feats = grp[self.feature_cols].values.astype(np.float32)
            labels = grp['_target'].values.astype(np.float32)
            splits = grp['_split'].values

            # Find the first test row index within this symbol
            test_start = np.where(splits == 'test')[0]
            if len(test_start) == 0:
                # All train
                train_arrays.append(feats)
                train_targets.append(labels)
                train_tids.append(sid)
                continue

            first_test = test_start[0]

            # Train: need seq_len rows before the window target,
            # so include rows up to first_test (the test portion
            # won't be indexed because of the split check below)
            # We keep the FULL symbol array for train windows whose
            # target row falls in [seq_len, first_test)
            if first_test > seq_len:
                train_arrays.append(feats[:first_test])
                train_targets.append(labels[:first_test])
                train_tids.append(sid)

            # Test: include seq_len lookback rows before first_test
            lookback_start = max(0, first_test - seq_len)
            test_arrays.append(feats[lookback_start:])
            test_targets.append(labels[lookback_start:])
            test_tids.append(sid)

            # Adjust: for the test dataset, valid indices start at
            # (first_test - lookback_start) which is min(seq_len, first_test)
            # The LazySequenceDataset will naturally handle this because
            # it only indexes rows >= seq_len within each sub-array

        if self.verbose:
            print(f"\n  Building lazy sequence index (seq_len={seq_len})...")

        train_ds = LazySequenceDataset(
            train_arrays, train_targets, train_tids, seq_len
        )
        test_ds = LazySequenceDataset(
            test_arrays, test_targets, test_tids, seq_len
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)

        n_train = len(train_ds)
        n_test = len(test_ds)

        if self.verbose:
            print(f"    train samples: {n_train:,} | test samples: {n_test:,}")
            # Compute class balance from index
            pos_train = sum(train_ds.symbol_targets[s][r]
                           for s, r in train_ds.index) / max(n_train, 1)
            pos_test = sum(test_ds.symbol_targets[s][r]
                          for s, r in test_ds.index) / max(n_test, 1)
            print(f"    Class balance -> train {pos_train*100:.1f}%, "
                  f"test {pos_test*100:.1f}% pos")
            ram_mb = sum(a.nbytes for a in train_arrays + test_arrays) / 1e6
            print(f"    RAM for base arrays: {ram_mb:.0f} MB "
                  f"(vs ~{n_train * seq_len * self.n_features * 4 / 1e9:.1f} GiB pre-materialised)")

        return train_loader, test_loader, {'n_train': n_train, 'n_test': n_test}

    # ── INTERNALS ───────────────────────────────────────────────────

    def _load_csv(self):
        if self.verbose:
            print(f"Loading {os.path.basename(self.csv_path)} ...")
        self.df = pd.read_csv(self.csv_path, parse_dates=['date'])
        self.df = self.df.sort_values(['symbol', 'date']).reset_index(drop=True)
        if self.verbose:
            print(f"  Loaded {len(self.df):,} rows x {len(self.df.columns)} cols")

    def _encode_symbols(self):
        self.label_enc = LabelEncoder()
        self.df['_symbol_id'] = self.label_enc.fit_transform(self.df['symbol'])
        self.n_symbols = self.df['_symbol_id'].nunique()
        counts = self.df.groupby('_symbol_id').size()
        keep = counts[counts >= self.min_rows_per_symbol].index
        before = len(self.df)
        self.df = self.df[self.df['_symbol_id'].isin(keep)].reset_index(drop=True)
        dropped = before - len(self.df)
        if self.verbose and dropped > 0:
            print(f"  Dropped {dropped:,} rows from symbols with < {self.min_rows_per_symbol} rows")

    def _create_targets(self):
        self.df['_forward_return'] = (
            self.df.groupby('symbol')['Daily_Return']
            .transform(lambda x: x.shift(-self.target_days))
        )
        self.df['_target'] = (
            self.df['_forward_return'] > self.target_threshold
        ).astype(np.float32)
        invalid = self.df['_forward_return'].isna()
        self.df.loc[invalid, '_target'] = np.nan
        if self.verbose:
            valid = self.df['_target'].notna()
            pos = self.df.loc[valid, '_target'].mean() * 100
            print(f"  Target: outperform_{self.target_days}d > {self.target_threshold*100:.1f}%")
            print(f"  Valid target rows: {valid.sum():,} ({pos:.1f}% positive)")

    def _select_features(self):
        exclude = set(self.META_COLS + ['_symbol_id', '_forward_return', '_target', '_split'])
        if self.drop_price_levels:
            exclude.update(self.PRICE_LEVEL_COLS)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = [c for c in numeric_cols if c not in exclude]
        nan_fracs = self.df[self.feature_cols].isna().mean()
        high_nan = nan_fracs[nan_fracs > 0.50].index.tolist()
        if high_nan:
            self.feature_cols = [c for c in self.feature_cols if c not in high_nan]
            if self.verbose:
                print(f"  Dropped {len(high_nan)} features with >50% NaN")
        self.n_features = len(self.feature_cols)
        if self.verbose:
            print(f"  Selected {self.n_features} numeric features"
                  f" (dropped {len(self.PRICE_LEVEL_COLS) if self.drop_price_levels else 0}"
                  f" price-level cols)")

    def _temporal_split(self):
        unique_dates = sorted(self.df['date'].unique())
        split_idx = int(len(unique_dates) * self.train_ratio)
        self.split_date = unique_dates[split_idx]
        self.df['_split'] = np.where(self.df['date'] < self.split_date, 'train', 'test')
        self.df = self.df[self.df['_target'].notna()].reset_index(drop=True)
        n_train = (self.df['_split'] == 'train').sum()
        n_test = (self.df['_split'] == 'test').sum()
        if self.verbose:
            print(f"  Split date: {pd.Timestamp(self.split_date).date()}")
            print(f"  Train: {n_train:,} rows | Test: {n_test:,} rows")

    def _normalise(self):
        self.scaler = RobustScaler()
        train_mask = self.df['_split'] == 'train'
        self.scaler.fit(self.df.loc[train_mask, self.feature_cols].fillna(0.0))
        self.df[self.feature_cols] = self.scaler.transform(
            self.df[self.feature_cols].fillna(0.0)
        )
        if self.verbose:
            print(f"  Normalised with RobustScaler (fit on train only)")

    def _print_summary(self):
        if not self.verbose:
            return
        print("\n" + "=" * 70)
        print("DATA LOADER SUMMARY")
        print("=" * 70)
        print(f"  CSV:            {os.path.basename(self.csv_path)}")
        print(f"  Total rows:     {len(self.df):,}")
        print(f"  Symbols:        {self.df['_symbol_id'].nunique()}")
        print(f"  Features:       {self.n_features}")
        print(f"  Target:         outperform_{self.target_days}d > {self.target_threshold*100:.0f}%")
        print(f"  Split date:     {pd.Timestamp(self.split_date).date()}")
        train_n = (self.df['_split'] == 'train').sum()
        test_n = (self.df['_split'] == 'test').sum()
        print(f"  Train rows:     {train_n:,}")
        print(f"  Test rows:      {test_n:,}")
        pos = self.df['_target'].mean() * 100
        print(f"  Overall % pos:  {pos:.1f}%")
        if torch.cuda.is_available():
            print(f"  Compute:        CUDA ({torch.cuda.get_device_name(0)})")
        else:
            print(f"  Compute:        CPU")
        print("=" * 70)


# ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    loader = DLDataLoader(verbose=True)
    loader.load()
    print("\n--- Flat data test ---")
    flat = loader.get_flat_data()
    print(f"  X_train: {flat['X_train'].shape}")
