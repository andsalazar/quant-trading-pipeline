# ============================================================================
# Configuration — Copy to config.py and fill in your values
# ============================================================================

import os

# ---------- Paths ----------
# Root directory for all project data and outputs
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")          # Raw & processed data
FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")   # Feature CSVs
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")       # Trained model artifacts
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")            # Pipeline logs

# ---------- Database ----------
DATABASE_PATH = os.path.join(DATA_DIR, "quant_trading_v2.db")

# ---------- API Keys (use environment variables) ----------
# export POLYGON_API_KEY="your_key_here"
# export OPENAI_API_KEY="your_key_here"  (optional, for sentiment)
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---------- Trading ----------
INITIAL_CAPITAL = 2000.0
BROKER = "IBKR"
IBKR_PORT = 7497          # 7497=paper, 7496=live
IBKR_CLIENT_ID = 1

# ---------- Model Defaults ----------
WALK_FORWARD_FOLDS = 5
RANDOM_STATE = 42
GPU_ENABLED = True         # Set False if no CUDA GPU

# ---------- Ticker Universe ----------
# The actual ticker list is loaded from your data source.
# Example: ~500 US equities from S&P 500 + mid-cap coverage
UNIVERSE_SIZE = 500
