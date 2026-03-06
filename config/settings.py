"""パス、DB URL、環境変数"""

from pathlib import Path

# ========== パス ==========

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "sports_bet.db"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = DATA_DIR / "models"
SPORTS_YAML = PROJECT_ROOT / "config" / "sports.yaml"

# ========== DB ==========

DATABASE_URL = f"sqlite:///{DB_PATH}"

# ========== ディレクトリ初期化 ==========


def ensure_dirs():
    """必要なディレクトリを作成"""
    for d in [DATA_DIR, CACHE_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
