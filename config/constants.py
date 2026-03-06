"""ハンデペイアウト表、スクレイピング設定、ML定数"""

# ========== ハンデ判定 ==========

# adjusted_margin の範囲 → (result_label, payout_multiplier)
# adjusted_margin = (有利チーム得点 - 不利チーム得点) - handicap_value
HANDICAP_OUTCOMES = [
    # (lower_bound, upper_bound, label, payout)
    # lower_bound inclusive, upper_bound exclusive (except edges)
    (2.0, float("inf"), "丸勝ち", 2.0),
    (1.0, 2.0, "7分勝ち", 1.7),
    (0.5, 1.0, "5分勝ち", 1.5),
    (0.0 + 1e-9, 0.5, "3分勝ち", 1.3),  # adj > 0
    (0.0, 0.0 + 1e-9, "勝負無し", 1.0),  # adj == 0 (返金)
    (-0.5, 0.0, "3分負け", 0.7),  # -0.5 < adj < 0
    (-1.0, -0.5, "5分負け", 0.5),
    (-2.0, -1.0, "7分負け", 0.3),
    (float("-inf"), -2.0, "丸負け", 0.0),
]

# 有利判定の閾値 (payout >= この値なら favorable)
FAVORABLE_PAYOUT_THRESHOLD = 1.5  # 5分勝ち以上

# ========== スクレイピング ==========

SCRAPE_DELAY_MIN = 3  # seconds
SCRAPE_DELAY_MAX = 5
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30  # seconds
CACHE_EXPIRY_HOURS = 24

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

# ========== データソース ==========

SOURCES = {
    "handenomori": {
        "base_url": "https://handenomori.com",
        "sports": ["npb", "mlb"],
    },
    "football_hande": {
        "base_url": "https://footbool-hande.com",
        "sports": ["jleague", "premier", "laliga", "seriea", "bundesliga"],
    },
    "b_handicap": {
        "base_url": "https://b-handicap.com",
        "sports": ["nba", "bleague"],
    },
}

# ========== ML ==========

LGBM_DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "max_depth": 5,
    "learning_rate": 0.01,
    "n_estimators": 3000,
    "min_child_samples": 20,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "verbosity": -1,
    "bagging_freq": 5,
}

CV_FOLDS = 5
CV_GAP_DAYS = 30  # 時系列CVのギャップ
EARLY_STOPPING_ROUNDS = 50

# ========== ベッティング ==========

KELLY_FRACTION = 0.25  # 1/4 Kelly
MAX_BET_FRACTION = 0.20  # 最大ベット比率
MIN_EV_THRESHOLD = 0.08  # 最低EV (バックテスト最適化済み)
STOP_LOSS_FRACTION = 0.30  # ストップロス
