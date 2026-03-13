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
        "sports": ["npb", "mlb", "wbc"],
    },
    "football_hande": {
        "base_url": "https://footbool-hande.com",
        "sports": ["jleague", "premier", "laliga", "seriea", "bundesliga", "ligue1", "eredivisie", "cl"],
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

# 野球専用パラメータ (Optuna最適化済み — ホーム勝敗ターゲット)
LGBM_BASEBALL_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 50,
    "max_depth": 3,
    "learning_rate": 0.018,
    "n_estimators": 3000,
    "min_child_samples": 38,
    "subsample": 0.61,
    "colsample_bytree": 0.78,
    "reg_alpha": 0.030,
    "reg_lambda": 1.69,
    "verbosity": -1,
    "bagging_freq": 1,
}

# サッカー専用パラメータ (Optuna最適化済み)
LGBM_SOCCER_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 39,
    "max_depth": 4,
    "learning_rate": 0.086,
    "n_estimators": 3000,
    "min_child_samples": 15,
    "subsample": 0.78,
    "colsample_bytree": 0.60,
    "reg_alpha": 0.012,
    "reg_lambda": 0.010,
    "verbosity": -1,
    "bagging_freq": 7,
}

# バスケ専用パラメータ (Optuna最適化済み)
LGBM_BASKETBALL_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 40,
    "max_depth": 4,
    "learning_rate": 0.079,
    "n_estimators": 3000,
    "min_child_samples": 28,
    "subsample": 0.83,
    "colsample_bytree": 0.70,
    "reg_alpha": 1.36,
    "reg_lambda": 0.028,
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

# リーグ別EV閾値 (バックテスト最適化済み)
# 未指定リーグは MIN_EV_THRESHOLD を使用
LEAGUE_EV_THRESHOLD = {
    "npb": 0.05,     # NPB: EV>=0.05でROI +0.8%
    "mlb": 0.12,     # MLB: 閾値高めで損失抑制
    "wbc": 0.05,     # WBC: NPBと同等
}

# ========== 逆張り (Contrarian) ==========

CONTRARIAN_MIN_EV_THRESHOLD = 0.10  # 逆張りの最低EV（通常EVが基準未満のとき逆側を検討）

# ========== リーグ / モデル設定 ==========

# 全リーグリスト (scrape対象)
ALL_LEAGUES = [
    "npb", "mlb", "wbc",
    "jleague", "premier", "laliga", "seriea", "bundesliga", "ligue1", "eredivisie", "cl",
    "nba", "bleague",
]

# リーグ → スポーツ マッピング
LEAGUE_TO_SPORT = {
    "npb": "baseball", "mlb": "baseball", "wbc": "baseball",
    "jleague": "soccer", "premier": "soccer",
    "laliga": "soccer", "seriea": "soccer",
    "bundesliga": "soccer", "ligue1": "soccer",
    "eredivisie": "soccer", "cl": "soccer",
    "nba": "basketball", "bleague": "basketball",
}

# スポーツ → リーグ マッピング
SPORT_TO_LEAGUES = {
    "baseball": [lg for lg, sp in LEAGUE_TO_SPORT.items() if sp == "baseball"],
    "soccer": [lg for lg, sp in LEAGUE_TO_SPORT.items() if sp == "soccer"],
    "basketball": [lg for lg, sp in LEAGUE_TO_SPORT.items() if sp == "basketball"],
}

# スポーツ → モデルバージョン
SPORT_MODEL_VERSION = {
    "baseball": "v4",
    "soccer": "v1",
    "basketball": "v1",
}
