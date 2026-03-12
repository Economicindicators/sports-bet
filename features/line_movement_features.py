"""ライン移動特徴量 — opening vs closing のハンデ値変動を捕捉"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from database.models import HandicapSnapshot, get_session

logger = logging.getLogger(__name__)


def add_line_movement_features(df: pd.DataFrame, session: Session | None = None) -> pd.DataFrame:
    """ライン移動の特徴量を追加。スナップショットが無い試合は0埋め。"""
    if df.empty:
        return df

    sess = session or get_session()
    own_session = session is None

    # 全スナップショットを一括取得
    match_ids = df["match_id"].tolist()
    snapshots = (
        sess.query(HandicapSnapshot)
        .filter(HandicapSnapshot.match_id.in_(match_ids))
        .all()
    )

    # match_id -> {snapshot_type: handicap_value}
    snap_map: dict[int, dict[str, float]] = {}
    for s in snapshots:
        snap_map.setdefault(s.match_id, {})[s.snapshot_type] = s.handicap_value

    # 特徴量を計算
    line_move = []        # opening → closing の変動量
    line_move_abs = []    # 変動の絶対値 (方向無関係の激しさ)
    has_movement = []     # ラインが動いたか (binary)
    line_move_mid = []    # opening → midday の変動量

    for _, row in df.iterrows():
        mid = int(row["match_id"])
        snaps = snap_map.get(mid, {})

        opening = snaps.get("opening")
        midday = snaps.get("midday")
        closing = snaps.get("closing")

        if opening is not None and closing is not None:
            move = closing - opening
            line_move.append(move)
            line_move_abs.append(abs(move))
            has_movement.append(1.0 if abs(move) > 0.01 else 0.0)
        else:
            line_move.append(0.0)
            line_move_abs.append(0.0)
            has_movement.append(0.0)

        if opening is not None and midday is not None:
            line_move_mid.append(midday - opening)
        else:
            line_move_mid.append(0.0)

    df["line_movement"] = line_move
    df["line_movement_abs"] = line_move_abs
    df["has_line_movement"] = has_movement
    df["line_movement_midday"] = line_move_mid

    if own_session:
        sess.close()

    logger.info(f"Added line movement features ({sum(has_movement)}/{len(df)} matches had movement)")
    return df
