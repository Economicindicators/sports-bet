"""スポーツ別モデル管理"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from config.settings import MODELS_DIR

logger = logging.getLogger(__name__)


def list_models() -> list[dict]:
    """保存済みモデルのリストを返す"""
    models = []
    for lgb_file in MODELS_DIR.glob("*.lgb"):
        meta_file = lgb_file.with_suffix(".json")
        info = {"path": str(lgb_file), "name": lgb_file.stem}
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            info["n_features"] = len(meta.get("feature_names", []))
        models.append(info)
    return models


def get_latest_model(sport_code: str) -> Optional[Path]:
    """指定スポーツの最新モデルパスを返す"""
    candidates = sorted(
        MODELS_DIR.glob(f"{sport_code}_*.lgb"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None
