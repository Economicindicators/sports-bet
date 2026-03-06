"""日付範囲・シーズン計算ユーティリティ"""

from datetime import date, timedelta
from typing import Iterator


def date_range(start: date, end: date) -> Iterator[date]:
    """start から end まで (inclusive) の日付を生成"""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def parse_date(date_str: str) -> date:
    """YYYYMMDD 形式の文字列を date に変換"""
    return date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))


def format_date(d: date) -> str:
    """date を YYYYMMDD 形式に変換"""
    return d.strftime("%Y%m%d")


# ========== シーズン期間 ==========

SEASON_RANGES = {
    "npb": (3, 10),  # 3月〜10月
    "mlb": (3, 10),
    "jleague": (2, 12),  # 2月〜12月
    "premier": (8, 5),  # 8月〜翌5月 (跨ぎ)
    "laliga": (8, 5),
    "seriea": (8, 5),
    "bundesliga": (8, 5),
    "nba": (10, 6),  # 10月〜翌6月
    "bleague": (10, 5),
}


def get_season_dates(league_code: str, year: int) -> tuple[date, date]:
    """
    指定リーグ・年のシーズン開始日と終了日を返す。

    Args:
        league_code: リーグコード
        year: シーズン開始年

    Returns:
        (start_date, end_date)
    """
    start_month, end_month = SEASON_RANGES.get(league_code, (1, 12))

    if start_month <= end_month:
        # 同年完結 (NPB, Jリーグ等)
        start = date(year, start_month, 1)
        # 終了月の末日
        if end_month == 12:
            end = date(year, 12, 31)
        else:
            end = date(year, end_month + 1, 1) - timedelta(days=1)
    else:
        # 年跨ぎ (プレミア, NBA等)
        start = date(year, start_month, 1)
        end = date(year + 1, end_month + 1, 1) - timedelta(days=1)

    return start, end


def get_backfill_dates(league_code: str, years_back: int = 2) -> tuple[date, date]:
    """バックフィル用の日付範囲を返す (今日から years_back 年前まで)"""
    today = date.today()
    start = date(today.year - years_back, today.month, today.day)
    return start, today
