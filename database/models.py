"""SQLAlchemy ORM models (8テーブル)"""

from datetime import date, time, datetime

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Date,
    Time,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, Session, sessionmaker

from config.settings import DATABASE_URL

Base = declarative_base()


class Sport(Base):
    """スポーツマスタ"""

    __tablename__ = "sports"

    code = Column(String, primary_key=True)  # baseball, soccer, basketball
    name = Column(String, nullable=False)
    scoring_type = Column(String, nullable=False)  # runs, goals, points

    leagues = relationship("League", back_populates="sport")


class League(Base):
    """リーグマスタ"""

    __tablename__ = "leagues"

    league_code = Column(String, primary_key=True)  # npb, mlb, jleague, etc.
    sport_code = Column(String, ForeignKey("sports.code"), nullable=False)
    name = Column(String, nullable=False)
    source_site = Column(String, nullable=False)
    source_section = Column(String, nullable=False)

    sport = relationship("Sport", back_populates="leagues")
    teams = relationship("Team", back_populates="league")


class Team(Base):
    """チームマスタ"""

    __tablename__ = "teams"

    team_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    league_code = Column(String, ForeignKey("leagues.league_code"), nullable=False)
    venue = Column(String, nullable=True)

    league = relationship("League", back_populates="teams")

    __table_args__ = (
        UniqueConstraint("name", "league_code", name="uq_team_name_league"),
    )


class Player(Base):
    """選手マスタ"""

    __tablename__ = "players"

    player_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=True)
    league_code = Column(String, ForeignKey("leagues.league_code"), nullable=False)
    position = Column(String, nullable=True)  # pitcher, fielder, forward, guard, etc.

    team = relationship("Team")
    stats = relationship("PlayerStats", back_populates="player")

    __table_args__ = (
        UniqueConstraint("name", "league_code", name="uq_player_name_league"),
    )


class PlayerStats(Base):
    """選手統計 (試合データから算出)"""

    __tablename__ = "player_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.player_id"), nullable=False)
    season = Column(String, nullable=False)  # e.g. "2025", "2025-26"
    stat_type = Column(String, nullable=False)  # win_rate, era, starts, recent_form
    value = Column(Float, nullable=False)
    sample_size = Column(Integer, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow)

    player = relationship("Player", back_populates="stats")

    __table_args__ = (
        UniqueConstraint("player_id", "season", "stat_type", name="uq_player_stat"),
    )


class Match(Base):
    """試合データ"""

    __tablename__ = "matches"

    match_id = Column(Integer, primary_key=True, autoincrement=True)
    sport_code = Column(String, ForeignKey("sports.code"), nullable=False)
    league_code = Column(String, ForeignKey("leagues.league_code"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    time = Column(Time, nullable=True)
    venue = Column(String, nullable=True)
    home_team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    status = Column(String, default="scheduled")  # scheduled, finished, cancelled
    # 野球: 予告先発
    home_pitcher_id = Column(Integer, ForeignKey("players.player_id"), nullable=True)
    away_pitcher_id = Column(Integer, ForeignKey("players.player_id"), nullable=True)

    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])
    home_pitcher = relationship("Player", foreign_keys=[home_pitcher_id])
    away_pitcher = relationship("Player", foreign_keys=[away_pitcher_id])
    handicap = relationship("HandicapData", back_populates="match", uselist=False)
    prediction = relationship("Prediction", back_populates="match", uselist=False)
    bookmaker_odds = relationship("BookmakerOdds", back_populates="match")

    __table_args__ = (
        UniqueConstraint(
            "sport_code",
            "date",
            "home_team_id",
            "away_team_id",
            name="uq_match",
        ),
    )


class HandicapData(Base):
    """ハンデデータ (各試合1レコード)"""

    __tablename__ = "handicap_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(
        Integer, ForeignKey("matches.match_id"), nullable=False, unique=True
    )
    handicap_team_id = Column(
        Integer, ForeignKey("teams.team_id"), nullable=False
    )
    handicap_value = Column(Float, nullable=False)
    handicap_display = Column(String, nullable=True)  # 原文表記 (例: "0半5", "0/3")
    result_type = Column(String, nullable=True)
    payout_rate = Column(Float, nullable=True)

    match = relationship("Match", back_populates="handicap")
    handicap_team = relationship("Team", foreign_keys=[handicap_team_id])


class HandicapSnapshot(Base):
    """ハンデ値スナップショット (ライン移動追跡用)"""

    __tablename__ = "handicap_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.match_id"), nullable=False)
    handicap_team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    handicap_value = Column(Float, nullable=False)
    handicap_display = Column(String, nullable=True)
    snapshot_type = Column(String, nullable=False)  # opening, midday, closing
    captured_at = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match")
    handicap_team = relationship("Team", foreign_keys=[handicap_team_id])

    __table_args__ = (
        UniqueConstraint("match_id", "snapshot_type", name="uq_snapshot"),
    )


class BookmakerOdds(Base):
    """ブックメーカーオッズ (試合×ブックメーカー)"""

    __tablename__ = "bookmaker_odds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.match_id"), nullable=False)
    bookmaker = Column(String, nullable=False)  # pinnacle, bet365, etc.
    home_odds = Column(Float, nullable=True)  # h2h home
    away_odds = Column(Float, nullable=True)  # h2h away
    draw_odds = Column(Float, nullable=True)  # h2h draw (soccer)
    home_spread = Column(Float, nullable=True)  # spread value
    home_spread_odds = Column(Float, nullable=True)
    away_spread_odds = Column(Float, nullable=True)
    over_under = Column(Float, nullable=True)  # total line
    over_odds = Column(Float, nullable=True)
    under_odds = Column(Float, nullable=True)
    source = Column(String, nullable=False, default="oddsportal")  # oddsportal, odds_api
    captured_at = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match", back_populates="bookmaker_odds")

    __table_args__ = (
        UniqueConstraint("match_id", "bookmaker", "source", name="uq_bm_odds"),
    )


class Prediction(Base):
    """予測結果"""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.match_id"), nullable=False)
    model_version = Column(String, nullable=False)
    prob_home = Column(Float, nullable=True)
    prob_away = Column(Float, nullable=True)
    prob_draw = Column(Float, nullable=True)
    handicap_ev = Column(Float, nullable=True)
    recommended_bet = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match", back_populates="prediction")

    __table_args__ = (
        UniqueConstraint("match_id", "model_version", name="uq_prediction"),
    )


# ========== Engine & Session ==========


def get_engine():
    from config.settings import ensure_dirs

    ensure_dirs()
    return create_engine(DATABASE_URL, echo=False)


def get_session() -> Session:
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def init_db():
    """全テーブルを作成"""
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine
