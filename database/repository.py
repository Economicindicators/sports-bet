"""CRUD操作 (upsertパターン)"""

from datetime import date
from typing import Optional

import yaml
from sqlalchemy.orm import Session

from config.settings import SPORTS_YAML
from database.models import (
    Sport,
    League,
    Team,
    Player,
    PlayerStats,
    Match,
    HandicapData,
    HandicapSnapshot,
    BookmakerOdds,
    Prediction,
    get_session,
)


class Repository:
    """全モデル共通のCRUD操作"""

    def __init__(self, session: Optional[Session] = None):
        self.session = session or get_session()

    def close(self):
        self.session.close()

    def commit(self):
        self.session.commit()

    # ========== Sport ==========

    def upsert_sport(self, code: str, name: str, scoring_type: str) -> Sport:
        sport = self.session.query(Sport).filter_by(code=code).first()
        if sport:
            sport.name = name
            sport.scoring_type = scoring_type
        else:
            sport = Sport(code=code, name=name, scoring_type=scoring_type)
            self.session.add(sport)
        self.session.flush()
        return sport

    # ========== League ==========

    def upsert_league(
        self,
        league_code: str,
        sport_code: str,
        name: str,
        source_site: str,
        source_section: str,
    ) -> League:
        league = self.session.query(League).filter_by(league_code=league_code).first()
        if league:
            league.sport_code = sport_code
            league.name = name
            league.source_site = source_site
            league.source_section = source_section
        else:
            league = League(
                league_code=league_code,
                sport_code=sport_code,
                name=name,
                source_site=source_site,
                source_section=source_section,
            )
            self.session.add(league)
        self.session.flush()
        return league

    # ========== Team ==========

    def upsert_team(
        self, name: str, league_code: str, venue: Optional[str] = None
    ) -> Team:
        team = (
            self.session.query(Team)
            .filter_by(name=name, league_code=league_code)
            .first()
        )
        if team:
            if venue:
                team.venue = venue
        else:
            team = Team(name=name, league_code=league_code, venue=venue)
            self.session.add(team)
        self.session.flush()
        return team

    def get_team_by_name(self, name: str, league_code: str) -> Optional[Team]:
        return (
            self.session.query(Team)
            .filter_by(name=name, league_code=league_code)
            .first()
        )

    # ========== Player ==========

    def upsert_player(
        self,
        name: str,
        league_code: str,
        team_id: Optional[int] = None,
        position: Optional[str] = None,
    ) -> Player:
        player = (
            self.session.query(Player)
            .filter_by(name=name, league_code=league_code)
            .first()
        )
        if player:
            if team_id:
                player.team_id = team_id
            if position:
                player.position = position
        else:
            player = Player(
                name=name,
                league_code=league_code,
                team_id=team_id,
                position=position,
            )
            self.session.add(player)
        self.session.flush()
        return player

    # ========== Match ==========

    def upsert_match(
        self,
        sport_code: str,
        league_code: str,
        match_date: date,
        home_team_id: int,
        away_team_id: int,
        home_score: Optional[int] = None,
        away_score: Optional[int] = None,
        match_time=None,
        venue: Optional[str] = None,
        status: str = "scheduled",
        home_pitcher_id: Optional[int] = None,
        away_pitcher_id: Optional[int] = None,
    ) -> Match:
        match = (
            self.session.query(Match)
            .filter_by(
                sport_code=sport_code,
                date=match_date,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
            )
            .first()
        )
        if match:
            match.home_score = home_score if home_score is not None else match.home_score
            match.away_score = away_score if away_score is not None else match.away_score
            match.status = status
            if match_time:
                match.time = match_time
            if venue:
                match.venue = venue
            if home_pitcher_id:
                match.home_pitcher_id = home_pitcher_id
            if away_pitcher_id:
                match.away_pitcher_id = away_pitcher_id
        else:
            match = Match(
                sport_code=sport_code,
                league_code=league_code,
                date=match_date,
                time=match_time,
                venue=venue,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                home_score=home_score,
                away_score=away_score,
                status=status,
                home_pitcher_id=home_pitcher_id,
                away_pitcher_id=away_pitcher_id,
            )
            self.session.add(match)
        self.session.flush()
        return match

    def get_matches_by_date(
        self, match_date: date, league_code: Optional[str] = None
    ) -> list[Match]:
        q = self.session.query(Match).filter_by(date=match_date)
        if league_code:
            q = q.filter_by(league_code=league_code)
        return q.all()

    def get_matches_range(
        self,
        start_date: date,
        end_date: date,
        sport_code: Optional[str] = None,
        league_code: Optional[str] = None,
    ) -> list[Match]:
        q = self.session.query(Match).filter(
            Match.date >= start_date, Match.date <= end_date
        )
        if sport_code:
            q = q.filter_by(sport_code=sport_code)
        if league_code:
            q = q.filter_by(league_code=league_code)
        return q.order_by(Match.date).all()

    # ========== HandicapData ==========

    def upsert_handicap(
        self,
        match_id: int,
        handicap_team_id: int,
        handicap_value: float,
        result_type: Optional[str] = None,
        payout_rate: Optional[float] = None,
        handicap_display: Optional[str] = None,
    ) -> HandicapData:
        hd = (
            self.session.query(HandicapData).filter_by(match_id=match_id).first()
        )
        if hd:
            hd.handicap_team_id = handicap_team_id
            hd.handicap_value = handicap_value
            hd.result_type = result_type if result_type else hd.result_type
            hd.payout_rate = payout_rate if payout_rate else hd.payout_rate
            if handicap_display:
                hd.handicap_display = handicap_display
        else:
            hd = HandicapData(
                match_id=match_id,
                handicap_team_id=handicap_team_id,
                handicap_value=handicap_value,
                handicap_display=handicap_display,
                result_type=result_type,
                payout_rate=payout_rate,
            )
            self.session.add(hd)
        self.session.flush()
        return hd

    # ========== Prediction ==========

    def upsert_prediction(
        self,
        match_id: int,
        model_version: str,
        prob_home: Optional[float] = None,
        prob_away: Optional[float] = None,
        prob_draw: Optional[float] = None,
        handicap_ev: Optional[float] = None,
        recommended_bet: Optional[str] = None,
    ) -> Prediction:
        pred = (
            self.session.query(Prediction)
            .filter_by(match_id=match_id, model_version=model_version)
            .first()
        )
        if pred:
            pred.prob_home = prob_home
            pred.prob_away = prob_away
            pred.prob_draw = prob_draw
            pred.handicap_ev = handicap_ev
            pred.recommended_bet = recommended_bet
        else:
            pred = Prediction(
                match_id=match_id,
                model_version=model_version,
                prob_home=prob_home,
                prob_away=prob_away,
                prob_draw=prob_draw,
                handicap_ev=handicap_ev,
                recommended_bet=recommended_bet,
            )
            self.session.add(pred)
        self.session.flush()
        return pred

    # ========== PlayerStats ==========

    def upsert_player_stat(
        self,
        player_id: int,
        season: str,
        stat_type: str,
        value: float,
        sample_size: int,
    ) -> PlayerStats:
        from datetime import datetime

        ps = (
            self.session.query(PlayerStats)
            .filter_by(player_id=player_id, season=season, stat_type=stat_type)
            .first()
        )
        if ps:
            ps.value = value
            ps.sample_size = sample_size
            ps.updated_at = datetime.utcnow()
        else:
            ps = PlayerStats(
                player_id=player_id,
                season=season,
                stat_type=stat_type,
                value=value,
                sample_size=sample_size,
            )
            self.session.add(ps)
        self.session.flush()
        return ps

    def get_player_stats(
        self, player_id: int, season: Optional[str] = None
    ) -> list[PlayerStats]:
        q = self.session.query(PlayerStats).filter_by(player_id=player_id)
        if season:
            q = q.filter_by(season=season)
        return q.all()

    def get_team_players_with_stats(
        self, team_id: int, season: Optional[str] = None
    ) -> list[dict]:
        """チームの選手一覧 + 統計"""
        players = self.session.query(Player).filter_by(team_id=team_id).all()
        result = []
        for p in players:
            stats = self.get_player_stats(p.player_id, season)
            result.append({
                "player_id": p.player_id,
                "name": p.name,
                "position": p.position,
                "stats": {s.stat_type: {"value": s.value, "sample_size": s.sample_size} for s in stats},
            })
        return result

    # ========== HandicapSnapshot ==========

    def upsert_snapshot(
        self,
        match_id: int,
        handicap_team_id: int,
        handicap_value: float,
        snapshot_type: str,
        handicap_display: Optional[str] = None,
    ) -> HandicapSnapshot:
        from datetime import datetime

        snap = (
            self.session.query(HandicapSnapshot)
            .filter_by(match_id=match_id, snapshot_type=snapshot_type)
            .first()
        )
        if snap:
            snap.handicap_team_id = handicap_team_id
            snap.handicap_value = handicap_value
            snap.handicap_display = handicap_display
            snap.captured_at = datetime.utcnow()
        else:
            snap = HandicapSnapshot(
                match_id=match_id,
                handicap_team_id=handicap_team_id,
                handicap_value=handicap_value,
                handicap_display=handicap_display,
                snapshot_type=snapshot_type,
            )
            self.session.add(snap)
        self.session.flush()
        return snap

    def get_snapshots(self, match_id: int) -> list[HandicapSnapshot]:
        return (
            self.session.query(HandicapSnapshot)
            .filter_by(match_id=match_id)
            .order_by(HandicapSnapshot.captured_at)
            .all()
        )

    # ========== BookmakerOdds ==========

    def upsert_bookmaker_odds(
        self,
        match_id: int,
        bookmaker: str,
        source: str = "oddsportal",
        home_odds: float = None,
        away_odds: float = None,
        draw_odds: float = None,
        home_spread: float = None,
        home_spread_odds: float = None,
        away_spread_odds: float = None,
        over_under: float = None,
        over_odds: float = None,
        under_odds: float = None,
    ) -> BookmakerOdds:
        from datetime import datetime

        bo = (
            self.session.query(BookmakerOdds)
            .filter_by(match_id=match_id, bookmaker=bookmaker, source=source)
            .first()
        )
        if bo:
            if home_odds is not None: bo.home_odds = home_odds
            if away_odds is not None: bo.away_odds = away_odds
            if draw_odds is not None: bo.draw_odds = draw_odds
            if home_spread is not None: bo.home_spread = home_spread
            if home_spread_odds is not None: bo.home_spread_odds = home_spread_odds
            if away_spread_odds is not None: bo.away_spread_odds = away_spread_odds
            if over_under is not None: bo.over_under = over_under
            if over_odds is not None: bo.over_odds = over_odds
            if under_odds is not None: bo.under_odds = under_odds
            bo.captured_at = datetime.utcnow()
        else:
            bo = BookmakerOdds(
                match_id=match_id,
                bookmaker=bookmaker,
                source=source,
                home_odds=home_odds,
                away_odds=away_odds,
                draw_odds=draw_odds,
                home_spread=home_spread,
                home_spread_odds=home_spread_odds,
                away_spread_odds=away_spread_odds,
                over_under=over_under,
                over_odds=over_odds,
                under_odds=under_odds,
            )
            self.session.add(bo)
        self.session.flush()
        return bo

    def get_bookmaker_odds(self, match_id: int) -> list[BookmakerOdds]:
        return (
            self.session.query(BookmakerOdds)
            .filter_by(match_id=match_id)
            .all()
        )

    # ========== マスタデータ初期化 ==========

    def init_master_data(self):
        """sports.yaml からスポーツ・リーグマスタを登録"""
        with open(SPORTS_YAML, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        for sport_key, sport_data in config["sports"].items():
            self.upsert_sport(
                code=sport_data["code"],
                name=sport_data["name"],
                scoring_type=sport_data["scoring_type"],
            )
            for league_data in sport_data["leagues"]:
                self.upsert_league(
                    league_code=league_data["code"],
                    sport_code=sport_data["code"],
                    name=league_data["name"],
                    source_site=league_data["source_site"],
                    source_section=league_data["source_section"],
                )

        self.commit()
