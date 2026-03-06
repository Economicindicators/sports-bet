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
    Match,
    HandicapData,
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
    ) -> HandicapData:
        hd = (
            self.session.query(HandicapData).filter_by(match_id=match_id).first()
        )
        if hd:
            hd.handicap_team_id = handicap_team_id
            hd.handicap_value = handicap_value
            hd.result_type = result_type if result_type else hd.result_type
            hd.payout_rate = payout_rate if payout_rate else hd.payout_rate
        else:
            hd = HandicapData(
                match_id=match_id,
                handicap_team_id=handicap_team_id,
                handicap_value=handicap_value,
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
