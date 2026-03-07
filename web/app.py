"""Streamlit Dashboard — Sports Bet Handicap Prediction"""

from __future__ import annotations

import sys
import traceback
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="Sports Bet", page_icon="⚽", layout="wide")

try:
    import pandas as pd
    import numpy as np
    import plotly.express as px
    from datetime import date, timedelta
    from sqlalchemy import func
    from config.settings import DB_PATH
    from config.constants import MIN_EV_THRESHOLD, KELLY_FRACTION
    from database.models import get_session, Match, HandicapData, Team
except Exception as e:
    st.error(f"Import error: {e}")
    st.code(traceback.format_exc())
    st.stop()

# --- DB接続 ---
try:
    sess = get_session()
    total_matches = sess.query(func.count(Match.match_id)).scalar()
    league_counts = dict(
        sess.query(Match.league_code, func.count(Match.match_id))
        .group_by(Match.league_code).all()
    )
    sess.close()
except Exception as e:
    st.error(f"DB error: {e}")
    st.code(traceback.format_exc())
    st.stop()

# --- サイドバー ---
st.sidebar.title("⚽ Sports Bet")
LEAGUE_OPTIONS = {
    "NBA": "nba",
    "Premier League": "premier",
    "J League": "jleague",
    "NPB": "npb",
    "MLB": "mlb",
}
LEAGUE_TO_SPORT = {
    "npb": "baseball", "mlb": "baseball",
    "premier": "soccer", "jleague": "soccer",
    "nba": "basketball",
}
selected_league_label = st.sidebar.selectbox("League", list(LEAGUE_OPTIONS.keys()))
selected_league = LEAGUE_OPTIONS[selected_league_label]
selected_sport = LEAGUE_TO_SPORT[selected_league]
ev_threshold = st.sidebar.slider("EV Threshold", 0.0, 0.20, MIN_EV_THRESHOLD, 0.01)
bankroll = st.sidebar.number_input("Bankroll", value=100000, step=10000)
st.sidebar.markdown("---")
st.sidebar.caption(f"Total: {total_matches:,} matches")
for lc, cnt in sorted(league_counts.items(), key=lambda x: -x[1]):
    st.sidebar.caption(f"{lc}: {cnt:,}")


# --- ヘルパー ---
@st.cache_data(ttl=300)
def load_match_data(league_code):
    s = get_session()
    rows = s.query(
        Match.match_id, Match.date, Match.home_team_id, Match.away_team_id,
        Match.home_score, Match.away_score,
        HandicapData.handicap_team_id, HandicapData.handicap_value,
        HandicapData.result_type, HandicapData.payout_rate,
    ).join(HandicapData, Match.match_id == HandicapData.match_id
    ).filter(Match.league_code == league_code, Match.status == "finished"
    ).order_by(Match.date).all()
    s.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "match_id", "date", "home_team_id", "away_team_id",
        "home_score", "away_score",
        "handicap_team_id", "handicap_value", "result_type", "payout_rate",
    ])
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_team_names(league_code):
    s = get_session()
    teams = s.query(Team.team_id, Team.name).filter(Team.league_code == league_code).all()
    s.close()
    return {t[0]: t[1] for t in teams}


@st.cache_data(ttl=300)
def load_scheduled_matches(league_code):
    s = get_session()
    rows = s.query(
        Match.match_id, Match.date, Match.home_team_id, Match.away_team_id,
        HandicapData.handicap_team_id, HandicapData.handicap_value,
    ).join(HandicapData, Match.match_id == HandicapData.match_id
    ).filter(Match.league_code == league_code, Match.status == "scheduled"
    ).order_by(Match.date).all()
    s.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=[
        "match_id", "date", "home_team_id", "away_team_id",
        "handicap_team_id", "handicap_value",
    ])


@st.cache_data(ttl=600)
def run_predictions(sport_code):
    """MLモデルで予測を実行"""
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    try:
        from features.feature_pipeline import load_matches_df, build_features
        from models.training import load_model
        from betting.handicap_ev import calculate_handicap_ev

        model = load_model(sport_code, "v1")
        df = load_matches_df(sport_code=sport_code, include_scheduled=True)
        if df.empty:
            return pd.DataFrame()

        df, feat_cols = build_features(df, sport_code)
        df["pred_prob"] = model.predict_proba(df[feat_cols])
        df["handicap_ev"] = df["pred_prob"].apply(calculate_handicap_ev)

        # scheduledのみ返す
        scheduled = df[df["status"] == "scheduled"].copy()
        return scheduled[["match_id", "date", "home_team_id", "away_team_id",
                          "handicap_team_id", "handicap_value",
                          "pred_prob", "handicap_ev"]].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()
    finally:
        logging.disable(logging.NOTSET)


@st.cache_data(ttl=600)
def run_backtest(league_code, sport_code, ev_thresh):
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    try:
        from features.feature_pipeline import build_training_data
        from betting.backtest import walk_forward_backtest
        df, feat_cols = build_training_data(sport_code, league_code)
        if df.empty:
            return None
        n_periods = 4 if len(df) > 500 else 3
        return walk_forward_backtest(df, feat_cols, n_periods=n_periods, ev_threshold=ev_thresh)
    except Exception:
        return None
    finally:
        logging.disable(logging.NOTSET)


# === メイン ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Today's Picks", "📋 Match Data", "📊 Backtest", "🏆 Team Analysis", "💰 Bankroll"
])

# --- Tab 1: Today's Picks ---
with tab1:
    st.header("🔮 Today's Picks")

    sched = load_scheduled_matches(selected_league)
    if sched.empty:
        st.info(f"No scheduled matches for {selected_league_label}")
    else:
        team_names = load_team_names(selected_league)
        st.write(f"**{len(sched)} scheduled matches**")

        with st.spinner("Running prediction model..."):
            preds = run_predictions(selected_sport)

        if preds.empty:
            st.warning("Model not available for this sport")
            # Show matches without predictions
            for _, row in sched.iterrows():
                ht = team_names.get(row["home_team_id"], "?")
                at = team_names.get(row["away_team_id"], "?")
                hdt = team_names.get(row["handicap_team_id"], "?")
                st.write(f"- **{ht}** vs **{at}** | Handicap: {hdt} {row['handicap_value']}")
        else:
            # Merge predictions with scheduled matches
            merged = sched.merge(
                preds[["match_id", "pred_prob", "handicap_ev"]],
                on="match_id", how="left"
            )

            for _, row in merged.sort_values("handicap_ev", ascending=False).iterrows():
                ht = team_names.get(row["home_team_id"], "?")
                at = team_names.get(row["away_team_id"], "?")
                hdt = team_names.get(row["handicap_team_id"], "?")
                ev = row.get("handicap_ev", None)
                prob = row.get("pred_prob", None)

                ev_val = ev if pd.notna(ev) else 0
                prob_val = prob if pd.notna(prob) else 0

                # Card-style display
                if ev_val >= ev_threshold:
                    color = "🟢"
                    rec = "**BET**"
                elif ev_val > 0:
                    color = "🟡"
                    rec = "WATCH"
                else:
                    color = "🔴"
                    rec = "SKIP"

                from betting.kelly import calculate_bet_size
                kelly = calculate_bet_size(prob_val, bankroll) if prob_val > 0 else 0

                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.markdown(f"{color} **{ht}** vs **{at}**")
                    st.caption(f"Handicap: {hdt} {row['handicap_value']}")
                with col2:
                    st.metric("Prob", f"{prob_val:.1%}")
                with col3:
                    st.metric("EV", f"{ev_val:+.3f}")
                with col4:
                    st.metric("Bet", f"¥{kelly:,.0f}" if kelly > 0 else rec)
                st.divider()

# --- Tab 2: Match Data ---
with tab2:
    st.header("📋 Match Data")
    df = load_match_data(selected_league)
    if df.empty:
        st.warning("No data available")
    else:
        team_names = load_team_names(selected_league)
        n = len(df)
        fav = (df["payout_rate"] >= 1.5).mean()
        avg_p = df["payout_rate"].mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Matches", f"{n:,}")
        c2.metric("Favorable Rate", f"{fav:.1%}")
        c3.metric("Avg Payout", f"{avg_p:.2f}x")

        col1, col2 = st.columns(2)
        with col1:
            rc = df["result_type"].value_counts()
            fig = px.bar(x=rc.index, y=rc.values, title="Handicap Result Distribution",
                         labels={"x": "Result", "y": "Count"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            monthly = df.set_index("date").resample("M")["payout_rate"].mean().reset_index()
            monthly.columns = ["Month", "Avg Payout"]
            fig2 = px.line(monthly, x="Month", y="Avg Payout", title="Monthly Avg Payout")
            fig2.add_hline(y=1.0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Recent 50 Matches")
        recent = df.tail(50).copy()
        recent["Home"] = recent["home_team_id"].map(team_names)
        recent["Away"] = recent["away_team_id"].map(team_names)
        recent["Handi Team"] = recent["handicap_team_id"].map(team_names)
        recent["Score"] = recent["home_score"].astype(str) + "-" + recent["away_score"].astype(str)
        disp = recent[["date", "Home", "Away", "Score", "Handi Team",
                        "handicap_value", "result_type", "payout_rate"]].copy()
        disp.columns = ["Date", "Home", "Away", "Score", "Handi Team",
                        "Handicap", "Result", "Payout"]
        st.dataframe(disp.sort_values("Date", ascending=False),
                     use_container_width=True, height=400)

# --- Tab 3: Backtest ---
with tab3:
    st.header("📊 Walk-Forward Backtest")
    if st.button("Run Backtest", type="primary"):
        try:
            with st.spinner("Running... (2-3 min first time)"):
                result = run_backtest(selected_league, selected_sport, ev_threshold)
            if result is None or result.total_bets == 0:
                st.warning("Insufficient data")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Bets", f"{result.total_bets}")
                c2.metric("ROI", f"{result.roi:+.1f}%")
                c3.metric("PnL", f"¥{result.total_pnl:+,.0f}")
                c4.metric("Sharpe", f"{result.sharpe:.2f}")

                st.dataframe(pd.DataFrame(result.period_results), use_container_width=True)

                if result.bet_history:
                    h = pd.DataFrame(result.bet_history)
                    h["cum"] = h["pnl"].cumsum()
                    h["date"] = pd.to_datetime(h["date"])
                    fig = px.line(h, x="date", y="cum", title="Cumulative PnL",
                                  labels={"date": "Date", "cum": "PnL"})
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
            st.code(traceback.format_exc())
    else:
        st.info("Click button to run backtest (2-3 min first time)")

# --- Tab 4: Team Analysis ---
with tab4:
    st.header("🏆 Team Analysis")
    df = load_match_data(selected_league)
    if not df.empty:
        team_names = load_team_names(selected_league)
        sel = st.selectbox("Team", sorted(team_names.values()))
        tid = next((k for k, v in team_names.items() if v == sel), None)
        if tid:
            hdf = df[df["handicap_team_id"] == tid].copy()
            alldf = df[(df["home_team_id"] == tid) | (df["away_team_id"] == tid)]
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Matches", f"{len(alldf)}")
            c2.metric("As Favorite", f"{len(hdf)}")
            if len(hdf) > 0:
                c3.metric("Win Rate", f"{(hdf['payout_rate'] >= 1.5).mean():.1%}")
                hdf = hdf.sort_values("date")
                hdf["cum"] = (hdf["payout_rate"] - 1.0).cumsum() * 1000
                fig = px.line(hdf, x="date", y="cum",
                              title=f"{sel} Handicap PnL",
                              labels={"date": "Date", "cum": "PnL"})
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)

# --- Tab 5: Bankroll ---
with tab5:
    st.header("💰 Bankroll Management")
    if st.button("Run Simulation", type="primary"):
        try:
            with st.spinner("Running..."):
                result = run_backtest(selected_league, selected_sport, ev_threshold)
            if result and result.bet_history:
                h = pd.DataFrame(result.bet_history)
                h["date"] = pd.to_datetime(h["date"])
                h["bankroll"] = bankroll + (h["pnl"] / 1000 * 1000).cumsum()
                final = h["bankroll"].iloc[-1]
                c1, c2, c3 = st.columns(3)
                c1.metric("Initial", f"¥{bankroll:,.0f}")
                c2.metric("Final", f"¥{final:,.0f}")
                c3.metric("Return", f"{(final - bankroll) / bankroll * 100:+.1f}%")
                fig = px.line(h, x="date", y="bankroll", title="Bankroll Over Time")
                fig.add_hline(y=bankroll, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Click to simulate bankroll based on backtest")
