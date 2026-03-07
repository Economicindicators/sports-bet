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
    import plotly.graph_objects as go
    from datetime import date, timedelta
    from sqlalchemy import func
    from config.settings import DB_PATH
    from config.constants import (
        MIN_EV_THRESHOLD, KELLY_FRACTION, HANDICAP_OUTCOMES,
        FAVORABLE_PAYOUT_THRESHOLD,
    )
    from database.models import get_session, Match, HandicapData, Team
except Exception as e:
    st.error(f"Import error: {e}")
    st.code(traceback.format_exc())
    st.stop()

# ============================================================
# Custom CSS
# ============================================================
CUSTOM_CSS = """
<style>
/* --- Global --- */
.block-container { padding-top: 1.5rem; }
h1, h2, h3 { letter-spacing: -0.02em; }

/* --- Metric Cards --- */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1A1F2E 0%, #232A3B 100%);
    border: 1px solid #2D3548;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
div[data-testid="stMetric"] label {
    color: #8892A4 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #FAFAFA !important;
}

/* --- Pick Cards --- */
.pick-card {
    background: linear-gradient(135deg, #1A1F2E 0%, #20273A 100%);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
    border-left: 4px solid #444;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    transition: transform 0.15s, box-shadow 0.15s;
}
.pick-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.35);
}
.pick-card.bet  { border-left-color: #00D4AA; }
.pick-card.watch { border-left-color: #FFD166; }
.pick-card.skip  { border-left-color: #EF476F; }

.pick-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.pick-matchup {
    font-size: 1.1rem;
    font-weight: 700;
    color: #FAFAFA;
}
.pick-handicap {
    font-size: 0.85rem;
    color: #8892A4;
    margin-top: 2px;
}
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-bet   { background: rgba(0,212,170,0.15); color: #00D4AA; border: 1px solid rgba(0,212,170,0.3); }
.badge-watch { background: rgba(255,209,102,0.15); color: #FFD166; border: 1px solid rgba(255,209,102,0.3); }
.badge-skip  { background: rgba(239,71,111,0.15); color: #EF476F; border: 1px solid rgba(239,71,111,0.3); }

.pick-metrics {
    display: flex;
    gap: 24px;
    margin-top: 12px;
}
.pick-metric {
    display: flex;
    flex-direction: column;
}
.pick-metric-label {
    font-size: 0.7rem;
    color: #8892A4;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.pick-metric-value {
    font-size: 1.15rem;
    font-weight: 700;
    color: #FAFAFA;
}

/* --- Confidence Bar --- */
.conf-bar-bg {
    background: #2D3548;
    border-radius: 6px;
    height: 8px;
    width: 100%;
    margin-top: 8px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.4s ease;
}

/* --- Payout Table --- */
.payout-table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
}
.payout-table th {
    background: #1A1F2E;
    color: #8892A4;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 2px solid #2D3548;
}
.payout-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #2D3548;
    font-size: 0.9rem;
}
.payout-table tr:hover { background: rgba(255,255,255,0.03); }
.payout-win  { color: #00D4AA; font-weight: 600; }
.payout-push { color: #FFD166; font-weight: 600; }
.payout-loss { color: #EF476F; font-weight: 600; }

/* --- Sidebar --- */
section[data-testid="stSidebar"] .stMarkdown h1 {
    font-size: 1.4rem;
    margin-bottom: 0;
}
.sidebar-logo {
    text-align: center;
    padding: 12px 0 20px 0;
    border-bottom: 1px solid #2D3548;
    margin-bottom: 16px;
}
.sidebar-logo .icon { font-size: 2.2rem; }
.sidebar-logo .title {
    font-size: 1.3rem;
    font-weight: 800;
    color: #FAFAFA;
    letter-spacing: -0.02em;
}
.sidebar-logo .subtitle {
    font-size: 0.75rem;
    color: #8892A4;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.sidebar-section {
    font-size: 0.7rem;
    color: #8892A4;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 20px;
    margin-bottom: 6px;
}
.sidebar-stat {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    font-size: 0.85rem;
}
.sidebar-stat .label { color: #8892A4; }
.sidebar-stat .value { color: #FAFAFA; font-weight: 600; }

/* --- Tabs --- */
button[data-baseweb="tab"] {
    border-radius: 8px !important;
    margin-right: 4px !important;
    font-weight: 600 !important;
}

/* --- Tooltip --- */
.tooltip {
    position: relative;
    display: inline-block;
    cursor: help;
    border-bottom: 1px dotted #8892A4;
}
.tooltip .tooltiptext {
    visibility: hidden;
    background: #232A3B;
    color: #FAFAFA;
    text-align: left;
    border-radius: 8px;
    padding: 10px 14px;
    position: absolute;
    z-index: 100;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    width: 240px;
    font-size: 0.8rem;
    line-height: 1.5;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    border: 1px solid #2D3548;
    opacity: 0;
    transition: opacity 0.2s;
}
.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# DB接続
# ============================================================
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

# ============================================================
# Helpers
# ============================================================
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


def plotly_dark_layout(fig, title: str = ""):
    """全チャート共通ダークテーマ"""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#FAFAFA")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8892A4"),
        xaxis=dict(gridcolor="#2D3548", zerolinecolor="#2D3548"),
        yaxis=dict(gridcolor="#2D3548", zerolinecolor="#2D3548"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def render_pick_card(
    home: str, away: str, handicap_team: str, handicap_value: float,
    prob: float, ev: float, kelly_amt: float, signal: str,
) -> str:
    """HTMLピックカード生成"""
    css_class = signal.lower()
    badge_class = f"badge-{css_class}"

    # Confidence bar color
    if signal == "BET":
        bar_color = "#00D4AA"
    elif signal == "WATCH":
        bar_color = "#FFD166"
    else:
        bar_color = "#EF476F"

    conf_pct = min(max(prob * 100, 0), 100)

    kelly_display = f"¥{kelly_amt:,.0f}" if kelly_amt > 0 else "—"

    return f"""
    <div class="pick-card {css_class}">
        <div class="pick-header">
            <div>
                <div class="pick-matchup">{home} vs {away}</div>
                <div class="pick-handicap">Handicap: {handicap_team} {handicap_value:+.1f}</div>
            </div>
            <span class="badge {badge_class}">{signal}</span>
        </div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{conf_pct:.0f}%; background:{bar_color};"></div>
        </div>
        <div class="pick-metrics">
            <div class="pick-metric">
                <span class="pick-metric-label">Probability</span>
                <span class="pick-metric-value">{prob:.1%}</span>
            </div>
            <div class="pick-metric">
                <span class="pick-metric-label">EV</span>
                <span class="pick-metric-value" style="color:{bar_color};">{ev:+.3f}</span>
            </div>
            <div class="pick-metric">
                <span class="pick-metric-label">Kelly Bet</span>
                <span class="pick-metric-value">{kelly_display}</span>
            </div>
        </div>
    </div>
    """


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="icon">⚽</div>
        <div class="title">Sports Bet</div>
        <div class="subtitle">Handicap Prediction System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">League & Settings</div>',
                unsafe_allow_html=True)
    selected_league_label = st.selectbox("League", list(LEAGUE_OPTIONS.keys()))
    selected_league = LEAGUE_OPTIONS[selected_league_label]
    selected_sport = LEAGUE_TO_SPORT[selected_league]

    ev_threshold = st.slider(
        "EV Threshold",
        0.0, 0.20, MIN_EV_THRESHOLD, 0.01,
        help="期待値がこの値以上ならBET推奨",
    )
    bankroll = st.number_input(
        "Bankroll (¥)",
        value=100000, step=10000,
        help="Kelly基準でベットサイズ計算に使用",
    )

    st.markdown('<div class="sidebar-section">Database Stats</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sidebar-stat"><span class="label">Total Matches</span><span class="value">{total_matches:,}</span></div>
    """, unsafe_allow_html=True)
    for lc, cnt in sorted(league_counts.items(), key=lambda x: -x[1]):
        st.markdown(f"""
        <div class="sidebar-stat"><span class="label">{lc.upper()}</span><span class="value">{cnt:,}</span></div>
        """, unsafe_allow_html=True)

    with st.expander("💡 Quick Help"):
        st.markdown("""
        - **BET**: EV ≥ threshold → 推奨ベット
        - **WATCH**: EV > 0 → 様子見
        - **SKIP**: EV ≤ 0 → 見送り
        - **EV**: Expected Value (期待値)
        - **Kelly**: 最適ベットサイズ計算
        """)


# ============================================================
# Data loaders (cached)
# ============================================================
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


# ============================================================
# KPI Header
# ============================================================
sched_df = load_scheduled_matches(selected_league)
match_df = load_match_data(selected_league)
team_names = load_team_names(selected_league)

sched_count = len(sched_df) if not sched_df.empty else 0
fav_rate = (match_df["payout_rate"] >= FAVORABLE_PAYOUT_THRESHOLD).mean() if not match_df.empty else 0

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("DB Matches", f"{total_matches:,}", help="全リーグの総試合数")
kpi2.metric(f"{selected_league_label} Matches",
            f"{league_counts.get(selected_league, 0):,}",
            help="選択リーグの試合数")
kpi3.metric("Scheduled", f"{sched_count}",
            help="未消化の試合数")
kpi4.metric("Fav Win Rate", f"{fav_rate:.1%}",
            help="ハンデ有利チームの5分勝ち以上率")
kpi5.metric("EV Threshold", f"{ev_threshold:.2f}",
            help="BET推奨の最低期待値")

st.markdown("")

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Picks", "📊 Data", "📈 Backtest", "🏆 Teams", "💰 Bankroll", "📖 Manual"
])

# --- Tab 1: Picks ---
with tab1:
    st.subheader("Today's Picks")

    if sched_df.empty:
        st.info(f"No scheduled matches for {selected_league_label}")
    else:
        st.caption(f"{sched_count} scheduled matches")

        with st.spinner("Running prediction model..."):
            preds = run_predictions(selected_sport)

        if preds.empty:
            st.warning("Model not available — showing matches without predictions")
            for _, row in sched_df.iterrows():
                ht = team_names.get(row["home_team_id"], "?")
                at = team_names.get(row["away_team_id"], "?")
                hdt = team_names.get(row["handicap_team_id"], "?")
                card_html = render_pick_card(
                    ht, at, hdt, row["handicap_value"],
                    prob=0, ev=0, kelly_amt=0, signal="SKIP",
                )
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            from betting.kelly import calculate_bet_size

            merged = sched_df.merge(
                preds[["match_id", "pred_prob", "handicap_ev"]],
                on="match_id", how="left",
            )

            # Determine signal and sort BET first
            def _signal(ev_val):
                if ev_val >= ev_threshold:
                    return "BET"
                elif ev_val > 0:
                    return "WATCH"
                return "SKIP"

            merged["ev_val"] = merged["handicap_ev"].fillna(0)
            merged["prob_val"] = merged["pred_prob"].fillna(0)
            merged["signal"] = merged["ev_val"].apply(_signal)
            signal_order = {"BET": 0, "WATCH": 1, "SKIP": 2}
            merged["sort_key"] = merged["signal"].map(signal_order)
            merged = merged.sort_values(["sort_key", "ev_val"], ascending=[True, False])

            # Summary counts
            bet_count = (merged["signal"] == "BET").sum()
            watch_count = (merged["signal"] == "WATCH").sum()
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("BET", bet_count)
            sc2.metric("WATCH", watch_count)
            sc3.metric("SKIP", len(merged) - bet_count - watch_count)

            for _, row in merged.iterrows():
                ht = team_names.get(row["home_team_id"], "?")
                at = team_names.get(row["away_team_id"], "?")
                hdt = team_names.get(row["handicap_team_id"], "?")
                kelly_amt = calculate_bet_size(row["prob_val"], bankroll) if row["prob_val"] > 0 else 0

                card_html = render_pick_card(
                    ht, at, hdt, row["handicap_value"],
                    prob=row["prob_val"], ev=row["ev_val"],
                    kelly_amt=kelly_amt, signal=row["signal"],
                )
                st.markdown(card_html, unsafe_allow_html=True)

# --- Tab 2: Data ---
with tab2:
    st.subheader("Match Data")

    if match_df.empty:
        st.warning("No data available")
    else:
        n = len(match_df)
        fav = (match_df["payout_rate"] >= FAVORABLE_PAYOUT_THRESHOLD).mean()
        avg_p = match_df["payout_rate"].mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Matches", f"{n:,}", help="終了済み試合数")
        c2.metric("Favorable Rate", f"{fav:.1%}",
                  help="5分勝ち以上 (payout ≥ 1.5) の割合")
        c3.metric("Avg Payout", f"{avg_p:.2f}x",
                  help="平均ペイアウト倍率")

        col1, col2 = st.columns(2)
        with col1:
            rc = match_df["result_type"].value_counts()
            fig = px.bar(x=rc.index, y=rc.values,
                         labels={"x": "Result", "y": "Count"},
                         color=rc.index,
                         color_discrete_map={
                             "丸勝ち": "#00D4AA", "7分勝ち": "#00D4AA",
                             "5分勝ち": "#66E0C8", "3分勝ち": "#66E0C8",
                             "勝負無し": "#FFD166",
                             "3分負け": "#FF8FA3", "5分負け": "#EF476F",
                             "7分負け": "#EF476F", "丸負け": "#C5283D",
                         })
            fig.update_layout(showlegend=False)
            plotly_dark_layout(fig, "Handicap Result Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            monthly = match_df.set_index("date").resample("M")["payout_rate"].mean().reset_index()
            monthly.columns = ["Month", "Avg Payout"]
            fig2 = px.line(monthly, x="Month", y="Avg Payout")
            fig2.add_hline(y=1.0, line_dash="dash", line_color="#8892A4", opacity=0.5)
            plotly_dark_layout(fig2, "Monthly Avg Payout")
            st.plotly_chart(fig2, use_container_width=True)

        show_n = st.slider("Recent matches to display", 20, 200, 50, 10)
        recent = match_df.tail(show_n).copy()
        recent["Home"] = recent["home_team_id"].map(team_names)
        recent["Away"] = recent["away_team_id"].map(team_names)
        recent["Handi Team"] = recent["handicap_team_id"].map(team_names)
        recent["Score"] = recent["home_score"].astype(str) + "-" + recent["away_score"].astype(str)
        disp = recent[["date", "Home", "Away", "Score", "Handi Team",
                        "handicap_value", "result_type", "payout_rate"]].copy()
        disp.columns = ["Date", "Home", "Away", "Score", "Handi Team",
                        "Handicap", "Result", "Payout"]
        st.dataframe(
            disp.sort_values("Date", ascending=False),
            use_container_width=True, height=400,
        )

# --- Tab 3: Backtest ---
with tab3:
    st.subheader("Walk-Forward Backtest")

    with st.spinner("Loading backtest... (cached after first run)"):
        result = run_backtest(selected_league, selected_sport, ev_threshold)

    if result is None or result.total_bets == 0:
        st.info("Insufficient data for backtest. Collect more match data first.")
    else:
        win_rate = result.wins / result.total_bets if result.total_bets > 0 else 0
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Bets", f"{result.total_bets}",
                  help="バックテスト期間中のベット回数")
        c2.metric("Win Rate", f"{win_rate:.1%}",
                  help="勝率 (payout ≥ 1.0)")
        c3.metric("ROI", f"{result.roi:+.1f}%",
                  help="投資収益率")
        c4.metric("PnL", f"¥{result.total_pnl:+,.0f}",
                  help="損益合計")
        c5.metric("Max DD", f"{result.max_drawdown:.1%}",
                  help="最大ドローダウン")

        st.dataframe(pd.DataFrame(result.period_results), use_container_width=True)

        if result.bet_history:
            h = pd.DataFrame(result.bet_history)
            h["cum"] = h["pnl"].cumsum()
            h["date"] = pd.to_datetime(h["date"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=h["date"], y=h["cum"],
                fill="tozeroy",
                fillcolor="rgba(0,212,170,0.1)",
                line=dict(color="#00D4AA", width=2),
                name="Cumulative PnL",
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="#8892A4", opacity=0.5)
            plotly_dark_layout(fig, "Cumulative PnL")
            st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: Teams ---
with tab4:
    st.subheader("Team Analysis")

    if not match_df.empty:
        sel = st.selectbox("Team", sorted(team_names.values()))
        tid = next((k for k, v in team_names.items() if v == sel), None)
        if tid:
            hdf = match_df[match_df["handicap_team_id"] == tid].copy()
            alldf = match_df[(match_df["home_team_id"] == tid) | (match_df["away_team_id"] == tid)]

            avg_payout = hdf["payout_rate"].mean() if len(hdf) > 0 else 0
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Matches", f"{len(alldf)}")
            c2.metric("As Favorite", f"{len(hdf)}")
            if len(hdf) > 0:
                c3.metric("Win Rate", f"{(hdf['payout_rate'] >= FAVORABLE_PAYOUT_THRESHOLD).mean():.1%}",
                          help="5分勝ち以上の割合")
                c4.metric("Avg Payout", f"{avg_payout:.2f}x",
                          help="ハンデ有利時の平均ペイアウト")

            col_l, col_r = st.columns(2)

            if len(hdf) > 0:
                with col_l:
                    hdf = hdf.sort_values("date")
                    hdf["cum"] = (hdf["payout_rate"] - 1.0).cumsum() * 1000
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hdf["date"], y=hdf["cum"],
                        fill="tozeroy",
                        fillcolor="rgba(0,212,170,0.1)",
                        line=dict(color="#00D4AA", width=2),
                        name="Handicap PnL",
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="#8892A4", opacity=0.5)
                    plotly_dark_layout(fig, f"{sel} — Handicap PnL")
                    st.plotly_chart(fig, use_container_width=True)

                with col_r:
                    rc = hdf["result_type"].value_counts()
                    fig2 = go.Figure(data=[go.Pie(
                        labels=rc.index, values=rc.values,
                        hole=0.5,
                        marker=dict(colors=[
                            "#00D4AA", "#66E0C8", "#FFD166",
                            "#FF8FA3", "#EF476F", "#C5283D",
                            "#8892A4", "#5A6275", "#3D4559",
                        ]),
                        textinfo="label+percent",
                        textfont=dict(size=12, color="#FAFAFA"),
                    )])
                    plotly_dark_layout(fig2, f"{sel} — Result Distribution")
                    st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No data available")

# --- Tab 5: Bankroll ---
with tab5:
    st.subheader("Bankroll Management")

    with st.spinner("Loading simulation..."):
        bk_result = run_backtest(selected_league, selected_sport, ev_threshold)

    if bk_result and bk_result.bet_history:
        h = pd.DataFrame(bk_result.bet_history)
        h["date"] = pd.to_datetime(h["date"])
        h["bankroll"] = bankroll + (h["pnl"] / 1000 * 1000).cumsum()
        final = h["bankroll"].iloc[-1]

        c1, c2, c3 = st.columns(3)
        c1.metric("Initial", f"¥{bankroll:,.0f}")
        c2.metric("Final", f"¥{final:,.0f}")
        ret_pct = (final - bankroll) / bankroll * 100
        c3.metric("Return", f"{ret_pct:+.1f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=h["date"], y=h["bankroll"],
            fill="tozeroy",
            fillcolor="rgba(0,212,170,0.08)",
            line=dict(color="#00D4AA", width=2),
            name="Bankroll",
        ))
        fig.add_hline(y=bankroll, line_dash="dash", line_color="#FFD166", opacity=0.5,
                      annotation_text="Initial")
        plotly_dark_layout(fig, "Bankroll Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insufficient data for bankroll simulation.")

# --- Tab 6: Manual ---
with tab6:
    st.subheader("System Manual")

    st.markdown("---")

    # --- Overview ---
    st.markdown("#### System Overview")
    st.markdown("""
    This system scrapes handicap data from Japanese sports betting sites,
    builds features, trains a LightGBM model, and generates EV-based bet recommendations.
    """)

    ov1, ov2 = st.columns(2)
    with ov1:
        st.markdown("**Data Sources**")
        st.markdown("""
        | Site | Sports |
        |------|--------|
        | ハンデの森 | NPB, MLB |
        | ハンデの嵐 | J League, Premier, La Liga, Serie A, Bundesliga |
        | ハンディマン | NBA, B League |
        """)
    with ov2:
        st.markdown("**Model**")
        st.markdown("""
        - **Algorithm**: LightGBM (binary classifier)
        - **Target**: Handicap favorite wins by ≥ 5分 (payout ≥ 1.5)
        - **Features**: 41 features (handicap, team stats, H2H, schedule, sport-specific)
        - **Validation**: Time-series CV (expanding window + 30-day gap)
        """)

    st.markdown("---")

    # --- Payout Table ---
    st.markdown("#### 9-Level Handicap Payout")
    st.markdown("""
    The handicap system uses `adjusted_margin = (Fav Score − Unfav Score) − Handicap Value`
    to determine one of 9 outcomes:
    """)

    payout_rows = ""
    for lower, upper, label, payout in HANDICAP_OUTCOMES:
        if payout >= 1.5:
            css = "payout-win"
        elif payout >= 1.0:
            css = "payout-push"
        else:
            css = "payout-loss"

        if lower == float("-inf"):
            range_str = f"< {upper:.1f}"
        elif upper == float("inf"):
            range_str = f"≥ {lower:.1f}"
        elif abs(upper - lower) < 0.01:
            range_str = f"= {lower:.1f}"
        else:
            range_str = f"{lower:.1f} ~ {upper:.1f}"

        payout_rows += f"""
        <tr>
            <td class="{css}">{label}</td>
            <td>{range_str}</td>
            <td class="{css}">{payout:.1f}x</td>
        </tr>"""

    st.markdown(f"""
    <table class="payout-table">
        <thead>
            <tr><th>Result</th><th>Adjusted Margin</th><th>Payout</th></tr>
        </thead>
        <tbody>{payout_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- EV Calculation ---
    st.markdown("#### Expected Value (EV)")
    st.markdown("""
    EV measures the expected profit per unit bet:

    ```
    EV = P(fav) × 1.73  +  P(push) × 1.0  +  P(unfav) × 0.50  −  1.0
    ```

    - **P(fav)**: Probability the favorite wins by ≥ 5分 (model output)
    - **P(push)**: Probability of a draw/push (~2%)
    - **P(unfav)**: 1 − P(fav) − P(push)
    - **1.73, 1.0, 0.50**: Weighted average payouts for each outcome group

    **Interpretation**: EV > 0 means positive expected return. Higher is better.
    """)

    st.markdown("---")

    # --- Kelly Criterion ---
    st.markdown("#### Kelly Criterion")
    st.markdown("""
    The Kelly criterion determines the optimal bet size to maximize long-term growth:

    ```
    Full Kelly:  f* = (p × b − q) / b
    Our system:  Bet = f* × 0.25 × Bankroll  (1/4 Kelly)
    ```

    - **p**: Win probability (model output)
    - **q**: 1 − p
    - **b**: Net odds (avg_payout − 1.0)
    - **0.25**: Conservative fraction to reduce variance

    Maximum bet is capped at 20% of bankroll.
    """)

    st.markdown("---")

    # --- Signal Logic ---
    st.markdown("#### Bet Signal Logic")

    sig1, sig2, sig3 = st.columns(3)
    with sig1:
        st.markdown(f"""
        <div class="pick-card bet" style="text-align:center;">
            <span class="badge badge-bet">BET</span>
            <div style="margin-top:8px; color:#8892A4; font-size:0.85rem;">
                EV ≥ {MIN_EV_THRESHOLD:.2f}
            </div>
            <div style="color:#FAFAFA; font-size:0.85rem;">
                Recommended bet with Kelly sizing
            </div>
        </div>
        """, unsafe_allow_html=True)
    with sig2:
        st.markdown("""
        <div class="pick-card watch" style="text-align:center;">
            <span class="badge badge-watch">WATCH</span>
            <div style="margin-top:8px; color:#8892A4; font-size:0.85rem;">
                0 < EV < threshold
            </div>
            <div style="color:#FAFAFA; font-size:0.85rem;">
                Positive EV but below threshold
            </div>
        </div>
        """, unsafe_allow_html=True)
    with sig3:
        st.markdown("""
        <div class="pick-card skip" style="text-align:center;">
            <span class="badge badge-skip">SKIP</span>
            <div style="margin-top:8px; color:#8892A4; font-size:0.85rem;">
                EV ≤ 0
            </div>
            <div style="color:#FAFAFA; font-size:0.85rem;">
                Negative expected value — do not bet
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Glossary ---
    st.markdown("#### Glossary")
    glossary = {
        "Handicap (ハンデ)": "The point spread given to the favorite team to level the playing field.",
        "Adjusted Margin": "(Favorite Score − Underdog Score) − Handicap Value. Determines the payout tier.",
        "Payout Rate": "The multiplier applied to your bet. 1.0 = break even, >1.0 = profit, <1.0 = loss.",
        "Favorable Rate": "Percentage of matches where payout ≥ 1.5 (5分勝ち or better).",
        "EV (Expected Value)": "The average profit per unit bet. Positive EV = profitable long-term.",
        "Kelly Criterion": "Mathematical formula for optimal bet sizing based on edge and odds.",
        "Walk-Forward Backtest": "Backtest method that re-trains the model at each period to avoid look-ahead bias.",
        "ROI (Return on Investment)": "Total profit divided by total amount wagered, expressed as %.",
        "Max Drawdown": "Largest peak-to-trough decline in cumulative PnL.",
        "Sharpe Ratio": "Risk-adjusted return. Higher = better risk/reward.",
    }
    for term, definition in glossary.items():
        st.markdown(f"**{term}** — {definition}")
