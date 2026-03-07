"""Streamlit ダッシュボード — スポーツベット ハンデ予測"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="Sports Bet", page_icon="⚽", layout="wide")
st.title("⚽ スポーツベット ハンデ予測")

# Step 1: 基本表示
st.write(f"Python: `{sys.version}`")
st.write(f"Project root: `{PROJECT_ROOT}`")

# Step 2: DB確認
try:
    from config.settings import DB_PATH
    st.write(f"DB path: `{DB_PATH}`")
    st.write(f"DB exists: `{DB_PATH.exists()}`")
    if DB_PATH.exists():
        st.write(f"DB size: `{DB_PATH.stat().st_size / 1024:.0f} KB`")
except Exception as e:
    st.error(f"Config error: {e}")
    st.code(traceback.format_exc())
    st.stop()

# Step 3: DB接続
try:
    import pandas as pd
    from sqlalchemy import func
    from database.models import get_session, Match, HandicapData, Team
    sess = get_session()
    total = sess.query(func.count(Match.match_id)).scalar()
    st.success(f"DB接続OK: {total:,} 試合")
    league_counts = dict(
        sess.query(Match.league_code, func.count(Match.match_id))
        .group_by(Match.league_code).all()
    )
    sess.close()
    for lc, cnt in sorted(league_counts.items(), key=lambda x: -x[1]):
        st.write(f"- {lc}: {cnt:,} 試合")
except Exception as e:
    st.error(f"DB error: {e}")
    st.code(traceback.format_exc())
    st.stop()

# Step 4: メイン機能
try:
    import numpy as np
    import plotly.express as px
    from datetime import date, timedelta
    from config.constants import MIN_EV_THRESHOLD, KELLY_FRACTION

    st.success("全モジュール読み込みOK")

    # === サイドバー ===
    st.sidebar.title("⚽ Sports Bet")
    LEAGUE_OPTIONS = {
        "NPB (野球)": "npb", "MLB (野球)": "mlb",
        "プレミアリーグ": "premier", "Jリーグ": "jleague", "NBA": "nba",
    }
    LEAGUE_TO_SPORT = {
        "npb": "baseball", "mlb": "baseball",
        "premier": "soccer", "jleague": "soccer", "nba": "basketball",
    }
    selected_league_label = st.sidebar.selectbox("リーグ", list(LEAGUE_OPTIONS.keys()), index=2)
    selected_league = LEAGUE_OPTIONS[selected_league_label]
    selected_sport = LEAGUE_TO_SPORT[selected_league]
    ev_threshold = st.sidebar.slider("EV閾値", 0.0, 0.20, MIN_EV_THRESHOLD, 0.01)
    bankroll = st.sidebar.number_input("資金 (円)", value=100000, step=10000)
    st.sidebar.markdown("---")
    for lc, cnt in sorted(league_counts.items(), key=lambda x: -x[1]):
        st.sidebar.caption(f"{lc}: {cnt:,}")

    # === ヘルパー ===
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

    @st.cache_data(ttl=600)
    def run_backtest(league_code, sport_code, ev_thresh):
        import warnings
        warnings.filterwarnings("ignore")
        from features.feature_pipeline import build_training_data
        from betting.backtest import walk_forward_backtest
        df, feat_cols = build_training_data(sport_code, league_code)
        if df.empty:
            return None
        n_periods = 4 if len(df) > 500 else 3
        return walk_forward_backtest(df, feat_cols, n_periods=n_periods, ev_threshold=ev_thresh)

    # === タブ ===
    tab1, tab2, tab3, tab4 = st.tabs(["📋 試合データ", "📊 バックテスト", "🏆 チーム分析", "💰 資金管理"])

    # --- Tab 1: 試合データ ---
    with tab1:
        df = load_match_data(selected_league)
        if df.empty:
            st.warning("データがありません")
        else:
            team_names = load_team_names(selected_league)
            n = len(df)
            fav = (df["payout_rate"] >= 1.5).mean()
            avg_p = df["payout_rate"].mean()

            c1, c2, c3 = st.columns(3)
            c1.metric("試合数", f"{n:,}")
            c2.metric("ハンデ有利率", f"{fav:.1%}")
            c3.metric("平均ペイアウト", f"{avg_p:.2f}x")

            col1, col2 = st.columns(2)
            with col1:
                rc = df["result_type"].value_counts()
                fig = px.bar(x=rc.index, y=rc.values, title="ハンデ結果分布",
                             labels={"x": "結果", "y": "件数"})
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                monthly = df.set_index("date").resample("M")["payout_rate"].mean().reset_index()
                monthly.columns = ["月", "平均ペイアウト"]
                fig2 = px.line(monthly, x="月", y="平均ペイアウト", title="月別平均ペイアウト")
                fig2.add_hline(y=1.0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("直近50試合")
            recent = df.tail(50).copy()
            recent["ホーム"] = recent["home_team_id"].map(team_names)
            recent["アウェイ"] = recent["away_team_id"].map(team_names)
            recent["ハンデチーム"] = recent["handicap_team_id"].map(team_names)
            recent["スコア"] = recent["home_score"].astype(str) + "-" + recent["away_score"].astype(str)
            disp = recent[["date", "ホーム", "アウェイ", "スコア", "ハンデチーム",
                           "handicap_value", "result_type", "payout_rate"]].copy()
            disp.columns = ["日付", "ホーム", "アウェイ", "スコア", "ハンデチーム",
                            "ハンデ値", "結果", "ペイアウト"]
            st.dataframe(disp.sort_values("日付", ascending=False),
                         use_container_width=True, height=400)

    # --- Tab 2: バックテスト ---
    with tab2:
        st.header("Walk-Forward バックテスト")
        if st.button("▶ バックテスト実行", type="primary"):
            try:
                with st.spinner("実行中... (初回2-3分)"):
                    result = run_backtest(selected_league, selected_sport, ev_threshold)
                if result is None or result.total_bets == 0:
                    st.warning("データ不足")
                else:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("ベット数", f"{result.total_bets}")
                    c2.metric("ROI", f"{result.roi:+.1f}%")
                    c3.metric("PnL", f"¥{result.total_pnl:+,.0f}")
                    c4.metric("Sharpe", f"{result.sharpe:.2f}")

                    st.dataframe(pd.DataFrame(result.period_results), use_container_width=True)

                    if result.bet_history:
                        h = pd.DataFrame(result.bet_history)
                        h["cum"] = h["pnl"].cumsum()
                        h["date"] = pd.to_datetime(h["date"])
                        fig = px.line(h, x="date", y="cum", title="累積PnL",
                                      labels={"date": "日付", "cum": "PnL (円)"})
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())
        else:
            st.info("👆 ボタンで実行 (初回2-3分)")

    # --- Tab 3: チーム分析 ---
    with tab3:
        st.header("チーム分析")
        df = load_match_data(selected_league)
        if not df.empty:
            team_names = load_team_names(selected_league)
            sel = st.selectbox("チーム", sorted(team_names.values()))
            tid = next((k for k, v in team_names.items() if v == sel), None)
            if tid:
                hdf = df[df["handicap_team_id"] == tid].copy()
                alldf = df[(df["home_team_id"] == tid) | (df["away_team_id"] == tid)]
                c1, c2, c3 = st.columns(3)
                c1.metric("総試合", f"{len(alldf)}")
                c2.metric("ハンデ有利", f"{len(hdf)}")
                if len(hdf) > 0:
                    c3.metric("勝率", f"{(hdf['payout_rate'] >= 1.5).mean():.1%}")
                    hdf = hdf.sort_values("date")
                    hdf["cum"] = (hdf["payout_rate"] - 1.0).cumsum() * 1000
                    fig = px.line(hdf, x="date", y="cum",
                                  title=f"{sel} ハンデ累積損益",
                                  labels={"date": "日付", "cum": "PnL (円)"})
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig, use_container_width=True)

    # --- Tab 4: 資金管理 ---
    with tab4:
        st.header("資金管理")
        if st.button("▶ シミュレーション", type="primary"):
            try:
                with st.spinner("実行中..."):
                    result = run_backtest(selected_league, selected_sport, ev_threshold)
                if result and result.bet_history:
                    h = pd.DataFrame(result.bet_history)
                    h["date"] = pd.to_datetime(h["date"])
                    h["bankroll"] = bankroll + (h["pnl"] / 1000 * 1000).cumsum()
                    final = h["bankroll"].iloc[-1]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("初期", f"¥{bankroll:,.0f}")
                    c2.metric("最終", f"¥{final:,.0f}")
                    c3.metric("収益率", f"{(final-bankroll)/bankroll*100:+.1f}%")
                    fig = px.line(h, x="date", y="bankroll", title="資金推移")
                    fig.add_hline(y=bankroll, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("👆 バックテスト結果で資金推移をシミュレーション")

except Exception as e:
    st.error(f"アプリエラー: {e}")
    st.code(traceback.format_exc())
