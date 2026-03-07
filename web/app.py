"""Streamlit ダッシュボード — スポーツベット ハンデ予測"""

from __future__ import annotations

import sys
import os
import traceback
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Sports Bet Dashboard",
    page_icon="⚽",
    layout="wide",
)

# デバッグ情報
st.sidebar.caption(f"Python {sys.version}")
st.sidebar.caption(f"Root: {PROJECT_ROOT}")

try:
    import pandas as pd
    import numpy as np
    import plotly.express as px
    from datetime import date, timedelta
    from sqlalchemy import func
    from config.settings import DB_PATH
    st.sidebar.caption(f"DB: {DB_PATH}")
    st.sidebar.caption(f"DB exists: {DB_PATH.exists()}")
    from database.models import get_session, Match, HandicapData, Team
    from config.constants import MIN_EV_THRESHOLD, KELLY_FRACTION
except Exception as e:
    st.error(f"Import error: {e}")
    st.code(traceback.format_exc())
    st.stop()

try:
    sess = get_session()
    total_matches = sess.query(func.count(Match.match_id)).scalar()
    league_counts = dict(
        sess.query(Match.league_code, func.count(Match.match_id))
        .group_by(Match.league_code).all()
    )
    sess.close()
except Exception as e:
    st.error(f"DB接続エラー: {e}")
    st.code(traceback.format_exc())
    st.stop()

# --- サイドバー ---
st.sidebar.title("⚽ Sports Bet")

LEAGUE_OPTIONS = {
    "NPB (野球)": "npb",
    "MLB (野球)": "mlb",
    "プレミアリーグ": "premier",
    "Jリーグ": "jleague",
    "NBA": "nba",
}
LEAGUE_TO_SPORT = {
    "npb": "baseball", "mlb": "baseball",
    "premier": "soccer", "jleague": "soccer",
    "nba": "basketball",
}

selected_league_label = st.sidebar.selectbox(
    "リーグ選択",
    list(LEAGUE_OPTIONS.keys()),
    index=2,  # プレミアリーグをデフォルト
)
selected_league = LEAGUE_OPTIONS[selected_league_label]
selected_sport = LEAGUE_TO_SPORT[selected_league]

ev_threshold = st.sidebar.slider("EV閾値", 0.0, 0.20, MIN_EV_THRESHOLD, 0.01)
bankroll = st.sidebar.number_input("資金 (円)", value=100000, step=10000)

st.sidebar.markdown("---")
for lc, cnt in sorted(league_counts.items(), key=lambda x: -x[1]):
    st.sidebar.caption(f"{lc}: {cnt:,} 試合")


# === ヘルパー関数 ===

@st.cache_data(ttl=300)
def load_match_data(league_code: str) -> pd.DataFrame:
    sess = get_session()
    rows = sess.query(
        Match.match_id, Match.date, Match.home_team_id, Match.away_team_id,
        Match.home_score, Match.away_score, Match.venue,
        HandicapData.handicap_team_id, HandicapData.handicap_value,
        HandicapData.result_type, HandicapData.payout_rate,
    ).join(HandicapData, Match.match_id == HandicapData.match_id
    ).filter(Match.league_code == league_code, Match.status == "finished"
    ).order_by(Match.date).all()
    sess.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "match_id", "date", "home_team_id", "away_team_id",
        "home_score", "away_score", "venue",
        "handicap_team_id", "handicap_value", "result_type", "payout_rate",
    ])
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_team_names(league_code: str) -> dict:
    sess = get_session()
    teams = sess.query(Team.team_id, Team.name).filter(Team.league_code == league_code).all()
    sess.close()
    return {t[0]: t[1] for t in teams}


@st.cache_data(ttl=600)
def run_backtest(league_code: str, sport_code: str, ev_thresh: float):
    import warnings
    warnings.filterwarnings("ignore")
    from features.feature_pipeline import build_training_data
    from betting.backtest import walk_forward_backtest
    df, feat_cols = build_training_data(sport_code, league_code)
    if df.empty:
        return None, None, None
    n_periods = 4 if len(df) > 500 else 3
    result = walk_forward_backtest(df, feat_cols, n_periods=n_periods, ev_threshold=ev_thresh)
    return result, df, feat_cols


@st.cache_data(ttl=600)
def get_feature_importance(sport_code: str, league_code: str):
    import warnings
    warnings.filterwarnings("ignore")
    from features.feature_pipeline import build_training_data
    from models.lgbm_model import LightGBMModel
    df, feat_cols = build_training_data(sport_code, league_code)
    if df.empty or len(feat_cols) == 0:
        return pd.DataFrame()
    model = LightGBMModel()
    split = int(len(df) * 0.8)
    model.train(df.iloc[:split][feat_cols], df.iloc[:split]["target"],
                df.iloc[split:][feat_cols], df.iloc[split:]["target"])
    return model.get_feature_importance(top_n=30)


# --- メインコンテンツ ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 試合データ", "📊 バックテスト", "🏆 チーム分析",
    "🔬 モデル", "💰 資金管理",
])


# =============================================
# Tab 1: 試合データ (自動表示)
# =============================================
with tab1:
    try:
        df = load_match_data(selected_league)
        if df.empty:
            st.warning("データがありません")
        else:
            team_names = load_team_names(selected_league)

            # --- サマリーカード ---
            n_matches = len(df)
            fav_rate = (df["payout_rate"] >= 1.5).mean()
            avg_payout = df["payout_rate"].mean()
            date_min = df["date"].min().strftime("%Y-%m-%d")
            date_max = df["date"].max().strftime("%Y-%m-%d")

            st.header(f"{selected_league_label}")
            st.caption(f"{date_min} 〜 {date_max}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("試合数", f"{n_matches:,}")
            c2.metric("ハンデ有利率", f"{fav_rate:.1%}")
            c3.metric("平均ペイアウト", f"{avg_payout:.2f}x")
            c4.metric("期間", f"{(df['date'].max() - df['date'].min()).days}日")

            # --- 結果分布 (自動表示) ---
            col1, col2 = st.columns(2)
            with col1:
                result_counts = df["result_type"].value_counts()
                fig = px.bar(
                    x=result_counts.index, y=result_counts.values,
                    title="ハンデ結果分布",
                    labels={"x": "結果", "y": "件数"},
                    color=result_counts.index,
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # 月別ペイアウト推移
                monthly = df.set_index("date").resample("M")["payout_rate"].mean().reset_index()
                monthly.columns = ["月", "平均ペイアウト"]
                fig2 = px.line(
                    monthly, x="月", y="平均ペイアウト",
                    title="月別平均ペイアウト推移",
                )
                fig2.add_hline(y=1.0, line_dash="dash", line_color="gray",
                               annotation_text="損益分岐")
                st.plotly_chart(fig2, use_container_width=True)

            # --- 直近の試合テーブル ---
            st.subheader("直近の試合")
            recent = df.tail(50).copy()
            recent["ホーム"] = recent["home_team_id"].map(team_names)
            recent["アウェイ"] = recent["away_team_id"].map(team_names)
            recent["ハンデチーム"] = recent["handicap_team_id"].map(team_names)
            recent["スコア"] = recent["home_score"].astype(str) + "-" + recent["away_score"].astype(str)
            display = recent[["date", "ホーム", "アウェイ", "スコア", "ハンデチーム",
                              "handicap_value", "result_type", "payout_rate"]].copy()
            display.columns = ["日付", "ホーム", "アウェイ", "スコア", "ハンデチーム",
                               "ハンデ値", "結果", "ペイアウト"]
            st.dataframe(
                display.sort_values("日付", ascending=False),
                use_container_width=True, height=400,
            )

    except Exception as e:
        st.error(f"エラー: {e}")


# =============================================
# Tab 2: バックテスト
# =============================================
with tab2:
    st.header("Walk-Forward バックテスト")
    st.caption(f"サイドバーでリーグ・EV閾値を選択 → 実行ボタン")

    if st.button("▶ バックテスト実行", key="run_bt", type="primary"):
        try:
            with st.spinner("バックテスト実行中... (初回は数分かかります)"):
                result, bt_df, feat_cols = run_backtest(
                    selected_league, selected_sport, ev_threshold
                )

            if result is None or result.total_bets == 0:
                st.warning("データが不足しています")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("総ベット", f"{result.total_bets}")
                c2.metric("ROI", f"{result.roi:+.1f}%",
                           delta=f"{'プラス' if result.roi > 0 else 'マイナス'}")
                c3.metric("PnL", f"¥{result.total_pnl:+,.0f}")
                c4.metric("Sharpe", f"{result.sharpe:.2f}")

                c5, c6, c7, c8 = st.columns(4)
                wr = f"{result.wins/result.total_bets*100:.1f}%" if result.total_bets else "N/A"
                c5.metric("勝率", wr)
                c6.metric("MaxDD", f"¥{result.max_drawdown:,.0f}")
                c7.metric("勝ちベット", f"{result.wins}")
                c8.metric("EV閾値", f"{ev_threshold:.2f}")

                st.subheader("期間別")
                st.dataframe(pd.DataFrame(result.period_results), use_container_width=True)

                if result.bet_history:
                    hist = pd.DataFrame(result.bet_history)
                    hist["cumulative_pnl"] = hist["pnl"].cumsum()
                    hist["date"] = pd.to_datetime(hist["date"])

                    fig = px.line(hist, x="date", y="cumulative_pnl",
                                  title="累積PnL推移",
                                  labels={"date": "日付", "cumulative_pnl": "PnL (円)"})
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig, use_container_width=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        fig2 = px.histogram(hist, x="pnl", nbins=30, title="PnL分布",
                                            color_discrete_sequence=["#636EFA"])
                        st.plotly_chart(fig2, use_container_width=True)
                    with col2:
                        hist["peak"] = hist["cumulative_pnl"].cummax()
                        hist["dd"] = hist["peak"] - hist["cumulative_pnl"]
                        fig3 = px.area(hist, x="date", y="dd", title="ドローダウン",
                                       color_discrete_sequence=["#EF553B"])
                        st.plotly_chart(fig3, use_container_width=True)

        except Exception as e:
            st.error(f"エラー: {e}")
    else:
        st.info("👆 ボタンを押すとバックテストが始まります。初回は2-3分かかります。")


# =============================================
# Tab 3: チーム分析
# =============================================
with tab3:
    try:
        df = load_match_data(selected_league)
        if df.empty:
            st.warning("データがありません")
        else:
            team_names = load_team_names(selected_league)
            team_list = sorted(team_names.values())

            st.header("チーム分析")
            sel_team = st.selectbox("チーム選択", team_list)
            team_id = next((k for k, v in team_names.items() if v == sel_team), None)

            if team_id:
                handi_df = df[df["handicap_team_id"] == team_id].copy()
                all_df = df[(df["home_team_id"] == team_id) | (df["away_team_id"] == team_id)]

                c1, c2, c3 = st.columns(3)
                c1.metric("総試合数", f"{len(all_df)}")
                c2.metric("ハンデ有利回数", f"{len(handi_df)}")
                if len(handi_df) > 0:
                    fr = (handi_df["payout_rate"] >= 1.5).mean()
                    c3.metric("ハンデ勝率", f"{fr:.1%}")

                    handi_df = handi_df.sort_values("date")
                    handi_df["cum_pnl"] = (handi_df["payout_rate"] - 1.0).cumsum() * 1000

                    fig = px.line(handi_df, x="date", y="cum_pnl",
                                  title=f"{sel_team} — ハンデ累積損益 (1000円/bet)",
                                  labels={"date": "日付", "cum_pnl": "累積PnL (円)"})
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig, use_container_width=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        rc = handi_df["result_type"].value_counts()
                        fig2 = px.pie(values=rc.values, names=rc.index,
                                      title="結果内訳")
                        st.plotly_chart(fig2, use_container_width=True)
                    with col2:
                        # 直近10試合
                        st.subheader("直近10試合")
                        last10 = handi_df.tail(10).copy()
                        last10["対戦相手"] = last10.apply(
                            lambda r: team_names.get(r["away_team_id"], "?")
                            if r["home_team_id"] == team_id
                            else team_names.get(r["home_team_id"], "?"), axis=1
                        )
                        last10["H/A"] = last10["home_team_id"].apply(
                            lambda x: "H" if x == team_id else "A"
                        )
                        disp = last10[["date", "対戦相手", "H/A", "handicap_value",
                                       "result_type", "payout_rate"]].copy()
                        disp.columns = ["日付", "対戦相手", "H/A", "ハンデ", "結果", "ペイアウト"]
                        st.dataframe(disp.sort_values("日付", ascending=False),
                                     use_container_width=True, hide_index=True)
                else:
                    st.info("ハンデ有利に指定された試合がありません")
    except Exception as e:
        st.error(f"エラー: {e}")


# =============================================
# Tab 4: モデル性能
# =============================================
with tab4:
    st.header("モデル性能")
    if st.button("▶ 特徴量重要度を表示", key="run_fi", type="primary"):
        try:
            with st.spinner("モデル学習中..."):
                fi_df = get_feature_importance(selected_sport, selected_league)
            if fi_df.empty:
                st.warning("データ不足")
            else:
                fig = px.bar(fi_df.sort_values("importance"), x="importance", y="feature",
                             orientation="h", title="特徴量重要度 (Top 30)",
                             labels={"importance": "重要度 (gain)", "feature": "特徴量"},
                             height=700)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"エラー: {e}")
    else:
        st.info("👆 ボタンを押すとモデルを学習して特徴量重要度を表示します")


# =============================================
# Tab 5: 資金管理
# =============================================
with tab5:
    st.header("資金管理シミュレーション")
    c1, c2 = st.columns(2)
    with c1:
        bet_strategy = st.selectbox("ベット戦略", ["固定額", "Kelly基準"])
    with c2:
        fixed_bet = st.number_input("固定ベット額 (円)", value=1000, step=100)

    if st.button("▶ シミュレーション実行", key="run_sim", type="primary"):
        try:
            with st.spinner("シミュレーション中..."):
                result, _, _ = run_backtest(selected_league, selected_sport, ev_threshold)
            if result is None or not result.bet_history:
                st.warning("データ不足")
            else:
                hist = pd.DataFrame(result.bet_history)
                hist["date"] = pd.to_datetime(hist["date"])

                if bet_strategy == "Kelly基準":
                    from betting.kelly import kelly_fraction as kf
                    from betting.handicap_ev import AVG_FAVORABLE_PAYOUT as avg_pay
                    bal = float(bankroll)
                    bals = [bal]
                    for _, r in hist.iterrows():
                        k = kf(r["pred_prob"], avg_pay, KELLY_FRACTION)
                        amt = bal * k
                        p = r["payout"]
                        if p is not None and not pd.isna(p):
                            bal += amt * (p - 1.0)
                        bals.append(bal)
                    hist["bankroll"] = bals[1:]
                else:
                    hist["bankroll"] = bankroll + (hist["pnl"] / 1000 * fixed_bet).cumsum()

                final = hist["bankroll"].iloc[-1]
                ret = (final - bankroll) / bankroll * 100

                c1, c2, c3 = st.columns(3)
                c1.metric("初期資金", f"¥{bankroll:,.0f}")
                c2.metric("最終資金", f"¥{final:,.0f}")
                c3.metric("収益率", f"{ret:+.1f}%")

                fig = px.line(hist, x="date", y="bankroll", title="資金推移",
                              labels={"date": "日付", "bankroll": "資金 (円)"})
                fig.add_hline(y=bankroll, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"エラー: {e}")
    else:
        st.info("👆 バックテスト結果をもとに資金推移をシミュレーションします")
