"""Streamlit ダッシュボード — スポーツベット ハンデ予測"""

from __future__ import annotations

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from sqlalchemy import func

from database.models import get_session, Match, HandicapData, Team, League, Sport
from config.constants import MIN_EV_THRESHOLD, KELLY_FRACTION

st.set_page_config(
    page_title="Sports Bet Dashboard",
    page_icon="⚽",
    layout="wide",
)

st.title("スポーツベット ハンデ予測")

# --- サイドバー ---
st.sidebar.header("設定")

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

selected_league_label = st.sidebar.selectbox("リーグ", list(LEAGUE_OPTIONS.keys()))
selected_league = LEAGUE_OPTIONS[selected_league_label]
selected_sport = LEAGUE_TO_SPORT[selected_league]

ev_threshold = st.sidebar.slider("EV閾値", 0.0, 0.20, MIN_EV_THRESHOLD, 0.01)
bankroll = st.sidebar.number_input("資金 (円)", value=100000, step=10000)

# --- タブ ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 バックテスト", "📋 過去の結果", "🏆 チーム分析",
    "🔬 モデル性能", "💰 資金管理",
])


# === ヘルパー関数 ===

@st.cache_data(ttl=300)
def load_match_data(league_code: str) -> pd.DataFrame:
    """DBからリーグの試合+ハンデデータを読み込む"""
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
        "handicap_team_id", "handicap_value",
        "result_type", "payout_rate",
    ])
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_team_names(league_code: str) -> dict:
    """チームID → 名前のマッピング"""
    sess = get_session()
    teams = sess.query(Team.team_id, Team.name).filter(Team.league_code == league_code).all()
    sess.close()
    return {t[0]: t[1] for t in teams}


@st.cache_data(ttl=600)
def run_backtest(league_code: str, sport_code: str, ev_thresh: float):
    """バックテストを実行"""
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
    """特徴量重要度を取得"""
    import warnings
    warnings.filterwarnings("ignore")

    from features.feature_pipeline import build_training_data
    from models.lgbm_model import LightGBMModel

    df, feat_cols = build_training_data(sport_code, league_code)
    if df.empty or len(feat_cols) == 0:
        return pd.DataFrame()

    model = LightGBMModel()
    split = int(len(df) * 0.8)
    X_train = df.iloc[:split][feat_cols]
    y_train = df.iloc[:split]["target"]
    X_val = df.iloc[split:][feat_cols]
    y_val = df.iloc[split:]["target"]
    model.train(X_train, y_train, X_val, y_val)

    return model.get_feature_importance(top_n=30)


# === Tab 1: バックテスト ===
with tab1:
    st.header("Walk-Forward バックテスト")

    if st.button("バックテスト実行", key="run_bt"):
        with st.spinner("バックテスト実行中..."):
            result, bt_df, feat_cols = run_backtest(selected_league, selected_sport, ev_threshold)

        if result is None or result.total_bets == 0:
            st.warning("データが不足しています")
        else:
            # サマリー
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("総ベット数", f"{result.total_bets}")
            col2.metric("ROI", f"{result.roi:+.1f}%")
            col3.metric("PnL", f"¥{result.total_pnl:+,.0f}")
            col4.metric("Sharpe", f"{result.sharpe:.2f}")

            col5, col6, col7, col8 = st.columns(4)
            col5.metric("勝率", f"{result.wins/result.total_bets*100:.1f}%" if result.total_bets > 0 else "N/A")
            col6.metric("MaxDD", f"¥{result.max_drawdown:,.0f}")
            col7.metric("勝ちベット", f"{result.wins}")
            col8.metric("EV閾値", f"{ev_threshold:.2f}")

            # 期間別結果
            st.subheader("期間別結果")
            period_df = pd.DataFrame(result.period_results)
            st.dataframe(period_df, use_container_width=True)

            # 累積PnLチャート
            if result.bet_history:
                st.subheader("累積PnL推移")
                hist_df = pd.DataFrame(result.bet_history)
                hist_df["cumulative_pnl"] = hist_df["pnl"].cumsum()
                hist_df["date"] = pd.to_datetime(hist_df["date"])

                fig = px.line(
                    hist_df, x="date", y="cumulative_pnl",
                    title="累積損益 (PnL)",
                    labels={"date": "日付", "cumulative_pnl": "累積PnL (円)"},
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)

                # PnL分布
                st.subheader("ベット別PnL分布")
                fig2 = px.histogram(
                    hist_df, x="pnl", nbins=30,
                    title="PnL分布",
                    labels={"pnl": "PnL (円)"},
                    color_discrete_sequence=["#636EFA"],
                )
                st.plotly_chart(fig2, use_container_width=True)

                # ドローダウン
                st.subheader("ドローダウン推移")
                hist_df["peak"] = hist_df["cumulative_pnl"].cummax()
                hist_df["drawdown"] = hist_df["peak"] - hist_df["cumulative_pnl"]
                fig3 = px.area(
                    hist_df, x="date", y="drawdown",
                    title="ドローダウン",
                    labels={"date": "日付", "drawdown": "ドローダウン (円)"},
                    color_discrete_sequence=["#EF553B"],
                )
                st.plotly_chart(fig3, use_container_width=True)


# === Tab 2: 過去の結果 ===
with tab2:
    st.header("過去の結果")

    df = load_match_data(selected_league)
    if df.empty:
        st.warning("データがありません")
    else:
        team_names = load_team_names(selected_league)

        # フィルタ
        col1, col2 = st.columns(2)
        with col1:
            min_date = df["date"].min().date()
            max_date = df["date"].max().date()
            date_range = st.date_input(
                "期間", value=(max_date - timedelta(days=30), max_date),
                min_value=min_date, max_value=max_date,
            )
        with col2:
            result_filter = st.multiselect(
                "結果フィルタ",
                options=df["result_type"].dropna().unique().tolist(),
                default=[],
            )

        # フィルタ適用
        if len(date_range) == 2:
            mask = (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
            filtered = df[mask].copy()
        else:
            filtered = df.copy()

        if result_filter:
            filtered = filtered[filtered["result_type"].isin(result_filter)]

        # 表示用カラム追加
        filtered["ホーム"] = filtered["home_team_id"].map(team_names)
        filtered["アウェイ"] = filtered["away_team_id"].map(team_names)
        filtered["ハンデチーム"] = filtered["handicap_team_id"].map(team_names)
        filtered["スコア"] = filtered["home_score"].astype(str) + "-" + filtered["away_score"].astype(str)

        display_cols = ["date", "ホーム", "アウェイ", "スコア", "ハンデチーム",
                        "handicap_value", "result_type", "payout_rate"]
        display_df = filtered[display_cols].sort_values("date", ascending=False)
        display_df.columns = ["日付", "ホーム", "アウェイ", "スコア", "ハンデチーム",
                              "ハンデ値", "結果", "ペイアウト"]

        st.dataframe(display_df, use_container_width=True, height=500)

        # 統計
        st.subheader("結果分布")
        col1, col2 = st.columns(2)
        with col1:
            result_counts = filtered["result_type"].value_counts()
            fig = px.bar(
                x=result_counts.index, y=result_counts.values,
                title="結果タイプ別件数",
                labels={"x": "結果", "y": "件数"},
                color_discrete_sequence=["#636EFA"],
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            payout_dist = filtered["payout_rate"].value_counts().sort_index()
            fig2 = px.bar(
                x=payout_dist.index.astype(str), y=payout_dist.values,
                title="ペイアウト分布",
                labels={"x": "ペイアウト", "y": "件数"},
                color_discrete_sequence=["#00CC96"],
            )
            st.plotly_chart(fig2, use_container_width=True)


# === Tab 3: チーム分析 ===
with tab3:
    st.header("チーム分析")

    df = load_match_data(selected_league)
    if df.empty:
        st.warning("データがありません")
    else:
        team_names = load_team_names(selected_league)
        team_list = sorted(team_names.values())

        selected_team_name = st.selectbox("チーム選択", team_list)
        team_id = next((k for k, v in team_names.items() if v == selected_team_name), None)

        if team_id:
            # ハンデチームとしての成績
            handi_df = df[df["handicap_team_id"] == team_id].copy()
            home_df = df[df["home_team_id"] == team_id].copy()
            away_df = df[df["away_team_id"] == team_id].copy()

            col1, col2, col3 = st.columns(3)
            col1.metric("総試合数", f"{len(home_df) + len(away_df)}")
            col2.metric("ハンデ有利回数", f"{len(handi_df)}")

            if len(handi_df) > 0:
                fav_rate = (handi_df["payout_rate"] >= 1.5).mean()
                col3.metric("ハンデ勝率", f"{fav_rate:.1%}")

                # ハンデ結果推移
                st.subheader(f"{selected_team_name} のハンデ結果推移")
                handi_df = handi_df.sort_values("date")
                handi_df["cum_pnl"] = (handi_df["payout_rate"] - 1.0).cumsum() * 1000
                handi_df["ホーム"] = handi_df["home_team_id"].map(team_names)
                handi_df["アウェイ"] = handi_df["away_team_id"].map(team_names)

                fig = px.line(
                    handi_df, x="date", y="cum_pnl",
                    title=f"{selected_team_name} ハンデ累積損益",
                    labels={"date": "日付", "cum_pnl": "累積PnL (1000円ベット)"},
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)

                # 結果内訳
                st.subheader("結果内訳")
                result_cnt = handi_df["result_type"].value_counts()
                fig2 = px.pie(
                    values=result_cnt.values, names=result_cnt.index,
                    title=f"{selected_team_name} ハンデ結果分布",
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("このチームはハンデ有利チームに指定された試合がありません")


# === Tab 4: モデル性能 ===
with tab4:
    st.header("モデル性能")

    if st.button("特徴量重要度を表示", key="run_fi"):
        with st.spinner("モデル学習中..."):
            fi_df = get_feature_importance(selected_sport, selected_league)

        if fi_df.empty:
            st.warning("データが不足しています")
        else:
            fig = px.bar(
                fi_df.sort_values("importance"),
                x="importance", y="feature",
                orientation="h",
                title="特徴量重要度 (Top 30)",
                labels={"importance": "重要度 (gain)", "feature": "特徴量"},
                height=800,
            )
            st.plotly_chart(fig, use_container_width=True)

    # AUC / CV結果
    st.subheader("時系列CV結果")
    if st.button("CV実行", key="run_cv"):
        with st.spinner("CV実行中..."):
            import warnings
            warnings.filterwarnings("ignore")
            from features.feature_pipeline import build_training_data
            from models.lgbm_model import LightGBMModel
            from sklearn.metrics import roc_auc_score, log_loss

            df, feat_cols = build_training_data(selected_sport, selected_league)
            if df.empty:
                st.warning("データ不足")
            else:
                df = df.sort_values("date").reset_index(drop=True)
                dates = df["date"].unique()
                n_dates = len(dates)
                n_folds = 5
                fold_size = n_dates // (n_folds + 1)

                cv_results = []
                for fold in range(n_folds):
                    train_end_idx = (fold + 1) * fold_size
                    train_end = dates[min(train_end_idx, n_dates - 1)]
                    gap = train_end + pd.Timedelta(days=30)

                    if fold < n_folds - 1:
                        test_end_idx = (fold + 2) * fold_size
                        test_end = dates[min(test_end_idx, n_dates - 1)]
                    else:
                        test_end = dates[-1]

                    train_mask = df["date"] <= train_end
                    test_mask = (df["date"] > gap) & (df["date"] <= test_end)

                    X_tr = df.loc[train_mask, feat_cols]
                    y_tr = df.loc[train_mask, "target"]
                    X_te = df.loc[test_mask, feat_cols]
                    y_te = df.loc[test_mask, "target"]

                    if len(X_tr) < 50 or len(X_te) < 10:
                        continue

                    model = LightGBMModel()
                    sp = int(len(X_tr) * 0.8)
                    model.train(X_tr.iloc[:sp], y_tr.iloc[:sp], X_tr.iloc[sp:], y_tr.iloc[sp:])
                    preds = model.predict_proba(X_te)
                    auc = roc_auc_score(y_te, preds)
                    ll = log_loss(y_te, preds)
                    cv_results.append({"Fold": fold, "AUC": auc, "LogLoss": ll, "N_test": len(X_te)})

                if cv_results:
                    cv_df = pd.DataFrame(cv_results)
                    st.dataframe(cv_df, use_container_width=True)
                    st.metric("平均AUC", f"{cv_df['AUC'].mean():.4f}")
                    st.metric("平均LogLoss", f"{cv_df['LogLoss'].mean():.4f}")


# === Tab 5: 資金管理 ===
with tab5:
    st.header("資金管理シミュレーション")

    col1, col2 = st.columns(2)
    with col1:
        bet_strategy = st.selectbox("ベット戦略", ["固定額", "Kelly基準"])
    with col2:
        fixed_bet = st.number_input("固定ベット額 (円)", value=1000, step=100)

    if st.button("シミュレーション実行", key="run_sim"):
        with st.spinner("シミュレーション中..."):
            result, bt_df, feat_cols = run_backtest(selected_league, selected_sport, ev_threshold)

        if result is None or not result.bet_history:
            st.warning("データが不足しています")
        else:
            hist_df = pd.DataFrame(result.bet_history)
            hist_df["date"] = pd.to_datetime(hist_df["date"])

            # Kelly or 固定
            if bet_strategy == "Kelly基準":
                from betting.kelly import kelly_fraction as kf
                from betting.handicap_ev import AVG_FAVORABLE_PAYOUT as avg_pay
                current_bankroll = float(bankroll)
                bankroll_history = [current_bankroll]
                for _, row in hist_df.iterrows():
                    k = kf(row["pred_prob"], avg_pay, KELLY_FRACTION)
                    bet_amt = current_bankroll * k
                    payout = row["payout"]
                    if payout is not None and not pd.isna(payout):
                        pnl = bet_amt * (payout - 1.0)
                        current_bankroll += pnl
                    bankroll_history.append(current_bankroll)
                hist_df["bankroll"] = bankroll_history[1:]
            else:
                hist_df["bankroll"] = bankroll + (hist_df["pnl"] / 1000 * fixed_bet).cumsum()

            # メトリクス
            final_bankroll = hist_df["bankroll"].iloc[-1]
            total_return = (final_bankroll - bankroll) / bankroll * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("初期資金", f"¥{bankroll:,.0f}")
            col2.metric("最終資金", f"¥{final_bankroll:,.0f}")
            col3.metric("収益率", f"{total_return:+.1f}%")

            # 資金推移
            fig = px.line(
                hist_df, x="date", y="bankroll",
                title="資金推移",
                labels={"date": "日付", "bankroll": "資金 (円)"},
            )
            fig.add_hline(y=bankroll, line_dash="dash", line_color="gray",
                          annotation_text="初期資金")
            st.plotly_chart(fig, use_container_width=True)

            # ベット履歴テーブル
            st.subheader("ベット履歴")
            display = hist_df[["date", "pred_prob", "ev", "payout", "pnl", "bankroll"]].copy()
            display.columns = ["日付", "予測確率", "EV", "ペイアウト", "PnL", "資金"]
            st.dataframe(display.sort_values("日付", ascending=False), use_container_width=True, height=400)


# --- フッター ---
st.sidebar.markdown("---")
sess = get_session()
total_matches = sess.query(func.count(Match.match_id)).scalar()
sess.close()
st.sidebar.caption(f"DB: {total_matches:,} 試合")
