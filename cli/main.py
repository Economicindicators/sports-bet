"""Typer CLI エントリポイント"""

import logging
from datetime import date

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="スポーツベット ハンデ予測システム")
console = Console()


@app.command()
def init_db():
    """データベースを初期化し、マスタデータを登録する"""
    from database.models import init_db as create_tables
    from database.repository import Repository

    console.print("[bold blue]データベースを初期化中...[/bold blue]")
    create_tables()
    console.print("  テーブル作成完了")

    repo = Repository()
    repo.init_master_data()
    repo.close()
    console.print("  マスタデータ登録完了")
    console.print("[bold green]初期化完了![/bold green]")


@app.command()
def status():
    """データベースの状態を表示する"""
    from sqlalchemy import func
    from database.models import get_session, Sport, League, Team, Player, Match, HandicapData

    session = get_session()
    table = Table(title="データベース状態")
    table.add_column("テーブル", style="cyan")
    table.add_column("件数", justify="right", style="green")

    for model, name in [
        (Sport, "スポーツ"),
        (League, "リーグ"),
        (Team, "チーム"),
        (Player, "選手"),
        (Match, "試合"),
        (HandicapData, "ハンデデータ"),
    ]:
        count = session.query(func.count()).select_from(model).scalar()
        table.add_row(name, str(count))

    console.print(table)
    session.close()


@app.command()
def scrape(
    league: str = typer.Argument(help="リーグコード (npb/mlb/jleague/premier/nba/bleague)"),
    date_str: str = typer.Option(None, "--date", "-d", help="日付 (YYYYMMDD)"),
    no_cache: bool = typer.Option(False, "--no-cache", help="キャッシュを使わない"),
):
    """指定日のデータをスクレイピングする"""
    from scraper.date_utils import parse_date, format_date
    from scraper.manager import ScrapeManager

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    target_date = parse_date(date_str) if date_str else date.today()

    console.print(f"[bold blue]スクレイピング: {league} / {format_date(target_date)}[/bold blue]")
    manager = ScrapeManager(use_cache=not no_cache)
    count = manager.scrape_and_save(league, target_date)
    console.print(f"[bold green]{count} 試合を保存しました[/bold green]")


@app.command()
def backfill(
    league: str = typer.Argument(help="リーグコード"),
    start: str = typer.Option(..., "--start", "-s", help="開始日 (YYYYMMDD)"),
    end: str = typer.Option(None, "--end", "-e", help="終了日 (YYYYMMDD)"),
    no_cache: bool = typer.Option(False, "--no-cache", help="キャッシュを使わない"),
):
    """日付範囲のデータを一括スクレイピングする"""
    from scraper.date_utils import parse_date
    from scraper.manager import ScrapeManager

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start_date = parse_date(start)
    end_date = parse_date(end) if end else date.today()

    console.print(f"[bold blue]バックフィル: {league} / {start_date} ~ {end_date}[/bold blue]")
    manager = ScrapeManager(use_cache=not no_cache)
    total = manager.scrape_range(league, start_date, end_date)
    console.print(f"[bold green]合計 {total} 試合を保存しました[/bold green]")


@app.command()
def train(
    sport: str = typer.Argument(help="スポーツ (baseball/soccer/basketball)"),
    league: str = typer.Option(None, "--league", "-l", help="リーグ指定 (省略=全リーグ)"),
    version: str = typer.Option("v1", "--version", "-v", help="モデルバージョン"),
):
    """モデルを学習する"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from features.feature_pipeline import build_training_data
    from models.training import train_model

    console.print(f"[bold blue]学習: {sport}/{league or 'all'}[/bold blue]")

    df, feature_cols = build_training_data(sport, league)
    if df.empty:
        console.print("[red]データがありません。先にスクレイプしてください。[/red]")
        return

    console.print(f"  {len(df)} 試合, {len(feature_cols)} 特徴量")
    model, metrics = train_model(df, feature_cols, sport, version)

    table = Table(title="学習結果")
    table.add_column("メトリクス", style="cyan")
    table.add_column("値", justify="right", style="green")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)

    # 特徴量重要度
    fi = model.get_feature_importance(top_n=20)
    fi_table = Table(title="特徴量重要度 Top 20")
    fi_table.add_column("特徴量", style="cyan")
    fi_table.add_column("重要度", justify="right", style="green")
    for _, row in fi.iterrows():
        fi_table.add_row(row["feature"], f"{row['importance']:.1f}")
    console.print(fi_table)


@app.command()
def cv(
    sport: str = typer.Argument(help="スポーツ"),
    league: str = typer.Option(None, "--league", "-l", help="リーグ指定"),
    folds: int = typer.Option(5, "--folds", "-f", help="CV分割数"),
):
    """時系列CVを実行する"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from features.feature_pipeline import build_training_data
    from models.training import time_series_cv

    console.print(f"[bold blue]時系列CV: {sport}/{league or 'all'}, {folds} folds[/bold blue]")

    df, feature_cols = build_training_data(sport, league)
    if df.empty:
        console.print("[red]データがありません[/red]")
        return

    results = time_series_cv(df, feature_cols, n_splits=folds)

    table = Table(title="CV結果")
    table.add_column("Fold", style="cyan")
    table.add_column("Train", justify="right")
    table.add_column("Val", justify="right")
    table.add_column("AUC", justify="right", style="green")
    table.add_column("LogLoss", justify="right")
    for r in results:
        table.add_row(
            str(r["fold"]),
            str(r["train_size"]),
            str(r["val_size"]),
            f"{r.get('val_auc', 0):.4f}",
            f"{r.get('val_logloss', 0):.4f}",
        )
    console.print(table)


@app.command()
def backtest(
    sport: str = typer.Argument(help="スポーツ"),
    league: str = typer.Option(None, "--league", "-l", help="リーグ指定"),
    ev_threshold: float = typer.Option(0.05, "--ev", help="EV閾値"),
    periods: int = typer.Option(4, "--periods", "-p", help="Walk-forward期間数"),
    bet_size: int = typer.Option(1000, "--bet-size", help="1ベット金額"),
):
    """Walk-forwardバックテストを実行する"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from features.feature_pipeline import build_training_data
    from betting.backtest import walk_forward_backtest

    console.print(f"[bold blue]バックテスト: {sport}, EV>={ev_threshold}[/bold blue]")

    df, feature_cols = build_training_data(sport, league)
    if df.empty:
        console.print("[red]データがありません[/red]")
        return

    result = walk_forward_backtest(
        df, feature_cols,
        n_periods=periods,
        ev_threshold=ev_threshold,
        bet_size=bet_size,
    )

    # 期間別結果
    table = Table(title="期間別結果")
    table.add_column("Period", style="cyan")
    table.add_column("範囲")
    table.add_column("Bets", justify="right")
    table.add_column("Wins", justify="right")
    table.add_column("ROI", justify="right", style="green")
    table.add_column("PnL", justify="right")

    for pr in result.period_results:
        roi_style = "green" if pr["roi"] > 0 else "red"
        table.add_row(
            str(pr["period"]),
            pr["test_range"],
            str(pr["bets"]),
            str(pr["wins"]),
            f"[{roi_style}]{pr['roi']:+.1f}%[/{roi_style}]",
            f"{pr['pnl']:+,.0f}",
        )
    console.print(table)

    # サマリー
    console.print(f"\n[bold]合計: {result.total_bets} bets, "
                  f"ROI={result.roi:+.1f}%, PnL={result.total_pnl:+,.0f}, "
                  f"MaxDD={result.max_drawdown:,.0f}, Sharpe={result.sharpe:.2f}[/bold]")


@app.command()
def recommend(
    sport: str = typer.Argument(help="スポーツ"),
    date_str: str = typer.Option(None, "--date", "-d", help="日付 (YYYYMMDD)"),
    version: str = typer.Option("v1", "--version", "-v", help="モデルバージョン"),
    ev_threshold: float = typer.Option(0.05, "--ev", help="EV閾値"),
):
    """推奨ベットを表示する"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from features.feature_pipeline import load_matches_df, build_features
    from models.training import load_model
    from betting.handicap_ev import calculate_handicap_ev
    from betting.kelly import calculate_bet_size
    from database.models import get_session, Team

    target_date = date.today()
    if date_str:
        from scraper.date_utils import parse_date
        target_date = parse_date(date_str)

    model = load_model(sport, version)
    df = load_matches_df(sport_code=sport, include_scheduled=True)

    if df.empty:
        console.print("[red]データがありません[/red]")
        return

    df, feature_cols = build_features(df, sport)

    # 予測
    df["pred_prob"] = model.predict_proba(df[feature_cols])
    df["handicap_ev"] = df["pred_prob"].apply(calculate_handicap_ev)

    # 対象日のみフィルタ
    today_df = df[df["date"] == target_date]
    if today_df.empty:
        console.print(f"[yellow]{target_date} の試合データがありません[/yellow]")
        return

    # EV閾値でフィルタ
    bets = today_df[today_df["handicap_ev"] >= ev_threshold].sort_values("handicap_ev", ascending=False)

    session = get_session()
    table = Table(title=f"推奨ベット ({target_date})")
    table.add_column("試合", style="cyan")
    table.add_column("予測確率", justify="right")
    table.add_column("EV", justify="right", style="green")
    table.add_column("Kelly", justify="right")

    for _, row in bets.iterrows():
        home = session.get(Team, int(row["home_team_id"]))
        away = session.get(Team, int(row["away_team_id"]))
        kelly = calculate_bet_size(row["pred_prob"], 100000)
        table.add_row(
            f"{home.name} vs {away.name}",
            f"{row['pred_prob']:.1%}",
            f"{row['handicap_ev']:+.3f}",
            f"¥{kelly:,}",
        )

    console.print(table)
    if bets.empty:
        console.print(f"[yellow]EV>={ev_threshold} のベットはありません[/yellow]")
    session.close()


@app.command()
def models():
    """保存済みモデルの一覧を表示する"""
    from models.registry import list_models

    model_list = list_models()
    if not model_list:
        console.print("[yellow]保存済みモデルはありません[/yellow]")
        return

    table = Table(title="保存済みモデル")
    table.add_column("名前", style="cyan")
    table.add_column("特徴量数", justify="right")
    table.add_column("パス")

    for m in model_list:
        table.add_row(m["name"], str(m.get("n_features", "?")), m["path"])
    console.print(table)


@app.command()
def dashboard():
    """Streamlitダッシュボードを起動する (Phase 5で実装)"""
    console.print("[yellow]dashboard — Phase 5で実装予定[/yellow]")


if __name__ == "__main__":
    app()
