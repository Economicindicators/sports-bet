"""Sports Bet 使用マニュアル PDF 生成スクリプト"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fpdf import FPDF

# --- Fonts ---
FONT_REGULAR = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
FONT_BOLD = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
FONT_LIGHT = "/System/Library/Fonts/ヒラギノ角ゴシック W2.ttc"

# --- Colors ---
C_BG = (14, 17, 23)
C_CARD = (26, 31, 46)
C_BORDER = (45, 53, 72)
C_TEXT = (250, 250, 250)
C_MUTED = (136, 146, 164)
C_ACCENT = (0, 212, 170)
C_WARN = (255, 209, 102)
C_DANGER = (239, 71, 111)
C_WHITE = (255, 255, 255)

OUTPUT_PATH = PROJECT_ROOT / "docs" / "sports_bet_manual.pdf"


class ManualPDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.add_font("jp", "", FONT_REGULAR)
        self.add_font("jpb", "", FONT_BOLD)
        self.add_font("jpl", "", FONT_LIGHT)
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(*C_BG)
        self.rect(0, 0, 210, 12, "F")
        self.set_font("jpl", size=7)
        self.set_text_color(*C_MUTED)
        self.set_y(4)
        self.cell(0, 5, "Sports Bet Handicap Prediction — User Manual", align="C")

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font("jpl", size=7)
        self.set_text_color(*C_MUTED)
        self.cell(0, 10, f"— {self.page_no()} —", align="C")

    # --- Drawing helpers ---

    def dark_bg(self):
        self.set_fill_color(*C_BG)
        self.rect(0, 0, 210, 297, "F")

    def section_title(self, text: str, icon: str = ""):
        self.set_font("jpb", size=16)
        self.set_text_color(*C_ACCENT)
        self.ln(6)
        display = f"{icon}  {text}" if icon else text
        self.cell(0, 10, display, ln=True)
        # Accent underline
        x = self.get_x() + 10
        y = self.get_y()
        self.set_draw_color(*C_ACCENT)
        self.set_line_width(0.6)
        self.line(x, y, x + 50, y)
        self.ln(4)

    def sub_title(self, text: str):
        self.set_font("jpb", size=12)
        self.set_text_color(*C_TEXT)
        self.ln(3)
        self.cell(0, 8, text, ln=True)
        self.ln(1)

    def body_text(self, text: str):
        self.set_font("jp", size=9.5)
        self.set_text_color(*C_TEXT)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def muted_text(self, text: str):
        self.set_font("jp", size=8.5)
        self.set_text_color(*C_MUTED)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def code_block(self, text: str):
        self.set_fill_color(30, 36, 52)
        self.set_font("jp", size=9)
        self.set_text_color(200, 210, 230)
        x = self.get_x() + 10
        w = 170
        lines = text.strip().split("\n")
        h = len(lines) * 5.5 + 8
        y = self.get_y()
        self.set_draw_color(*C_BORDER)
        self.set_line_width(0.3)
        self.rect(x, y, w, h, "FD")
        self.set_xy(x + 6, y + 4)
        for line in lines:
            self.cell(0, 5.5, line, ln=True)
            self.set_x(x + 6)
        self.ln(3)

    def card(self, color: tuple, title: str, content: str):
        x = self.get_x() + 10
        y = self.get_y()
        w = 170
        lines = content.split("\n")
        h = 10 + len(lines) * 5.5 + 8
        # Card background
        self.set_fill_color(*C_CARD)
        self.set_draw_color(*C_BORDER)
        self.set_line_width(0.3)
        self.rect(x, y, w, h, "FD")
        # Left accent bar
        self.set_fill_color(*color)
        self.rect(x, y, 3, h, "F")
        # Title
        self.set_xy(x + 8, y + 4)
        self.set_font("jpb", size=10)
        self.set_text_color(*color)
        self.cell(0, 6, title, ln=True)
        # Content
        self.set_x(x + 8)
        self.set_font("jp", size=9)
        self.set_text_color(*C_TEXT)
        for line in lines:
            self.cell(0, 5.5, line, ln=True)
            self.set_x(x + 8)
        self.set_y(y + h + 4)

    def table_row(self, cells: list, widths: list, bold: bool = False,
                  colors: list | None = None, fill: tuple | None = None):
        x_start = self.get_x() + 10
        x = x_start
        h = 7
        if fill:
            self.set_fill_color(*fill)
            self.rect(x_start, self.get_y(), sum(widths), h, "F")
        self.set_draw_color(*C_BORDER)
        self.set_line_width(0.15)
        self.line(x_start, self.get_y() + h, x_start + sum(widths), self.get_y() + h)

        for i, (cell, w) in enumerate(zip(cells, widths)):
            self.set_xy(x + 2, self.get_y())
            if bold:
                self.set_font("jpb", size=8)
            else:
                self.set_font("jp", size=9)
            if colors and i < len(colors) and colors[i]:
                self.set_text_color(*colors[i])
            elif bold:
                self.set_text_color(*C_MUTED)
            else:
                self.set_text_color(*C_TEXT)
            self.cell(w - 4, h, str(cell))
            x += w
        self.ln(h)


def build_pdf():
    pdf = ManualPDF()

    # ===== Cover Page =====
    pdf.add_page()
    pdf.dark_bg()

    # Center logo area
    pdf.ln(60)
    pdf.set_font("jpb", size=48)
    pdf.set_text_color(*C_ACCENT)
    pdf.cell(0, 20, "Sports Bet", align="C", ln=True)

    pdf.set_font("jpl", size=14)
    pdf.set_text_color(*C_MUTED)
    pdf.cell(0, 10, "Handicap Prediction System", align="C", ln=True)

    pdf.ln(5)
    # Accent line
    pdf.set_draw_color(*C_ACCENT)
    pdf.set_line_width(1)
    pdf.line(70, pdf.get_y(), 140, pdf.get_y())
    pdf.ln(10)

    pdf.set_font("jp", size=11)
    pdf.set_text_color(*C_TEXT)
    pdf.cell(0, 8, "User Manual", align="C", ln=True)

    pdf.ln(50)
    pdf.set_font("jpl", size=9)
    pdf.set_text_color(*C_MUTED)
    pdf.cell(0, 6, "LightGBM x Walk-Forward Backtest x Kelly Criterion", align="C", ln=True)
    pdf.cell(0, 6, "NPB / MLB / J League / Premier League / NBA", align="C", ln=True)

    # ===== Page 2: TOC =====
    pdf.add_page()
    pdf.dark_bg()
    pdf.section_title("Table of Contents", "")

    toc = [
        ("1", "System Overview", "システム概要"),
        ("2", "Dashboard Guide", "ダッシュボード操作ガイド"),
        ("3", "Handicap Payout Table", "9段階ハンデペイアウト表"),
        ("4", "Expected Value (EV)", "期待値の計算"),
        ("5", "Kelly Criterion", "Kelly基準（ベットサイジング）"),
        ("6", "Bet Signals", "ベット判定ロジック"),
        ("7", "Backtest & Validation", "バックテスト & 検証"),
        ("8", "CLI Commands", "コマンドラインリファレンス"),
        ("9", "Glossary", "用語集"),
    ]
    for num, en, ja in toc:
        pdf.set_font("jpb", size=11)
        pdf.set_text_color(*C_ACCENT)
        pdf.cell(12, 8, num)
        pdf.set_font("jpb", size=11)
        pdf.set_text_color(*C_TEXT)
        pdf.cell(80, 8, en)
        pdf.set_font("jp", size=9)
        pdf.set_text_color(*C_MUTED)
        pdf.cell(0, 8, ja, ln=True)

    # ===== Page 3: System Overview =====
    pdf.add_page()
    pdf.dark_bg()
    pdf.section_title("1. System Overview", "")

    pdf.body_text(
        "Sports Betは、日本のスポーツハンデサイト3つからデータを収集し、"
        "LightGBMで勝敗予測を行い、期待値ベースのベット推奨を生成するシステムです。"
    )

    pdf.sub_title("Architecture")
    pdf.code_block(
        "scraper/   Data collection from 3 sites\n"
        "  -> database/   SQLite via SQLAlchemy\n"
        "    -> features/   41 features per match\n"
        "      -> models/   LightGBM binary classifier\n"
        "        -> betting/   EV, Kelly, Backtest\n"
        "          -> web/   Streamlit Dashboard\n"
        "          -> cli/   Typer CLI"
    )

    pdf.sub_title("Data Sources")
    headers = ["Site", "URL", "Sports"]
    widths = [40, 65, 65]
    pdf.table_row(headers, widths, bold=True, fill=C_CARD)
    data = [
        ("ハンデの森", "handenomori.com", "NPB, MLB"),
        ("ハンデの嵐", "footbool-hande.com", "J League, Premier, etc."),
        ("ハンディマン", "b-handicap.com", "NBA, B League"),
    ]
    for row in data:
        pdf.table_row(list(row), widths)

    pdf.ln(4)
    pdf.sub_title("Supported Leagues")
    pdf.body_text(
        "Baseball: NPB（日本プロ野球）、MLB（メジャーリーグ）\n"
        "Soccer: Jリーグ、Premier League、La Liga、Serie A、Bundesliga\n"
        "Basketball: NBA、Bリーグ"
    )

    pdf.sub_title("Model")
    pdf.body_text(
        "Algorithm: LightGBM (binary classifier)\n"
        "Target: ハンデ有利チームが5分勝ち以上 (payout >= 1.5)\n"
        "Features: 41個（ハンデ系5, チーム成績12, H2H 4, 順位5, 日程6, スポーツ固有）\n"
        "Validation: 時系列CV (expanding window + 30日gap)"
    )

    # ===== Page 4: Dashboard Guide =====
    pdf.add_page()
    pdf.dark_bg()
    pdf.section_title("2. Dashboard Guide", "")

    pdf.body_text(
        "Streamlitダッシュボードは6つのタブで構成されています。"
        "サイドバーでリーグ選択、EV閾値、バンクロールを設定します。"
    )

    tabs = [
        ("Picks", C_ACCENT,
         "本日のベット推奨を表示。BET/WATCH/SKIPの3段階で\n"
         "色分け表示。コンフィデンスバーで確率を視覚化。\n"
         "Kelly基準によるベット金額を自動計算。"),
        ("Data", (100, 180, 255),
         "終了済み試合データの閲覧。ハンデ結果分布チャート、\n"
         "月別平均ペイアウト推移、直近試合テーブルを表示。\n"
         "表示件数はスライダーで20~200件に調整可能。"),
        ("Backtest", (180, 130, 255),
         "Walk-forwardバックテストを自動実行（キャッシュ済み）。\n"
         "ベット数、勝率、ROI、PnL、Max Drawdownを表示。\n"
         "累積PnLのエリアチャートで収益推移を確認。"),
        ("Teams", C_WARN,
         "チーム別の分析。総試合数、ハンデ有利回数、勝率、\n"
         "平均ペイアウトを表示。PnL推移チャートと\n"
         "結果分布ドーナツチャートで詳細分析。"),
        ("Bankroll", (255, 150, 100),
         "バンクロールシミュレーション。初期資金からの\n"
         "推移をバックテスト結果に基づいて表示。\n"
         "最終資金とリターン率を自動計算。"),
        ("Manual", C_MUTED,
         "システムの使い方マニュアル（このPDFと同内容）。\n"
         "ペイアウト表、EV計算、Kelly基準、\n"
         "用語集をダッシュボード内で確認可能。"),
    ]
    for name, color, desc in tabs:
        pdf.card(color, name, desc)

    # ===== Page 5: Sidebar =====
    pdf.add_page()
    pdf.dark_bg()

    pdf.sub_title("Sidebar Settings")
    settings = [
        ("League", "対象リーグを選択（NBA, Premier, J League, NPB, MLB）"),
        ("EV Threshold", "BET推奨の最低期待値。デフォルト0.08。低くすると推奨数増加。"),
        ("Bankroll", "資金額（円）。Kelly基準のベットサイズ計算に使用。"),
    ]
    for name, desc in settings:
        pdf.set_font("jpb", size=10)
        pdf.set_text_color(*C_ACCENT)
        pdf.set_x(20)
        pdf.cell(35, 7, name)
        pdf.set_font("jp", size=9)
        pdf.set_text_color(*C_TEXT)
        pdf.cell(0, 7, desc, ln=True)

    pdf.ln(3)
    pdf.sub_title("KPI Header")
    pdf.body_text(
        "タブの上部に5つのKPIメトリクスを常時表示:\n"
        "DB Matches — 全リーグの総試合数\n"
        "League Matches — 選択リーグの試合数\n"
        "Scheduled — 未消化の試合数\n"
        "Fav Win Rate — ハンデ有利チームの勝率\n"
        "EV Threshold — 設定中のEV閾値"
    )

    # ===== Page 6: Payout Table =====
    pdf.add_page()
    pdf.dark_bg()
    pdf.section_title("3. Handicap Payout Table", "")

    pdf.body_text(
        "ハンデシステムは adjusted_margin に基づいて9段階の結果に分類されます:\n"
        "adjusted_margin = (有利チーム得点 - 不利チーム得点) - ハンデ値"
    )

    pdf.ln(3)
    headers = ["Result", "Adjusted Margin", "Payout", "Category"]
    widths = [40, 45, 30, 55]
    pdf.table_row(headers, widths, bold=True, fill=C_CARD)

    from config.constants import HANDICAP_OUTCOMES
    for lower, upper, label, payout in HANDICAP_OUTCOMES:
        if payout >= 1.5:
            color = C_ACCENT
            cat = "WIN (Profitable)"
        elif payout >= 1.0:
            color = C_WARN
            cat = "PUSH (Break-even)"
        else:
            color = C_DANGER
            cat = "LOSS"

        if lower == float("-inf"):
            range_str = f"< {upper:.1f}"
        elif upper == float("inf"):
            range_str = f">= {lower:.1f}"
        elif abs(upper - lower) < 0.01:
            range_str = f"= {lower:.1f}"
        else:
            range_str = f"{lower:.1f} ~ {upper:.1f}"

        pdf.table_row(
            [label, range_str, f"{payout:.1f}x", cat],
            widths,
            colors=[color, None, color, C_MUTED],
        )

    pdf.ln(6)
    pdf.card(C_ACCENT, "Example",
             "巨人 vs 阪神 (ハンデ: 巨人 -1.5)\n"
             "巨人 5-2 阪神 -> adjusted = (5-2) - 1.5 = 1.5\n"
             "-> 7分勝ち -> payout 1.7x")

    # ===== Page 7: EV =====
    pdf.add_page()
    pdf.dark_bg()
    pdf.section_title("4. Expected Value (EV)", "")

    pdf.body_text(
        "期待値（EV）は、1単位のベットあたりの期待利益を測定します。"
        "EVが正の値であれば、長期的に利益が期待できます。"
    )

    pdf.ln(2)
    pdf.sub_title("Calculation Formula")
    pdf.code_block(
        "EV = P(fav) x 1.73 + P(push) x 1.0 + P(unfav) x 0.50 - 1.0"
    )

    pdf.sub_title("Parameters")
    params = [
        ("P(fav)", "モデルが予測する有利チームの5分勝ち以上の確率"),
        ("P(push)", "引き分け（勝負無し）の確率。固定値 ~2%"),
        ("P(unfav)", "1 - P(fav) - P(push)。不利な結果の確率"),
        ("1.73", "有利結果の加重平均ペイアウト (丸勝ち~3分勝ち)"),
        ("1.0", "勝負無しのペイアウト（返金）"),
        ("0.50", "不利結果の加重平均ペイアウト (3分負け~丸負け)"),
    ]
    for name, desc in params:
        pdf.set_font("jpb", size=9)
        pdf.set_text_color(*C_ACCENT)
        pdf.set_x(20)
        pdf.cell(25, 6, name)
        pdf.set_font("jp", size=9)
        pdf.set_text_color(*C_TEXT)
        pdf.cell(0, 6, desc, ln=True)

    pdf.ln(4)
    pdf.card(C_ACCENT, "Example: Positive EV",
             "P(fav) = 0.65, P(push) = 0.02, P(unfav) = 0.33\n"
             "EV = 0.65 x 1.73 + 0.02 x 1.0 + 0.33 x 0.50 - 1.0\n"
             "   = 1.1245 + 0.02 + 0.165 - 1.0\n"
             "   = +0.310 (31% expected return)")

    pdf.ln(2)
    pdf.card(C_DANGER, "Example: Negative EV",
             "P(fav) = 0.40, P(push) = 0.02, P(unfav) = 0.58\n"
             "EV = 0.40 x 1.73 + 0.02 x 1.0 + 0.58 x 0.50 - 1.0\n"
             "   = 0.692 + 0.02 + 0.29 - 1.0\n"
             "   = +0.002 (marginal, likely WATCH)")

    # ===== Page 8: Kelly =====
    pdf.add_page()
    pdf.dark_bg()
    pdf.section_title("5. Kelly Criterion", "")

    pdf.body_text(
        "Kelly基準は、長期的な資産成長を最大化する最適ベットサイズを計算する数学的公式です。"
        "このシステムではリスク低減のため1/4 Kellyを使用しています。"
    )

    pdf.sub_title("Formula")
    pdf.code_block(
        "Full Kelly:   f* = (p x b - q) / b\n"
        "1/4 Kelly:    Bet = f* x 0.25 x Bankroll\n"
        "\n"
        "p = Win probability (model output)\n"
        "q = 1 - p (loss probability)\n"
        "b = Net odds (avg_payout - 1.0 = 0.7)"
    )

    pdf.sub_title("Safety Caps")
    caps = [
        ("Kelly Fraction", "0.25 (1/4 Kelly)", "フル Kelly のボラティリティを大幅に削減"),
        ("Max Bet", "20% of Bankroll", "1回のベットが資金の20%を超えない"),
        ("Min EV", "0.08 (default)", "EV閾値以下はベットしない"),
        ("Stop Loss", "30%", "資金が30%減少したらストップ"),
    ]
    headers = ["Parameter", "Value", "Description"]
    widths = [40, 45, 85]
    pdf.table_row(headers, widths, bold=True, fill=C_CARD)
    for row in caps:
        pdf.table_row(list(row), widths)

    pdf.ln(6)
    pdf.card(C_ACCENT, "Example",
             "Probability = 0.65, Bankroll = 100,000 yen\n"
             "b = 1.7 - 1.0 = 0.7\n"
             "f* = (0.65 x 0.7 - 0.35) / 0.7 = 0.15\n"
             "Bet = 0.15 x 0.25 x 100,000 = 3,750 yen")

    # ===== Page 9: Signals =====
    pdf.add_page()
    pdf.dark_bg()
    pdf.section_title("6. Bet Signals", "")

    pdf.body_text(
        "各試合は予測確率とEVに基づいて3段階のシグナルに分類されます。"
    )

    pdf.ln(3)
    signals = [
        ("BET", C_ACCENT,
         f"EV >= {0.08:.2f} (threshold)\n"
         "推奨ベット。Kelly基準でベット金額を算出。\n"
         "モデルが十分なエッジを検出した試合。"),
        ("WATCH", C_WARN,
         "0 < EV < threshold\n"
         "正のEVだが閾値未満。様子見推奨。\n"
         "閾値を下げるか、追加情報で判断。"),
        ("SKIP", C_DANGER,
         "EV <= 0\n"
         "負のEV。ベットすべきでない。\n"
         "長期的に損失が期待される試合。"),
    ]
    for name, color, desc in signals:
        pdf.card(color, name, desc)

    pdf.ln(4)
    pdf.sub_title("Decision Flow")
    pdf.code_block(
        "1. Model predicts P(fav) for each match\n"
        "2. Calculate EV from P(fav)\n"
        "3. If EV >= threshold -> BET (calculate Kelly size)\n"
        "4. If 0 < EV < threshold -> WATCH\n"
        "5. If EV <= 0 -> SKIP"
    )

    # ===== Page 10: Backtest =====
    pdf.add_page()
    pdf.dark_bg()
    pdf.section_title("7. Backtest & Validation", "")

    pdf.body_text(
        "Walk-forwardバックテストは、実際の運用をシミュレートする最も現実的な検証方法です。"
        "過去データで学習 -> ギャップ期間 -> 未来データでベットを繰り返します。"
    )

    pdf.sub_title("Walk-Forward Method")
    pdf.code_block(
        "Period 1: Train [------] Gap [..] Test [===]\n"
        "Period 2: Train [---------] Gap [..] Test [===]\n"
        "Period 3: Train [------------] Gap [..] Test [===]\n"
        "Period 4: Train [---------------] Gap [..] Test [===]\n"
        "\n"
        "Gap = 30 days (prevent data leakage)\n"
        "Model is retrained at each period"
    )

    pdf.sub_title("Metrics")
    metrics = [
        ("Total Bets", "バックテスト期間中のベット回数"),
        ("Win Rate", "ベットの勝率（payout >= 1.0）"),
        ("ROI", "投資収益率（Total PnL / Total Wagered）"),
        ("PnL", "損益合計（円）"),
        ("Max Drawdown", "最大ドローダウン（ピークからの最大下落率）"),
        ("Sharpe Ratio", "リスク調整後リターン（高いほど良い）"),
    ]
    headers = ["Metric", "Description"]
    widths = [45, 125]
    pdf.table_row(headers, widths, bold=True, fill=C_CARD)
    for row in metrics:
        pdf.table_row(list(row), widths)

    # ===== Page 11: CLI =====
    pdf.add_page()
    pdf.dark_bg()
    pdf.section_title("8. CLI Commands", "")

    pdf.body_text(
        "CLIはTyperベースで、全操作をコマンドラインから実行できます。"
    )

    pdf.code_block("cd ~/sports-bet && source .venv/bin/activate")

    commands = [
        ("init-db", "python -m cli.main init-db", "DBの初期化"),
        ("scrape", "python -m cli.main scrape npb --date 20250801", "1日分のスクレイピング"),
        ("backfill", "python -m cli.main backfill npb\n  --start 20240301 --end 20251031", "日付範囲のバックフィル"),
        ("train", "python -m cli.main train baseball", "モデル学習"),
        ("cv", "python -m cli.main cv baseball --folds 5", "時系列クロスバリデーション"),
        ("backtest", "python -m cli.main backtest baseball\n  --ev 0.05 --periods 4", "Walk-forwardバックテスト"),
        ("recommend", "python -m cli.main recommend baseball\n  --date 20260308", "本日のベット推奨"),
        ("status", "python -m cli.main status", "DB状態確認"),
        ("models", "python -m cli.main models", "学習済みモデル一覧"),
    ]
    for name, cmd, desc in commands:
        pdf.set_font("jpb", size=9)
        pdf.set_text_color(*C_ACCENT)
        pdf.set_x(20)
        pdf.cell(24, 6, name)
        pdf.set_font("jp", size=8.5)
        pdf.set_text_color(*C_MUTED)
        pdf.cell(0, 6, desc, ln=True)
        pdf.code_block(cmd)

    # ===== Page 12: Glossary =====
    pdf.add_page()
    pdf.dark_bg()
    pdf.section_title("9. Glossary", "")

    glossary = [
        ("Handicap (ハンデ)", "有利チームに与えるポイントスプレッド。公平な試合条件を作る。"),
        ("Adjusted Margin", "(有利チーム得点 - 不利チーム得点) - ハンデ値。ペイアウト段階を決定。"),
        ("Payout Rate", "ベットに対する倍率。1.0=元本返却、>1.0=利益、<1.0=損失。"),
        ("Favorable Rate", "ペイアウト >= 1.5（5分勝ち以上）の試合の割合。"),
        ("EV (Expected Value)", "1単位ベットあたりの期待利益。正のEV = 長期的に利益。"),
        ("Kelly Criterion", "エッジとオッズに基づく最適ベットサイズの数学的公式。"),
        ("Walk-Forward", "各期間で再学習する先読みバイアスのないバックテスト手法。"),
        ("ROI", "投資収益率。総利益 / 総投資額 (%)。"),
        ("Max Drawdown", "累積PnLのピークからの最大下落幅。リスク指標。"),
        ("Sharpe Ratio", "リスク調整後リターン。高いほどリスク対比で良いパフォーマンス。"),
        ("LightGBM", "勾配ブースティング決定木アルゴリズム。高速・高精度。"),
        ("Time-Series CV", "時系列データ専用のクロスバリデーション。未来データのリークを防止。"),
        ("1/4 Kelly", "フルKellyの25%でベット。ボラティリティを大幅に低減。"),
        ("Stop Loss", "資金が一定割合減少した時にベットを停止する安全装置。"),
    ]

    headers = ["Term", "Definition"]
    widths = [50, 120]
    pdf.table_row(headers, widths, bold=True, fill=C_CARD)
    for term, defn in glossary:
        pdf.set_x(10)
        x = pdf.get_x() + 10
        y = pdf.get_y()
        h = 7

        # Draw bottom border
        pdf.set_draw_color(*C_BORDER)
        pdf.set_line_width(0.15)

        pdf.set_xy(x + 2, y)
        pdf.set_font("jpb", size=8.5)
        pdf.set_text_color(*C_ACCENT)
        pdf.cell(widths[0] - 4, h, term)

        pdf.set_xy(x + widths[0] + 2, y)
        pdf.set_font("jp", size=8.5)
        pdf.set_text_color(*C_TEXT)
        pdf.cell(widths[1] - 4, h, defn)

        pdf.set_y(y + h)
        pdf.set_draw_color(*C_BORDER)
        pdf.line(x, pdf.get_y(), x + sum(widths), pdf.get_y())

    # ===== Back Cover =====
    pdf.add_page()
    pdf.dark_bg()
    pdf.ln(100)
    pdf.set_font("jpb", size=20)
    pdf.set_text_color(*C_ACCENT)
    pdf.cell(0, 12, "Sports Bet", align="C", ln=True)
    pdf.set_font("jpl", size=10)
    pdf.set_text_color(*C_MUTED)
    pdf.cell(0, 8, "Handicap Prediction System", align="C", ln=True)
    pdf.ln(10)
    pdf.set_draw_color(*C_ACCENT)
    pdf.set_line_width(0.5)
    pdf.line(85, pdf.get_y(), 125, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("jp", size=9)
    pdf.set_text_color(*C_MUTED)
    pdf.cell(0, 6, "Built with LightGBM, Streamlit, and Python", align="C", ln=True)

    # ===== Output =====
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT_PATH))
    print(f"PDF generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
