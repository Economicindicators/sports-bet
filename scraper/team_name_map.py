"""チーム名の英語→日本語マッピング"""

from typing import Optional

# The Odds API の英語名 → ハンデの嵐/森の日本語名
EN_TO_JA: dict[str, str] = {
    # ── Serie A ──
    "AC Milan": "ACミラン",
    "AC Monza": "モンツァ",
    "AS Roma": "ローマ",
    "Atalanta BC": "アタランタ",
    "Bologna FC": "ボローニャ",
    "Cagliari Calcio": "カリアリ",
    "Como 1907": "コモ",
    "Empoli FC": "エンポリ",
    "Fiorentina": "フィオレンティーナ",
    "Genoa CFC": "ジェノア",
    "Hellas Verona": "ヴェローナ",
    "Inter Milan": "インテル",
    "Juventus": "ユヴェントス",
    "Lazio": "ラツィオ",
    "Lecce": "レッチェ",
    "SSC Napoli": "ナポリ",
    "Parma Calcio 1913": "パルマ",
    "Torino": "トリノ",
    "Udinese": "ウディネーゼ",
    "Venezia FC": "ヴェネツィア",
    # ── La Liga ──
    "Athletic Bilbao": "ビルバオ",
    "Atletico Madrid": "Aマドリード",
    "FC Barcelona": "バルセロナ",
    "Real Betis": "ベティス",
    "Celta Vigo": "セルタ",
    "Deportivo Alavés": "アラベス",
    "RCD Espanyol": "エスパニョール",
    "Getafe CF": "ヘタフェ",
    "Girona FC": "ジローナ",
    "CD Leganés": "レガネス",
    "RCD Mallorca": "マジョルカ",
    "CA Osasuna": "オサスナ",
    "Rayo Vallecano": "ラージョ",
    "Real Madrid": "レアル・マドリード",
    "Real Sociedad": "ソシエダ",
    "Sevilla FC": "セビージャ",
    "Valencia CF": "バレンシア",
    "Real Valladolid": "バジャドリー",
    "Villarreal CF": "ビジャレアル",
    "UD Las Palmas": "ラスパルマス",
    # ── Bundesliga ──
    "Bayern Munich": "バイエルン",
    "Borussia Dortmund": "ドルトムント",
    "Bayer Leverkusen": "レバークーゼン",
    "RB Leipzig": "RBライプツィヒ",
    "Eintracht Frankfurt": "フランクフルト",
    "SC Freiburg": "フライブルク",
    "VfB Stuttgart": "シュトゥットガルト",
    "VfL Wolfsburg": "ヴォルフスブルク",
    "Borussia Mönchengladbach": "ボルシアMG",
    "TSG Hoffenheim": "ホッフェンハイム",
    "1. FC Union Berlin": "ウニオン・ベルリン",
    "1. FSV Mainz 05": "マインツ",
    "FC Augsburg": "アウクスブルク",
    "Werder Bremen": "ブレーメン",
    "VfL Bochum 1848": "ボーフム",
    "Holstein Kiel": "ホルシュタイン・キール",
    "FC St. Pauli": "ザンクトパウリ",
    "1. FC Heidenheim 1846": "ハイデンハイム",
    # ── Ligue 1 ──
    "Paris Saint Germain": "パリSG",
    "Olympique Marseille": "マルセイユ",
    "AS Monaco": "モナコ",
    "Olympique Lyonnais": "リヨン",
    "LOSC Lille": "リール",
    "OGC Nice": "ニース",
    "RC Lens": "ランス",
    "Stade Rennais FC": "レンヌ",
    "Stade Brestois 29": "ブレスト",
    "RC Strasbourg Alsace": "ストラスブール",
    "Toulouse FC": "トゥールーズ",
    "FC Nantes": "ナント",
    "Montpellier HSC": "モンペリエ",
    "Stade de Reims": "ランス",
    "Angers SCO": "アンジェ",
    "Le Havre AC": "ル・アーヴル",
    "AJ Auxerre": "オセール",
    "AS Saint-Étienne": "サンテティエンヌ",
    # ── Eredivisie ──
    "Ajax Amsterdam": "アヤックス",
    "PSV Eindhoven": "PSV",
    "Feyenoord Rotterdam": "フェイエノールト",
    "AZ Alkmaar": "AZ",
    "FC Twente Enschede": "トゥエンテ",
    "FC Utrecht": "ユトレヒト",
    "SC Heerenveen": "ヘーレンフェーン",
    "Heracles Almelo": "ヘラクレス",
    "NEC Nijmegen": "NECナイメヘン",
    "PEC Zwolle": "ズウォレ",
    "Go Ahead Eagles": "ゴーアヘッド",
    "Fortuna Sittard": "フォルトゥナ",
    "FC Volendam": "フォレンダム",
    "Sparta Rotterdam": "スパルタ",
    "FC Groningen": "フローニンゲン",
    "Willem II": "ウィレムII",
    "RKC Waalwijk": "RKCワールウェイク",
    "NAC Breda": "NACブレダ",
    # ── Premier League ──
    "Arsenal": "アーセナル",
    "Aston Villa": "アストン・ヴィラ",
    "Bournemouth": "ボーンマス",
    "Brentford": "ブレントフォード",
    "Brighton and Hove Albion": "ブライトン",
    "Burnley": "バーンリー",
    "Chelsea": "チェルシー",
    "Crystal Palace": "クリスタル・パレス",
    "Everton": "エヴァートン",
    "Fulham": "フラム",
    "Leeds United": "リーズ",
    "Liverpool": "リヴァプール",
    "Manchester City": "マンチェスター・C",
    "Manchester United": "マンチェスター・U",
    "Newcastle United": "ニューカッスル",
    "Nottingham Forest": "ノッティンガム",
    "Sunderland AFC": "サンダーランド",
    "Tottenham Hotspur": "トッテナム",
    "West Ham United": "ウェストハム",
    "Wolverhampton Wanderers": "ウルブズ",
    # ── NBA ──
    "Atlanta Hawks": "ホークス",
    "Boston Celtics": "セルティックス",
    "Brooklyn Nets": "ネッツ",
    "Charlotte Hornets": "ホーネッツ",
    "Chicago Bulls": "ブルズ",
    "Cleveland Cavaliers": "キャバリアーズ",
    "Dallas Mavericks": "マーベリックス",
    "Denver Nuggets": "ナゲッツ",
    "Detroit Pistons": "ピストンズ",
    "Golden State Warriors": "ウォリアーズ",
    "Houston Rockets": "ロケッツ",
    "Indiana Pacers": "ペイサーズ",
    "Los Angeles Clippers": "クリッパーズ",
    "Los Angeles Lakers": "レイカーズ",
    "Memphis Grizzlies": "グリズリーズ",
    "Miami Heat": "ヒート",
    "Milwaukee Bucks": "バックス",
    "Minnesota Timberwolves": "ウルブズ",
    "New Orleans Pelicans": "ペリカンズ",
    "New York Knicks": "ニックス",
    "Oklahoma City Thunder": "サンダー",
    "Orlando Magic": "マジック",
    "Philadelphia 76ers": "シクサーズ",
    "Phoenix Suns": "サンズ",
    "Portland Trail Blazers": "ブレイザーズ",
    "Sacramento Kings": "キングス",
    "San Antonio Spurs": "スパーズ",
    "Toronto Raptors": "ラプターズ",
    "Utah Jazz": "ジャズ",
    "Washington Wizards": "ウィザーズ",
    # ── MLB ──
    # TODO: シーズン開始時に追加
}

# 逆引き
JA_TO_EN = {v: k for k, v in EN_TO_JA.items()}


def find_ja_name(en_name: str) -> Optional[str]:
    """英語チーム名 → 日本語名"""
    if en_name in EN_TO_JA:
        return EN_TO_JA[en_name]
    # 部分一致
    for en, ja in EN_TO_JA.items():
        if en_name in en or en in en_name:
            return ja
    return None


def find_en_name(ja_name: str) -> Optional[str]:
    """日本語チーム名 → 英語名"""
    if ja_name in JA_TO_EN:
        return JA_TO_EN[ja_name]
    for ja, en in JA_TO_EN.items():
        if ja_name in ja or ja in ja_name:
            return en
    return None
