# ── NBA Contract Value Index v2 ───────────────────────────────────────────────
# Dark/light mode — landing panel, leaderboard, builder notes
# Run: python app_v2.py → http://localhost:8050

import traceback
import anthropic
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_PATH = "/Users/fis/Desktop/data/processed/master_scored.csv"
df        = pd.read_csv(DATA_PATH)
current   = df[df["SEASON"] == "2025-26"].copy()
current   = current[current["SALARY_M"].notna()].copy()
current   = current.sort_values("PLAYER_NAME")

def team_display(t):
    return "Traded" if t in ["2TM","3TM"] else t

current["TEAM_DISPLAY"] = current["TEAM"].apply(team_display)
TEAMS = sorted([t for t in current["TEAM_DISPLAY"].unique() if t != "Traded"]) + ["Traded"]

# ── Features ──────────────────────────────────────────────────────────────────
FEATURES = {
    "WIN_IMPACT_SCORE": {
        "label":"Win Impact","weight":18.1,"icon":"🏆",
        "short":"How much does this player actually help the team win?",
        "detail":"Composite of BPM (40%), VORP (35%), and Win Shares (25%), normalized within salary tier. A max player is benchmarked against other max players, not role players. BPM measures how much better the team is per 100 possessions with this player on the floor.",
        "builders_note":"I weighted this highest at 18% because the entire point of a contract is winning games. A player can score 30 points a night but if the team plays worse with them on the floor, the contract is failing. The peer normalization within salary tier was critical — Jokić's BPM of 14 shouldn't be compared to a role player's −2. Within franchise tier peers, that 14 is correctly identified as exceptional.",
        "good":"Score ≥ 70: Producing at an elite level for their salary tier",
        "bad":"Score < 40: Significantly underperforming peers at the same salary level",
    },
    "AVAILABILITY_SCORE": {
        "label":"Availability","weight":15.2,"icon":"📅",
        "short":"Is this player actually on the floor?",
        "detail":"3-year weighted rolling average of games played percentage. Weights: most recent season 50%, one year ago 30%, two years ago 20%. This recency weighting is deliberate — a player who missed 40 games last year is a bigger risk than one who missed 40 games three years ago.",
        "builders_note":"Second-highest weight at 15% because an injured star is worth exactly zero. I kept seeing front offices justify overpaying players with injury histories using 'when healthy' arguments. The recency weighting came from noticing that injury patterns are predictive — soft tissue injuries in the last 2 years re-injure at much higher rates. Kawhi's score reflects this correctly.",
        "good":"Score ≥ 70: Consistently available — reliable investment",
        "bad":"Score < 40: Chronic injury risk — significant contract liability",
    },
    "MARKET_SCORE": {
        "label":"Market Value","weight":13.3,"icon":"💰",
        "short":"Is this player paid what the market says they're worth?",
        "detail":"OLS wage regression trained on historical contracts: predicted salary = f(BPM, VORP, Win Shares, age, position). The residual between actual and predicted salary is the market score. Underpaid (negative residual) = high score. Overpaid = low score. Regression trained only on training set to prevent data leakage.",
        "builders_note":"Think of this like a Zillow estimate for players. Known limitation: young players who signed extensions before their peak (Luka at 23) look overpaid by current production but were market-rate at signing time. V2 adds age-at-signing as an interaction term to capture this.",
        "good":"Score ≥ 70: Underpaid relative to production — buy low candidate",
        "bad":"Score < 40: Significantly overpaid vs market rate",
    },
    "AGE_SCORE": {
        "label":"Age/Trajectory","weight":11.4,"icon":"📈",
        "short":"What does the future of this contract look like?",
        "detail":"Gaussian decay curve centered at position-specific peak ages: Guards 27, Forwards 26.5, Centers 27.5. Curve is asymmetric — rises faster than it falls because development is slower than decline. Multi-year contracts add extra penalty for older players: each year remaining beyond 30 costs additional points.",
        "builders_note":"Age in the NBA is brutal and the contract market consistently overvalues it. Teams pay for what a player was at their peak. I made the curve asymmetric after looking at the data. The years-remaining multiplier: a bad 1-year contract is manageable. A bad 4-year contract is a roster-defining problem.",
        "good":"Score ≥ 70: Prime years or ascending — contract value likely improves",
        "bad":"Score < 40: Decline phase locked in — contract gets worse each year",
    },
    "CAP_EFFICIENCY_SCORE": {
        "label":"Cap Efficiency","weight":10.5,"icon":"⚡",
        "short":"Are they producing more than their salary rank predicts?",
        "detail":"BPM tier percentile minus salary tier percentile within peer group. SGA: 83rd percentile BPM in franchise tier but only 33rd percentile salary → +50 excess = high score. Jokić: 100th BPM but also 100th salary → zero excess, scores 50. He earns his contract — just doesn't exceed it.",
        "builders_note":"This went through three complete redesigns. V1 gave minimum salary players scores of 200+. V2 normalized league-wide which compressed all max players to near zero. V3 (current) does peer normalization within tier — finally asking the right question: relative to what you cost in your salary group, how much do you produce?",
        "good":"Score ≥ 70: Outproducing salary rank within peer group",
        "bad":"Score < 40: Underproducing relative to salary rank",
    },
    "ROLE_FIT_SCORE": {
        "label":"Role Fit","weight":9.5,"icon":"🎯",
        "short":"Are they playing the minutes their contract implies?",
        "detail":"Actual minutes per game vs expected minutes for salary tier. Expected: Franchise 34 min, Max 32, Star 28, Role 18. A $40M player logging 19 minutes is a role mismatch — injured, ineffective, or creating chemistry problems. A $5M player logging 32 minutes is punching above their weight. Capped at 1.3× expected.",
        "builders_note":"Minutes = trust from the coaching staff. I added this after noticing pure production metrics miss the 'is this player actually being used?' question. Ben Simmons situations don't show up in BPM until games-played collapses. Role fit catches this earlier. The expected minute benchmarks came from averaging actual minutes across each salary tier over 5 seasons.",
        "good":"Score ≥ 70: Playing the minutes expected for their pay level",
        "bad":"Score < 40: Significant mismatch between pay and playing time",
    },
    "APRON_SCORE": {
        "label":"Apron Impact","weight":8.6,"icon":"📋",
        "short":"How much does this contract limit what the team can do next?",
        "detail":"Penalty structure based on salary and years remaining. The 2023 CBA created the second apron (~$17.5M above luxury tax). Teams above it: cannot trade first-round picks 7+ years out, cannot aggregate salaries in trades, cannot use bi-annual exception. Expiring deals receive a bonus for creating flexibility.",
        "builders_note":"The 2023 CBA made being a second apron team genuinely punishing. I built this to capture the long-term roster construction cost, not just the immediate cap hit. A $55M player on year 1 of 5 is more damaging than year 4 of 5 — the former locks you into pick restrictions for four more offseasons.",
        "good":"Score ≥ 70: Contract creates or preserves future flexibility",
        "bad":"Score < 40: Long expensive deal — restricts roster moves for years",
    },
    "PAYROLL_CONTEXT_SCORE": {
        "label":"Team Payroll","weight":8.6,"icon":"🏦",
        "short":"Is this contract pushing the team over the luxury tax line?",
        "detail":"Real Spotrac apron data combined with player's share of team payroll. Base scores: Second apron = 25, First apron = 60, Below = 100. Additional penalty if player salary exceeds 25% of team total. 2025-26: CLE on second apron, GSW and NYK on first apron. Excluded from Player Perspective CVI.",
        "builders_note":"The most controversial feature. Donovan Mitchell's CVI drops because Cleveland is on the second apron — but that's not his fault. The dual scoring system was the solution: Team CVI includes this. Player CVI excludes it. The gap between the two scores is often the most useful number — it shows exactly how much team context is suppressing or inflating the verdict.",
        "good":"Score ≥ 70: Team has cap flexibility, player not a payroll burden",
        "bad":"Score < 40: Team in second apron — this contract limits everything",
    },
    "ACCOLADES_SCORE": {
        "label":"Accolades","weight":4.8,"icon":"🏅",
        "short":"Has this player proven they can perform at the highest level?",
        "detail":"Weighted career award points with 5-year half-life recency decay. MVP=100, Finals MVP=85, DPOY=55, All-NBA 1st=75, All-Star=30. Decay: points × (0.5 ^ (years_ago / 5)). Award from 5 years ago = 50% value. Not capped — real separation between Jokić (3 MVPs) and a one-time All-Star.",
        "builders_note":"Original version capped raw scores at 100, flattening everyone to the same score — Jokić, Kawhi, and Booker all hit 100. Removing the cap and letting MinMaxScaler handle normalization fixed this: Jokić now scores 57, Booker scores 17 in the same season. 35 key players manually curated. V2 will automate via BBRef scraping.",
        "good":"Score ≥ 50: Decorated player — proven excellence at highest level",
        "bad":"Score < 15: No major accolades — unproven in elite competition",
    },
    "PRESSURE_SCORE": {
        "label":"Playoff Pressure","weight":7.9,"icon":"🔥",
        "short":"Does this player rise or fall when it matters most?",
        "detail":"Three components: BPM delta (50%) — playoff minus regular season BPM. Absolute playoff BPM (30%) — actually good in playoffs? Playoff experience (20%) — games played, capped at 80. Minimum 10 playoff games required — raised from 3 after seeing a player score 49.5 BPM in 6 games due to small sample noise.",
        "builders_note":"Last major feature added and it changed several verdicts. Players with no qualifying data score 50 by design — penalizing Cade Cunningham for the Pistons missing playoffs would be unfair. Most interesting finding: Luka's playoff BPM actually exceeds his regular season BPM, which partly explains why the model rates his contract higher than the rule-based score does.",
        "good":"Score ≥ 70: Elevates under pressure — most valuable when stakes are highest",
        "bad":"Score < 35: Documented playoff regression — concerning for contenders",
    },
}

VERDICT_CONFIG = {
    "good":       {"color_d":"#00ff87","color_l":"#16a34a","bg_d":"#001a0d","bg_l":"#f0fdf4","border_d":"#00ff87","border_l":"#16a34a","label":"Good Contract","icon":"✓"},
    "borderline": {"color_d":"#ffd700","color_l":"#b45309","bg_d":"#1a1500","bg_l":"#fffbeb","border_d":"#ffd700","border_l":"#b45309","label":"Borderline","icon":"~"},
    "bad":        {"color_d":"#ff4444","color_l":"#dc2626","bg_d":"#1a0000","bg_l":"#fef2f2","border_d":"#ff4444","border_l":"#dc2626","label":"Bad Contract","icon":"✗"},
}

TIER_CONFIG = {
    "franchise":    {"color":"#a78bfa","label":"Franchise"},
    "max":          {"color":"#60a5fa","label":"Max"},
    "star":         {"color":"#34d399","label":"Star"},
    "role":         {"color":"#94a3b8","label":"Role"},
    "rookie scale": {"color":"#4ade80","label":"Rookie Scale"},
    "unknown":      {"color":"#64748b","label":"Unknown"},
}

THEMES = {
    "dark": {
        "body_bg":"#0f1117","sidebar_bg":"#0d0d15","card_bg":"#161622",
        "card_bg2":"#1a1a28","border":"#2a2a3a","text_primary":"#f0f0ff",
        "text_secondary":"#8888aa","text_muted":"#4a4a6a",
        "plot_bg":"#0d0d15","plot_paper":"#161622","grid":"#1a1a2e","tick":"#6a6a8a",
        "tab_active":"#00c6ff","tab_inactive":"#1e1e2e",
        "dd_bg":"#12121a","dd_border":"#2a2a3a","dd_text":"#f0f0ff",
        "dd_option_bg":"#12121a","dd_option_hover":"#1a1a2e",
    },
    "light": {
        "body_bg":"#f1f5f9","sidebar_bg":"#ffffff","card_bg":"#ffffff",
        "card_bg2":"#f8fafc","border":"#e2e8f0","text_primary":"#0f172a",
        "text_secondary":"#475569","text_muted":"#94a3b8",
        "plot_bg":"#ffffff","plot_paper":"#ffffff","grid":"#f1f5f9","tick":"#94a3b8",
        "tab_active":"#0284c7","tab_inactive":"#e2e8f0",
        "dd_bg":"#ffffff","dd_border":"#e2e8f0","dd_text":"#0f172a",
        "dd_option_bg":"#ffffff","dd_option_hover":"#f1f5f9",
    },
}

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap",
    ],
    title="NBA CVI — Contract Value Index",
    suppress_callback_exceptions=True,
)

app.index_string = '''<!DOCTYPE html>
<html><head>
    {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
    <style>
        *{box-sizing:border-box;margin:0;padding:0;}
        body{font-family:'Inter',sans-serif;min-height:100vh;transition:background 0.3s,color 0.3s;}
        ::-webkit-scrollbar{width:6px;}
        ::-webkit-scrollbar-track{background:transparent;}
        ::-webkit-scrollbar-thumb{background:#4a4a6a;border-radius:3px;}

        /* Dropdown dark mode overrides */
        .Select-control{background-color:#12121a !important;border:1px solid #2a2a3a !important;border-radius:8px !important;color:#f0f0ff !important;}
        .Select-menu-outer{background-color:#12121a !important;border:1px solid #2a2a3a !important;z-index:9999 !important;}
        .Select-option{background-color:#12121a !important;color:#c0c0d8 !important;}
        .Select-option:hover,.Select-option.is-focused{background-color:#1a1a2e !important;color:#f0f0ff !important;}
        .Select-option.is-selected{background-color:#00c6ff20 !important;color:#00c6ff !important;}
        .Select-value-label{color:#f0f0ff !important;}
        .Select--single>.Select-control .Select-value{color:#f0f0ff !important;}
        .Select-placeholder{color:#4a4a6a !important;}
        .Select-input>input{color:#f0f0ff !important;background:transparent !important;}
        .Select-arrow{border-top-color:#4a4a6a !important;}
        .Select-clear{color:#4a4a6a !important;}
        .is-open .Select-arrow{border-bottom-color:#4a4a6a !important;}

        .header-logo{font-family:'Barlow Condensed',sans-serif;font-weight:800;font-size:26px;letter-spacing:2px;text-transform:uppercase;background:linear-gradient(90deg,#00c6ff,#00ff87);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
        .header-sub{font-size:11px;letter-spacing:1.5px;text-transform:uppercase;margin-top:2px;}
        .analyze-btn{width:100%;background:linear-gradient(135deg,#00c6ff,#00ff87);border:none;border-radius:8px;padding:12px;font-family:'Barlow Condensed',sans-serif;font-size:16px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#0a0a0f;cursor:pointer;margin-top:12px;transition:opacity 0.2s;}
        .analyze-btn:hover{opacity:0.9;}
        .theme-toggle{background:none;border-radius:20px;padding:6px 14px;font-size:12px;cursor:pointer;transition:all 0.2s;letter-spacing:1px;}
        .tab-btn{padding:10px 0;font-family:'Barlow Condensed',sans-serif;font-size:13px;font-weight:600;letter-spacing:2px;text-transform:uppercase;border:none;cursor:pointer;transition:all 0.2s;border-radius:6px;width:48%;}
        .cvi-number{font-family:'Barlow Condensed',sans-serif;font-size:64px;font-weight:800;line-height:1;letter-spacing:-2px;}
        .dual-value{font-family:'Barlow Condensed',sans-serif;font-size:36px;font-weight:800;line-height:1;}
        .info-value{font-family:'Barlow Condensed',sans-serif;font-size:22px;font-weight:700;}
        .feature-score{font-family:'Barlow Condensed',sans-serif;font-size:16px;font-weight:700;text-align:right;}
        .section-title{font-family:'Barlow Condensed',sans-serif;font-size:13px;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;display:flex;align-items:center;gap:10px;}
        .section-title::after{content:'';flex:1;height:1px;}
        .sidebar-title{font-family:'Barlow Condensed',sans-serif;font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;display:flex;align-items:center;gap:8px;}
        .sidebar-title::after{content:'';flex:1;height:1px;}
        .hiw-title{font-family:'Barlow Condensed',sans-serif;font-size:12px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;color:#00c6ff;margin-bottom:6px;}
        .hiw-text{font-size:12px;line-height:1.6;}
        .feature-row{display:grid;grid-template-columns:150px 1fr 48px;align-items:center;gap:12px;margin-bottom:10px;cursor:pointer;padding:8px;border-radius:6px;transition:background 0.15s;}
        .feature-row:hover{background:#1a1a2e;}
        .feature-bar-track{height:8px;border-radius:4px;overflow:hidden;}
        .main-layout{display:grid;grid-template-columns:300px 1fr;min-height:calc(100vh - 65px);}
        .sidebar{padding:20px 18px;overflow-y:auto;max-height:calc(100vh - 65px);}
        .main-content{padding:24px 28px;overflow-y:auto;max-height:calc(100vh - 65px);}
        .sidebar-section{margin-bottom:24px;}
        .verdict-card{border-radius:12px;padding:24px;margin-bottom:20px;position:relative;overflow:hidden;}
        .verdict-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
        .verdict-good::before{background:linear-gradient(90deg,#00ff87,#00c6ff);}
        .verdict-borderline::before{background:linear-gradient(90deg,#ffd700,#ff8c00);}
        .verdict-bad::before{background:linear-gradient(90deg,#ff4444,#ff0066);}
        .info-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:20px;}
        .info-card{border-radius:8px;padding:14px 12px;text-align:center;}
        .info-label{font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;}
        .dual-score-grid{display:grid;grid-template-columns:1fr 1px 1fr 1px 1fr;border-radius:10px;padding:20px;margin-bottom:20px;align-items:center;}
        .dual-item{text-align:center;padding:0 16px;}
        .dual-label{font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;}
        .dual-sub{font-size:10px;margin-top:4px;}
        .features-section{border-radius:10px;padding:20px;margin-bottom:20px;}
        .feature-detail{border-radius:8px;padding:14px 16px;margin:0 8px 12px;font-size:12px;line-height:1.7;}
        .builders-note{font-size:11px;margin-top:10px;padding:10px 12px;border-radius:6px;border-left:3px solid #00c6ff;line-height:1.6;}
        .builders-note-label{font-family:'Barlow Condensed',sans-serif;font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#00c6ff;margin-bottom:4px;}
        .feature-detail-good{font-size:11px;margin-top:8px;padding:6px 10px;border-radius:6px;border-left:2px solid #00ff87;background:#001a0d;color:#00ff87;}
        .feature-detail-bad{font-size:11px;margin-top:6px;padding:6px 10px;border-radius:6px;border-left:2px solid #ff4444;background:#1a0000;color:#ff4444;}
        .ai-card{border-radius:10px;padding:20px;margin-bottom:20px;}
        .ai-label{font-size:10px;color:#00c6ff;text-transform:uppercase;letter-spacing:2px;font-weight:600;margin-bottom:10px;}
        .ai-text{font-size:13px;line-height:1.8;}
        .apron-badge{display:inline-flex;align-items:center;font-size:11px;font-weight:500;padding:3px 10px;border-radius:20px;}
        .tier-badge{display:inline-flex;align-items:center;font-family:'Barlow Condensed',sans-serif;font-size:12px;font-weight:600;letter-spacing:1px;text-transform:uppercase;padding:3px 10px;border-radius:4px;}
        .prob-track{height:6px;border-radius:3px;overflow:hidden;margin:16px 0 4px;}
        .charts-row{display:flex;gap:16px;margin-bottom:20px;}
        .chart-card{flex:1;border-radius:10px;padding:20px;}
        .feature-pill{border-radius:8px;padding:10px 12px;margin-bottom:8px;}
        .feature-pill-header{display:flex;align-items:center;justify-content:space-between;}
        .feature-pill-label{font-size:12px;font-weight:500;}
        .feature-pill-weight{font-size:10px;font-family:'Barlow Condensed',sans-serif;letter-spacing:1px;}
        .feature-pill-short{font-size:11px;margin-top:4px;line-height:1.4;}
        .how-it-works{border-radius:10px;padding:16px;}
        .hiw-section{margin-bottom:14px;}
        .landing-wrap{padding:4px 0;}
        .landing-hero{border-radius:16px;padding:36px 40px;margin-bottom:20px;position:relative;overflow:hidden;}
        .landing-hero::before{content:'';position:absolute;top:0;left:0;right:0;height:4px;background:linear-gradient(90deg,#00c6ff,#00ff87,#a78bfa);}
        .landing-title{font-family:'Barlow Condensed',sans-serif;font-size:42px;font-weight:800;letter-spacing:2px;text-transform:uppercase;background:linear-gradient(90deg,#00c6ff,#00ff87);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1;margin-bottom:12px;}
        .landing-subtitle{font-size:15px;line-height:1.7;max-width:600px;}
        .score-card-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:20px;}
        .score-card{border-radius:12px;padding:20px;border-left:3px solid;}
        .score-card-title{font-family:'Barlow Condensed',sans-serif;font-size:14px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;}
        .score-card-body{font-size:12px;line-height:1.7;}
        .verdict-legend{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px;}
        .verdict-chip{border-radius:8px;padding:14px;text-align:center;}
        .verdict-chip-icon{font-size:24px;margin-bottom:6px;}
        .verdict-chip-label{font-family:'Barlow Condensed',sans-serif;font-size:16px;font-weight:700;letter-spacing:1px;}
        .verdict-chip-range{font-size:11px;margin-top:3px;}
        .stat-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px;}
        .stat-box{border-radius:10px;padding:16px;text-align:center;}
        .stat-number{font-family:'Barlow Condensed',sans-serif;font-size:32px;font-weight:800;color:#00c6ff;}
        .stat-label{font-size:11px;margin-top:4px;text-transform:uppercase;letter-spacing:1px;}
        .leaderboard-wrap{padding:4px 0;}
        .lb-stat-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px;}
        .lb-stat{border-radius:10px;padding:16px;text-align:center;}
        .lb-stat-num{font-family:'Barlow Condensed',sans-serif;font-size:28px;font-weight:800;}
        .lb-stat-label{font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-top:4px;}
    </style>
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>'''

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div(id="app-shell", children=[

    dcc.Store(id="theme-store", data="dark"),
    dcc.Store(id="active-tab",  data="analyze"),

    # Header
    html.Div(id="header-bar", children=[
        html.Div([
            html.Div("NBA CVI", className="header-logo"),
            html.Div("Contract Value Index — 2025-26 Season",
                     id="header-sub", className="header-sub"),
        ]),
        html.Button("☀ Light Mode", id="theme-btn",
                    className="theme-toggle", n_clicks=0),
    ], style={"padding":"14px 32px","display":"flex","alignItems":"center",
              "justifyContent":"space-between","borderBottom":"1px solid #2a2a3a"}),

    html.Div(id="main-layout-wrap", className="main-layout", children=[

        # ── Sidebar ───────────────────────────────────────────────────────────
        html.Div(id="sidebar", className="sidebar", children=[

            # Tab buttons
            html.Div([
                html.Div("Navigation", className="sidebar-title"),
                html.Div([
                    html.Button("Analyze", id="tab-analyze",
                                className="tab-btn", n_clicks=0,
                                style={"marginRight":"4%"}),
                    html.Button("Leaderboard", id="tab-leaderboard",
                                className="tab-btn", n_clicks=0),
                ], style={"display":"flex"}),
            ], className="sidebar-section"),

            # ── Analyze controls ──────────────────────────────────────────────
            html.Div(id="analyze-controls", children=[

                html.Div([
                    html.Div("Select Player", className="sidebar-title"),
                    dcc.Dropdown(
                        id="player-dropdown",
                        options=[
                            {"label": (f"{r['PLAYER_NAME']}  ·  Traded  ·  ${r['SALARY_M']:.1f}M"
                                       if r["TEAM"] in ["2TM","3TM"]
                                       else f"{r['PLAYER_NAME']}  ·  {r['TEAM']}  ·  ${r['SALARY_M']:.1f}M"),
                             "value": r["PLAYER_NAME"]}
                            for _, r in current.sort_values("PLAYER_NAME").iterrows()
                        ],
                        placeholder="Search players...",
                        searchable=True,
                        clearable=False,
                        style={"fontSize":"13px"},
                    ),
                    html.Button("Analyze Contract →", id="analyze-btn",
                                className="analyze-btn", n_clicks=0),
                ], className="sidebar-section"),

                html.Div([
                    html.Div("How It Works", className="sidebar-title"),
                    html.Div(id="hiw-panel", className="how-it-works", children=[
                        html.Div([
                            html.Div("The Model", className="hiw-title"),
                            html.Div("Binary logistic regression trained on 1,068 historical player-seasons (2021-22 through 2024-25). Outputs a 0–100 probability that a contract is good.", className="hiw-text"),
                        ], className="hiw-section"),
                        html.Div([
                            html.Div("Labeling", className="hiw-title"),
                            html.Div("Historical contracts labeled using 4 raw criteria: team win% > .500, BPM above tier threshold, 65%+ games played, salary < 30% of team payroll. 3 of 4 = good contract.", className="hiw-text"),
                        ], className="hiw-section"),
                        html.Div([
                            html.Div("Two Scores", className="hiw-title"),
                            html.Div("Team CVI includes apron and cap context. Player CVI measures individual value only. The gap reveals how much team situation helps or hurts the verdict.", className="hiw-text"),
                        ], className="hiw-section"),
                        html.Div([
                            html.Div("Salary Tiers", className="hiw-title"),
                            html.Div([
                                html.Div("Franchise: Max + top 12 BPM + age ≤32 + 40+ games", style={"color":"#a78bfa","fontSize":"11px","marginBottom":"3px"}),
                                html.Div("Max: $30M+ (not franchise tier)", style={"color":"#60a5fa","fontSize":"11px","marginBottom":"3px"}),
                                html.Div("Star: $15–30M", style={"color":"#34d399","fontSize":"11px","marginBottom":"3px"}),
                                html.Div("Role: Under $15M", style={"color":"#94a3b8","fontSize":"11px"}),
                            ]),
                        ], className="hiw-section"),
                        html.Div([
                            html.Div("ROC-AUC: 0.981", className="hiw-title"),
                            html.Div("5-fold cross-validation. 0.5 = random, 1.0 = perfect. 98% accuracy on held-out test set.", className="hiw-text"),
                        ]),
                    ]),
                ], className="sidebar-section"),

                html.Div([
                    html.Div("Feature Weights", className="sidebar-title"),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Span(f"{m['icon']} {m['label']}", className="feature-pill-label"),
                                html.Span(f"{m['weight']:.1f}%", className="feature-pill-weight"),
                            ], className="feature-pill-header"),
                            html.Div(m["short"], className="feature-pill-short"),
                        ], className="feature-pill")
                        for m in FEATURES.values()
                    ]),
                ], className="sidebar-section"),

            ]),

            # ── Leaderboard controls ──────────────────────────────────────────
            html.Div(id="lb-controls", style={"display":"none"}, children=[

                html.Div([
                    html.Div("Filter", className="sidebar-title"),
                    dcc.Dropdown(
                        id="lb-team-filter",
                        options=[{"label":"All Teams","value":"ALL"}] +
                                [{"label":t,"value":t} for t in TEAMS],
                        value="ALL", clearable=False,
                        style={"fontSize":"12px","marginBottom":"8px"},
                    ),
                    dcc.Dropdown(
                        id="lb-tier-filter",
                        options=[
                            {"label":"All Tiers",    "value":"ALL"},
                            {"label":"Franchise",    "value":"franchise"},
                            {"label":"Max",          "value":"max"},
                            {"label":"Star",         "value":"star"},
                            {"label":"Role",         "value":"role"},
                            {"label":"Rookie Scale", "value":"rookie scale"},
                        ],
                        value="ALL", clearable=False,
                        style={"fontSize":"12px","marginBottom":"8px"},
                    ),
                    dcc.Dropdown(
                        id="lb-verdict-filter",
                        options=[
                            {"label":"All Verdicts","value":"ALL"},
                            {"label":"✓ Good",      "value":"good"},
                            {"label":"~ Borderline","value":"borderline"},
                            {"label":"✗ Bad",       "value":"bad"},
                        ],
                        value="ALL", clearable=False,
                        style={"fontSize":"12px","marginBottom":"8px"},
                    ),
                ], className="sidebar-section"),

                html.Div([
                    html.Div("Sort By", className="sidebar-title"),
                    dcc.Dropdown(
                        id="lb-sort",
                        options=[
                            {"label":"CVI Score (Team)",   "value":"CVI_SCORE_FINAL"},
                            {"label":"CVI Score (Player)", "value":"CVI_PLAYER_SCORE_FINAL"},
                            {"label":"Model Probability",  "value":"MODEL_PROB"},
                            {"label":"Salary (High→Low)",  "value":"SALARY_M"},
                            {"label":"Age",                "value":"AGE"},
                        ],
                        value="CVI_SCORE_FINAL", clearable=False,
                        style={"fontSize":"12px"},
                    ),
                ], className="sidebar-section"),

            ]),

        ]),

        # ── Main content ──────────────────────────────────────────────────────
        html.Div(id="main-content-area", className="main-content", children=[
            html.Div(id="main-panel"),
        ]),

    ]),
])


# ── Theme callbacks ───────────────────────────────────────────────────────────
@app.callback(
    Output("theme-store", "data"),
    Output("theme-btn",   "children"),
    Input("theme-btn",    "n_clicks"),
    State("theme-store",  "data"),
    prevent_initial_call=True,
)
def toggle_theme(n, cur):
    new   = "light" if cur == "dark" else "dark"
    label = "☀ Light Mode" if new == "dark" else "🌙 Dark Mode"
    return new, label


@app.callback(
    Output("app-shell",         "style"),
    Output("header-bar",        "style"),
    Output("header-sub",        "style"),
    Output("sidebar",           "style"),
    Output("main-content-area", "style"),
    Output("hiw-panel",         "style"),
    Output("theme-btn",         "style"),
    Input("theme-store",        "data"),
)
def apply_theme(theme):
    t = THEMES[theme]
    return (
        {"background":t["body_bg"],"color":t["text_primary"],"minHeight":"100vh"},
        {"padding":"14px 32px","display":"flex","alignItems":"center","justifyContent":"space-between","borderBottom":f"1px solid {t['border']}","background":t["sidebar_bg"]},
        {"fontSize":"11px","color":t["text_muted"],"letterSpacing":"1.5px","textTransform":"uppercase","marginTop":"2px"},
        {"background":t["sidebar_bg"],"borderRight":f"1px solid {t['border']}","padding":"20px 18px","overflowY":"auto","maxHeight":"calc(100vh - 65px)"},
        {"padding":"24px 28px","overflowY":"auto","maxHeight":"calc(100vh - 65px)","background":t["body_bg"]},
        {"background":t["card_bg2"],"border":f"1px solid {t['border']}","borderRadius":"10px","padding":"16px"},
        {"background":"none","border":f"1px solid {t['text_muted']}","borderRadius":"20px","padding":"6px 14px","fontSize":"12px","cursor":"pointer","color":t["text_primary"],"letterSpacing":"1px"},
    )


# ── Tab callbacks ─────────────────────────────────────────────────────────────
@app.callback(
    Output("active-tab",       "data"),
    Output("analyze-controls", "style"),
    Output("lb-controls",      "style"),
    Input("tab-analyze",       "n_clicks"),
    Input("tab-leaderboard",   "n_clicks"),
    prevent_initial_call=False,
)
def switch_tab(n_a, n_lb):
    from dash import ctx as dctx
    triggered = dctx.triggered_id if dctx.triggered_id else "tab-analyze"
    if triggered == "tab-leaderboard":
        return "leaderboard", {"display":"none"}, {"display":"block"}
    return "analyze", {"display":"block"}, {"display":"none"}


@app.callback(
    Output("tab-analyze",     "style"),
    Output("tab-leaderboard", "style"),
    Input("active-tab",       "data"),
    Input("theme-store",      "data"),
)
def style_tabs(tab, theme):
    t = THEMES[theme]
    act = {"background":t["tab_active"],"color":"#0a0a0f","fontWeight":"700","width":"48%","marginRight":"4%"}
    ina = {"background":t["tab_inactive"],"color":t["text_secondary"],"width":"48%","marginRight":"4%"}
    act_lb = {"background":t["tab_active"],"color":"#0a0a0f","fontWeight":"700","width":"48%"}
    ina_lb = {"background":t["tab_inactive"],"color":t["text_secondary"],"width":"48%"}
    if tab == "leaderboard":
        return ina, act_lb
    return act, ina_lb


# ── Main panel router ─────────────────────────────────────────────────────────
@app.callback(
    Output("main-panel",       "children"),
    Input("active-tab",        "data"),
    Input("analyze-btn",       "n_clicks"),
    Input("lb-team-filter",    "value"),
    Input("lb-tier-filter",    "value"),
    Input("lb-verdict-filter", "value"),
    Input("lb-sort",           "value"),
    State("player-dropdown",   "value"),
    State("theme-store",       "data"),
)
def render_main(tab, n_clicks, lb_team, lb_tier, lb_verdict, lb_sort, player, theme):
    from dash import ctx as dctx
    t = THEMES[theme]
    if tab == "leaderboard":
        return build_leaderboard(lb_team, lb_tier, lb_verdict, lb_sort, t)
    triggered = dctx.triggered_id
    if triggered != "analyze-btn" or not player:
        return build_landing(t)
    return build_analysis(player, theme, t)


# ── Feature toggles ───────────────────────────────────────────────────────────
def make_toggle(col):
    @app.callback(
        Output(f"fdetail-{col}", "style"),
        Input(f"frow-{col}",    "n_clicks"),
        State(f"fdetail-{col}", "style"),
        State("theme-store",    "data"),
        prevent_initial_call=True,
    )
    def toggle(n, style, theme):
        t      = THEMES[theme]
        hidden = (style or {}).get("display","none") == "none"
        base   = {"background":t["card_bg2"],"color":t["text_secondary"]}
        return {**base,"display":"block"} if hidden else {**base,"display":"none"}
    toggle.__name__ = f"toggle_{col}"
    return toggle

_toggles = {col: make_toggle(col) for col in FEATURES}


# ── Landing ───────────────────────────────────────────────────────────────────
def build_landing(t):
    good_n = (current["CVI_VERDICT_FINAL"]=="good").sum()
    bord_n = (current["CVI_VERDICT_FINAL"]=="borderline").sum()
    bad_n  = (current["CVI_VERDICT_FINAL"]=="bad").sum()
    avg    = current["CVI_SCORE_FINAL"].mean()

    return html.Div([

        html.Div([
            html.Div("Contract Value Index", className="landing-title"),
            html.Div(
                "The CVI is a 0–100 score that measures how much value an NBA player "
                "delivers relative to their contract. It combines 10 data-driven features "
                "— from win impact and playoff pressure to salary burden and team cap "
                "context — into a single verdict every front office should know.",
                className="landing-subtitle",
                style={"color":t["text_secondary"]},
            ),
            html.Div("← Select any player from the sidebar to begin",
                     style={"color":"#00c6ff","fontSize":"13px","fontWeight":"500","marginTop":"16px"}),
        ], className="landing-hero",
           style={"background":t["card_bg"],"border":f"1px solid {t['border']}"}),

        html.Div([
            _stat_box("411",           "Players Scored", t),
            _stat_box(f"{avg:.0f}",    "Avg CVI Score",  t),
            _stat_box("1,068",         "Training Contracts", t),
            _stat_box("0.981",         "Model ROC-AUC",  t),
        ], className="stat-row"),

        html.Div([
            html.Div([
                html.Div("Team CVI", className="score-card-title", style={"color":"#00c6ff"}),
                html.Div("Evaluates the contract from the front office's perspective. Includes apron status, luxury tax impact, and team payroll context. Use this when deciding whether to sign, trade, or waive a player.", className="score-card-body", style={"color":t["text_secondary"]}),
            ], className="score-card", style={"background":t["card_bg"],"border":f"1px solid {t['border']}","borderLeftColor":"#00c6ff"}),
            html.Div([
                html.Div("Player CVI", className="score-card-title", style={"color":"#a78bfa"}),
                html.Div("Evaluates the player's individual value — ignoring their team's cap situation. Use this for trade targets: a player with a low Team CVI but high Player CVI may be undervalued due to team context.", className="score-card-body", style={"color":t["text_secondary"]}),
            ], className="score-card", style={"background":t["card_bg"],"border":f"1px solid {t['border']}","borderLeftColor":"#a78bfa"}),
            html.Div([
                html.Div("Model Probability", className="score-card-title", style={"color":"#00ff87"}),
                html.Div("Logistic regression probability (0–100%) that a contract is 'good' based on patterns in 1,068 historical contracts. Trained independently — disagreements with the rule-based CVI are the most interesting cases.", className="score-card-body", style={"color":t["text_secondary"]}),
            ], className="score-card", style={"background":t["card_bg"],"border":f"1px solid {t['border']}","borderLeftColor":"#00ff87"}),
        ], className="score-card-grid"),

        html.Div([
            html.Div([
                html.Div("✓", className="verdict-chip-icon", style={"color":"#00ff87"}),
                html.Div("Good Contract", className="verdict-chip-label", style={"color":"#00ff87"}),
                html.Div(f"{good_n} players in 2025-26", className="verdict-chip-range", style={"color":t["text_muted"]}),
                html.Div("High production relative to salary. Player is earning or exceeding their contract.", style={"fontSize":"11px","color":t["text_secondary"],"marginTop":"8px","lineHeight":"1.5"}),
            ], className="verdict-chip", style={"background":t["card_bg"],"border":"1px solid #00ff8730"}),
            html.Div([
                html.Div("~", className="verdict-chip-icon", style={"color":"#ffd700"}),
                html.Div("Borderline", className="verdict-chip-label", style={"color":"#ffd700"}),
                html.Div(f"{bord_n} players in 2025-26", className="verdict-chip-range", style={"color":t["text_muted"]}),
                html.Div("Mixed signals. Contract has both positive and negative factors — monitor trajectory.", style={"fontSize":"11px","color":t["text_secondary"],"marginTop":"8px","lineHeight":"1.5"}),
            ], className="verdict-chip", style={"background":t["card_bg"],"border":"1px solid #ffd70030"}),
            html.Div([
                html.Div("✗", className="verdict-chip-icon", style={"color":"#ff4444"}),
                html.Div("Bad Contract", className="verdict-chip-label", style={"color":"#ff4444"}),
                html.Div(f"{bad_n} players in 2025-26", className="verdict-chip-range", style={"color":t["text_muted"]}),
                html.Div("Salary significantly exceeds production. Contract is a net negative for roster construction.", style={"fontSize":"11px","color":t["text_secondary"],"marginTop":"8px","lineHeight":"1.5"}),
            ], className="verdict-chip", style={"background":t["card_bg"],"border":"1px solid #ff444430"}),
        ], className="verdict-legend"),

        html.Div([
            html.Div("Data Sources", style={"fontFamily":"'Barlow Condensed',sans-serif","fontSize":"11px","fontWeight":"600","letterSpacing":"2px","textTransform":"uppercase","color":t["text_muted"],"marginBottom":"10px"}),
            html.Div("Stats: Basketball-Reference (2021-22 → 2025-26)  ·  Salaries: ESPN + BBRef  ·  Apron: Spotrac (2025-26)  ·  Model: scikit-learn logistic regression", style={"fontSize":"12px","color":t["text_muted"],"lineHeight":"1.7"}),
        ], style={"background":t["card_bg"],"border":f"1px solid {t['border']}","borderRadius":"10px","padding":"16px"}),

    ], className="landing-wrap")


# ── Leaderboard ───────────────────────────────────────────────────────────────
def build_leaderboard(team, tier, verdict, sort_col, t):
    filtered = current[current["SALARY_M"].notna()].copy()
    if team and team != "ALL":
        if team == "Traded":
            filtered = filtered[filtered["TEAM"].isin(["2TM","3TM"])]
        else:
            filtered = filtered[filtered["TEAM_DISPLAY"] == team]
    if tier and tier != "ALL":
        tc = "DISPLAY_TIER" if "DISPLAY_TIER" in filtered.columns else "SALARY_TIER"
        filtered = filtered[filtered[tc] == tier]
    if verdict and verdict != "ALL":
        filtered = filtered[filtered["CVI_VERDICT_FINAL"] == verdict]
    filtered = filtered.sort_values(sort_col, ascending=(sort_col=="AGE"))

    rows = []
    for i, (_, r) in enumerate(filtered.head(100).iterrows()):
        v      = str(r.get("CVI_VERDICT_FINAL","borderline"))
        vc     = {"good":"#00ff87","borderline":"#ffd700","bad":"#ff4444"}.get(v,"#8888aa")
        vi     = {"good":"✓","borderline":"~","bad":"✗"}.get(v,"")
        tier_r = str(r.get("DISPLAY_TIER",r.get("SALARY_TIER","")))
        tc_c   = TIER_CONFIG.get(tier_r,TIER_CONFIG["unknown"])["color"]
        td     = str(r.get("TEAM_DISPLAY",""))
        cvi    = float(r.get("CVI_SCORE_FINAL",50) or 50)
        prob   = float(r.get("MODEL_PROB",50) or 50)
        sal    = float(r.get("SALARY_M",0) or 0)
        rows.append(html.Div([
            html.Div(f"{i+1}", style={"width":"32px","fontSize":"12px","color":t["text_muted"],"flexShrink":"0"}),
            html.Div(vi, style={"width":"20px","color":vc,"fontWeight":"700","flexShrink":"0"}),
            html.Div([
                html.Div(r.get("PLAYER_NAME",""), style={"fontSize":"13px","fontWeight":"500","color":t["text_primary"]}),
                html.Div(td, style={"fontSize":"11px","color":t["text_muted"]}),
            ], style={"flex":"2","minWidth":"0"}),
            html.Div(tier_r.title(), style={"fontSize":"10px","color":tc_c,"background":tc_c+"20","padding":"2px 8px","borderRadius":"4px","width":"90px","textAlign":"center","flexShrink":"0","fontWeight":"600","fontFamily":"'Barlow Condensed',sans-serif","letterSpacing":"1px"}),
            html.Div(f"${sal:.1f}M", style={"width":"70px","textAlign":"right","fontSize":"13px","color":t["text_secondary"],"flexShrink":"0"}),
            html.Div([
                html.Div([html.Div(style={"height":"100%","width":f"{cvi}%","background":f"linear-gradient(90deg,{vc}cc,{vc})","borderRadius":"2px"})],
                         style={"flex":"1","height":"6px","background":t["border"],"borderRadius":"2px","overflow":"hidden","marginRight":"8px"}),
                html.Div(f"{cvi:.0f}", style={"width":"30px","textAlign":"right","fontSize":"13px","fontWeight":"700","color":vc,"fontFamily":"'Barlow Condensed',sans-serif"}),
            ], style={"display":"flex","alignItems":"center","width":"120px","flexShrink":"0"}),
            html.Div(f"{prob:.0f}%", style={"width":"50px","textAlign":"right","fontSize":"12px","color":"#a78bfa","flexShrink":"0"}),
        ], style={"display":"flex","alignItems":"center","gap":"12px","padding":"10px 16px","borderBottom":f"1px solid {t['border']}","background":t["card_bg"] if i%2==0 else t["card_bg2"]}))

    avg_cvi = filtered["CVI_SCORE_FINAL"].mean() if len(filtered) > 0 else 0
    avg_sal = filtered["SALARY_M"].mean() if len(filtered) > 0 else 0
    good_n  = (filtered["CVI_VERDICT_FINAL"]=="good").sum()

    return html.Div([
        html.Div([
            _lb_stat(f"{len(filtered)}", "Players",        "#00c6ff", t),
            _lb_stat(f"{avg_cvi:.0f}",  "Avg CVI",         "#00ff87", t),
            _lb_stat(f"{good_n}",        "Good Contracts", "#00ff87", t),
            _lb_stat(f"${avg_sal:.0f}M", "Avg Salary",     "#ffd700", t),
        ], className="lb-stat-row"),

        html.Div([
            html.Div("", style={"width":"32px","flexShrink":"0"}),
            html.Div("", style={"width":"20px","flexShrink":"0"}),
            html.Div("Player", style={"flex":"2","fontSize":"10px","textTransform":"uppercase","letterSpacing":"1px","color":t["text_muted"]}),
            html.Div("Tier",   style={"width":"90px","fontSize":"10px","textTransform":"uppercase","letterSpacing":"1px","color":t["text_muted"],"flexShrink":"0"}),
            html.Div("Salary", style={"width":"70px","textAlign":"right","fontSize":"10px","textTransform":"uppercase","letterSpacing":"1px","color":t["text_muted"],"flexShrink":"0"}),
            html.Div("CVI",    style={"width":"120px","fontSize":"10px","textTransform":"uppercase","letterSpacing":"1px","color":t["text_muted"],"flexShrink":"0"}),
            html.Div("Model",  style={"width":"50px","textAlign":"right","fontSize":"10px","textTransform":"uppercase","letterSpacing":"1px","color":t["text_muted"],"flexShrink":"0"}),
        ], style={"display":"flex","alignItems":"center","gap":"12px","padding":"8px 16px","borderBottom":f"2px solid {t['border']}","marginBottom":"4px"}),

        html.Div(rows, style={"borderRadius":"10px","overflow":"hidden","border":f"1px solid {t['border']}"}),
        html.Div(f"Showing top 100 of {len(filtered)} players",
                 style={"textAlign":"center","fontSize":"11px","color":t["text_muted"],"marginTop":"12px","paddingBottom":"20px"}),
    ], className="leaderboard-wrap")


# ── Player analysis ───────────────────────────────────────────────────────────
def build_analysis(player_name, theme, t):
    print(f"DEBUG: build_analysis — {player_name}")
    try:
        rows = current[current["PLAYER_NAME"] == player_name]
        if len(rows) == 0:
            return html.Div("Player not found.", style={"color":"#ff4444","padding":"20px"})

        row          = rows.iloc[0]
        verdict      = str(row.get("CVI_VERDICT_FINAL","borderline"))
        vc           = VERDICT_CONFIG.get(verdict, VERDICT_CONFIG["borderline"])
        v_color      = vc[f"color_{theme[0]}"]
        v_bg         = vc[f"bg_{theme[0]}"]
        v_border     = vc[f"border_{theme[0]}"]
        tier         = str(row.get("DISPLAY_TIER", row.get("SALARY_TIER","")))
        tc           = TIER_CONFIG.get(tier, TIER_CONFIG["unknown"])
        cvi          = float(row.get("CVI_SCORE_FINAL",50) or 50)
        model_prob   = float(row.get("MODEL_PROB",50) or 50)
        player_score = float(row.get("CVI_PLAYER_SCORE_FINAL",50) or 50)
        gap          = float(row.get("CVI_GAP_FINAL",0) or 0)
        salary       = float(row.get("SALARY_M",0) or 0)
        age          = row.get("AGE","N/A")
        years        = row.get("YEARS_REMAINING","N/A")
        total        = row.get("SALARY_TOTAL_M",None)
        apron        = str(row.get("APRON_STATUS","below"))
        team_raw     = str(row.get("TEAM",""))
        team_disp    = "Traded" if team_raw in ["2TM","3TM"] else team_raw

        fscore = {col: float(row.get(col,50) or 50) for col in FEATURES}
        srt    = sorted(fscore.items(), key=lambda x:x[1], reverse=True)
        top2   = srt[:2]
        bot2   = srt[-2:]

        ai_text = generate_ai_analysis(row)

        apron_cfg = {
            "second":("#ff444420","#ff4444","Second Apron ⚠"),
            "first": ("#ffd70020","#ffd700","First Apron"),
            "below": ("#00ff8720","#00ff87","Below First Apron"),
        }
        ab_bg, ab_color, ab_label = apron_cfg.get(apron,("#ffffff10","#888888","Unknown"))

        def fmt(v):
            try: return str(int(float(v)))
            except: return "N/A"

        verdict_card = html.Div([
            html.Div([
                html.Div([
                    html.Div(vc["icon"], style={"width":"56px","height":"56px","borderRadius":"50%","background":v_color,"color":"#0a0a0f","display":"flex","alignItems":"center","justifyContent":"center","fontSize":"22px","fontWeight":"700","flexShrink":"0","marginRight":"16px"}),
                    html.Div([
                        html.Div(vc["label"], style={"fontFamily":"'Barlow Condensed',sans-serif","fontSize":"24px","fontWeight":"800","color":v_color,"letterSpacing":"1px","textTransform":"uppercase"}),
                        html.Div([html.Span(player_name,style={"color":t["text_primary"],"fontSize":"13px","fontWeight":"500"}),html.Span(f" · {team_disp}",style={"color":t["text_muted"],"fontSize":"13px"})],style={"marginTop":"3px"}),
                        html.Div([
                            html.Span(tc["label"],className="tier-badge",style={"background":tc["color"]+"20","color":tc["color"],"marginRight":"8px"}),
                            html.Span(ab_label,className="apron-badge",style={"background":ab_bg,"color":ab_color,"border":f"1px solid {ab_color}40"}),
                        ],style={"marginTop":"8px"}),
                    ]),
                ],style={"display":"flex","alignItems":"center","flex":"1"}),
                html.Div([
                    html.Div(f"{cvi:.0f}",className="cvi-number",style={"color":v_color}),
                    html.Div("CVI SCORE",style={"fontSize":"10px","color":t["text_muted"],"letterSpacing":"2px","marginTop":"2px","textAlign":"right"}),
                ]),
            ],style={"display":"flex","alignItems":"center","justifyContent":"space-between"}),

            html.Div([html.Div(style={"height":"100%","width":f"{model_prob}%","background":f"linear-gradient(90deg,{v_color},{v_color}aa)","borderRadius":"3px"})],
                     className="prob-track",style={"background":t["border"]}),
            html.Div([
                html.Span("Bad contract",style={"fontSize":"10px","color":t["text_muted"]}),
                html.Span(f"Model probability: {model_prob:.0f}%",style={"fontSize":"11px","color":t["text_secondary"],"fontWeight":"500"}),
                html.Span("Good contract",style={"fontSize":"10px","color":t["text_muted"]}),
            ],style={"display":"flex","justifyContent":"space-between","marginTop":"4px"}),

            html.Div([
                html.Div([
                    html.Div("What's helping",style={"fontSize":"10px","color":t["text_muted"],"letterSpacing":"1px","textTransform":"uppercase","marginBottom":"6px"}),
                    html.Div([html.Span(f"{FEATURES[c]['icon']} {FEATURES[c]['label']}: {s:.0f}",style={"background":"#001a0d","color":"#00ff87","fontSize":"11px","padding":"4px 10px","borderRadius":"20px","marginRight":"6px","border":"1px solid #00ff8740","display":"inline-block","marginBottom":"4px"}) for c,s in top2]),
                ],style={"flex":"1"}),
                html.Div([
                    html.Div("What's hurting",style={"fontSize":"10px","color":t["text_muted"],"letterSpacing":"1px","textTransform":"uppercase","marginBottom":"6px"}),
                    html.Div([html.Span(f"{FEATURES[c]['icon']} {FEATURES[c]['label']}: {s:.0f}",style={"background":"#1a0000","color":"#ff4444","fontSize":"11px","padding":"4px 10px","borderRadius":"20px","marginRight":"6px","border":"1px solid #ff444440","display":"inline-block","marginBottom":"4px"}) for c,s in bot2]),
                ],style={"flex":"1"}),
            ],style={"display":"flex","gap":"20px","marginTop":"16px","paddingTop":"16px","borderTop":f"1px solid {t['border']}"}),
        ],className=f"verdict-card verdict-{verdict}",style={"background":v_bg,"border":f"1px solid {v_border}40"})

        info_grid = html.Div([
            _info_card("Annual Salary",f"${salary:.1f}M",t),
            _info_card("Age",fmt(age),t),
            _info_card("Years Left",fmt(years) if years!="N/A" else "N/A",t),
            _info_card("Total Value",f"${float(total):.0f}M" if total and not pd.isna(float(total)) else "N/A",t),
            _info_card("Team Gap",f"{gap:+.1f}",t,color="#00ff87" if gap<0 else "#ff4444" if gap>3 else t["text_primary"]),
        ],className="info-grid")

        dual_score = html.Div([
            html.Div([html.Div("Team Perspective",className="dual-label",style={"color":t["text_muted"]}),html.Div(f"{cvi:.1f}",className="dual-value",style={"color":"#00c6ff"}),html.Div("Includes cap + apron context",className="dual-sub",style={"color":t["text_muted"]})],className="dual-item"),
            html.Div(style={"background":t["border"],"height":"50px"}),
            html.Div([html.Div("Player Perspective",className="dual-label",style={"color":t["text_muted"]}),html.Div(f"{player_score:.1f}",className="dual-value",style={"color":"#a78bfa"}),html.Div("Individual value only",className="dual-sub",style={"color":t["text_muted"]})],className="dual-item"),
            html.Div(style={"background":t["border"],"height":"50px"}),
            html.Div([html.Div("Model Probability",className="dual-label",style={"color":t["text_muted"]}),html.Div(f"{model_prob:.0f}%",className="dual-value",style={"color":"#00ff87" if model_prob>=65 else "#ffd700" if model_prob>=45 else "#ff4444"}),html.Div("Logistic regression",className="dual-sub",style={"color":t["text_muted"]})],className="dual-item"),
        ],className="dual-score-grid",style={"background":t["card_bg"],"border":f"1px solid {t['border']}"})

        feature_rows = []
        for col, meta in FEATURES.items():
            score = float(row.get(col,50) or 50)
            color = "#00ff87" if score>=70 else "#ffd700" if score>=40 else "#ff4444"
            feature_rows.append(html.Div([
                html.Div([
                    html.Div(f"{meta['icon']} {meta['label']}",style={"fontSize":"12px","color":t["text_secondary"]}),
                    html.Div([html.Div(style={"height":"100%","width":f"{score}%","background":f"linear-gradient(90deg,{color}cc,{color})","borderRadius":"4px"})],
                             className="feature-bar-track",style={"background":t["border"]}),
                    html.Div(f"{score:.0f}",className="feature-score",style={"color":color}),
                ],className="feature-row",id=f"frow-{col}",n_clicks=0),
                html.Div([
                    html.Div(meta["detail"],style={"marginBottom":"10px","color":t["text_secondary"]}),
                    html.Div([
                        html.Div("Builder's Note",className="builders-note-label"),
                        html.Div(meta["builders_note"],style={"color":t["text_secondary"]}),
                    ],className="builders-note",style={"background":t["card_bg"]}),
                    html.Div(meta["good"],className="feature-detail-good"),
                    html.Div(meta["bad"], className="feature-detail-bad"),
                    html.Div(f"Weight in model: {meta['weight']:.1f}%",style={"marginTop":"8px","fontSize":"11px","color":t["text_muted"],"fontStyle":"italic"}),
                ],id=f"fdetail-{col}",className="feature-detail",
                  style={"display":"none","background":t["card_bg2"],"color":t["text_secondary"]}),
            ]))

        features_section = html.Div([
            html.Div("Feature Breakdown  ·  Click any row to expand",className="section-title",style={"color":t["text_muted"]}),
            *feature_rows,
        ],className="features-section",style={"background":t["card_bg"],"border":f"1px solid {t['border']}"})

        charts_row = html.Div([
            html.Div([
                html.Div("Franchise Tier Radar",className="section-title",style={"color":t["text_muted"]}),
                dcc.Graph(figure=make_radar(player_name,row,t),config={"displayModeBar":False},style={"height":"320px"}),
            ],className="chart-card",style={"background":t["card_bg"],"border":f"1px solid {t['border']}"}),
            html.Div([
                html.Div("CVI Score History",className="section-title",style={"color":t["text_muted"]}),
                dcc.Graph(figure=make_history(player_name,t),config={"displayModeBar":False},style={"height":"320px"}),
            ],className="chart-card",style={"background":t["card_bg"],"border":f"1px solid {t['border']}"}),
        ],className="charts-row")

        ai_card = html.Div([
            html.Div("⬡  AI Analysis — Claude Sonnet",className="ai-label"),
            html.Div(ai_text,className="ai-text",style={"color":t["text_secondary"]}),
        ],className="ai-card",style={"background":t["card_bg"],"border":f"1px solid {t['border']}"})

        print(f"DEBUG: success — {player_name}")
        return [verdict_card,info_grid,dual_score,features_section,charts_row,ai_card]

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return html.Div([
            html.Div(f"Error: {player_name}",style={"color":"#ff4444","fontWeight":"600","marginBottom":"8px"}),
            html.Div(str(e),style={"fontSize":"12px","fontFamily":"monospace","color":"#8888aa"}),
        ],style={"padding":"20px"})


# ── Charts ────────────────────────────────────────────────────────────────────
def make_radar(player_name, player_row, t):
    cats = [m["label"] for m in FEATURES.values()]
    keys = list(FEATURES.keys())
    fig  = go.Figure()
    for _, peer in current[current["SALARY_TIER"]=="franchise"].iterrows():
        vals   = [float(peer.get(c,50) or 50) for c in keys]+[float(peer.get(keys[0],50) or 50)]
        is_sel = peer["PLAYER_NAME"]==player_name
        pv     = str(peer.get("CVI_VERDICT_FINAL","borderline"))
        pc     = VERDICT_CONFIG.get(pv,VERDICT_CONFIG["borderline"])["color_d"]
        fig.add_trace(go.Scatterpolar(r=vals,theta=cats+[cats[0]],fill="toself",name=peer["PLAYER_NAME"],line=dict(color=pc,width=3 if is_sel else 1),opacity=1.0 if is_sel else 0.15,showlegend=is_sel))
    if player_row is not None and str(player_row.get("SALARY_TIER"))!="franchise":
        vals = [float(player_row.get(c,50) or 50) for c in keys]+[float(player_row.get(keys[0],50) or 50)]
        pv   = str(player_row.get("CVI_VERDICT_FINAL","borderline"))
        pc   = VERDICT_CONFIG.get(pv,VERDICT_CONFIG["borderline"])["color_d"]
        fig.add_trace(go.Scatterpolar(r=vals,theta=cats+[cats[0]],fill="toself",name=player_name,line=dict(color=pc,width=3),opacity=1.0,showlegend=True))
    fig.update_layout(
        polar=dict(bgcolor=t["plot_bg"],radialaxis=dict(visible=True,range=[0,100],gridcolor=t["grid"],color=t["text_muted"],tickfont=dict(color=t["tick"],size=9)),angularaxis=dict(gridcolor=t["grid"],color=t["tick"],tickfont=dict(color=t["text_secondary"],size=10))),
        paper_bgcolor=t["plot_paper"],plot_bgcolor=t["plot_bg"],height=280,margin=dict(l=40,r=40,t=20,b=60),
        legend=dict(font=dict(color=t["text_secondary"],size=10),bgcolor=t["plot_bg"],bordercolor=t["border"],borderwidth=1,orientation="h",y=-0.2),showlegend=True,
    )
    return fig


def make_history(player_name, t):
    history = df[df["PLAYER_NAME"]==player_name].sort_values("SEASON")
    fig = go.Figure()
    if len(history)==0: return fig
    fig.add_trace(go.Scatter(x=history["SEASON"],y=history["CVI_SCORE_FINAL"],mode="lines+markers",name="CVI Score",line=dict(color="#00c6ff",width=2),marker=dict(size=8,color="#00c6ff"),hovertemplate="<b>%{x}</b><br>CVI: %{y:.1f}<extra></extra>"))
    md = history[history["MODEL_PROB"].notna()]
    if len(md)>0:
        fig.add_trace(go.Scatter(x=md["SEASON"],y=md["MODEL_PROB"],mode="lines+markers",name="Model %",line=dict(color="#a78bfa",width=2,dash="dash"),marker=dict(size=8,color="#a78bfa"),hovertemplate="<b>%{x}</b><br>Model: %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=65,line_dash="dot",line_color="#00ff87",line_width=1,opacity=0.4)
    fig.add_hline(y=45,line_dash="dot",line_color="#ff4444",line_width=1,opacity=0.4)
    fig.update_layout(paper_bgcolor=t["plot_paper"],plot_bgcolor=t["plot_bg"],height=280,margin=dict(l=10,r=20,t=10,b=30),yaxis=dict(range=[0,105],gridcolor=t["grid"],tickfont=dict(color=t["tick"],size=10)),xaxis=dict(gridcolor=t["grid"],tickfont=dict(color=t["tick"],size=10)),legend=dict(font=dict(color=t["text_secondary"],size=10),bgcolor=t["plot_bg"],bordercolor=t["border"]),font=dict(color=t["text_secondary"]))
    return fig


# ── UI helpers ────────────────────────────────────────────────────────────────
def _info_card(label, value, t, color=None):
    return html.Div([
        html.Div(label,className="info-label",style={"color":t["text_muted"]}),
        html.Div(value,className="info-value",style={"color":color or t["text_primary"]}),
    ],className="info-card",style={"background":t["card_bg"],"border":f"1px solid {t['border']}"})


def _stat_box(number, label, t):
    return html.Div([
        html.Div(number,className="stat-number"),
        html.Div(label,className="stat-label",style={"color":t["text_muted"]}),
    ],className="stat-box",style={"background":t["card_bg"],"border":f"1px solid {t['border']}"})


def _lb_stat(number, label, color, t):
    return html.Div([
        html.Div(number,className="lb-stat-num",style={"color":color}),
        html.Div(label,className="lb-stat-label",style={"color":t["text_muted"]}),
    ],className="lb-stat",style={"background":t["card_bg"],"border":f"1px solid {t['border']}"})


def generate_ai_analysis(player_row):
    try:
        client = anthropic.Anthropic()
        prompt = (
            f"You are an NBA front office analyst. Analyze this contract in 2-3 sentences. Be direct and specific.\n\n"
            f"Player: {player_row.get('PLAYER_NAME')} | Team: {player_row.get('TEAM')} | Salary: ${player_row.get('SALARY_M',0):.1f}M/yr\n"
            f"Age: {player_row.get('AGE')} | Tier: {player_row.get('SALARY_TIER')} | Years: {player_row.get('YEARS_REMAINING','N/A')}\n"
            f"CVI: {player_row.get('CVI_SCORE_FINAL',50):.1f}/100 | Model: {player_row.get('MODEL_PROB',50):.0f}% | Verdict: {player_row.get('CVI_VERDICT_FINAL')}\n"
            f"Win Impact: {player_row.get('WIN_IMPACT_SCORE',50):.0f} | Availability: {player_row.get('AVAILABILITY_SCORE',50):.0f} | Market: {player_row.get('MARKET_SCORE',50):.0f}\n"
            f"Playoff: {player_row.get('PRESSURE_SCORE',50):.0f} | Team Payroll: {player_row.get('PAYROLL_CONTEXT_SCORE',50):.0f}\n\n"
            f"2-3 sentences, plain prose, no bullets."
        )
        resp = client.messages.create(model="claude-sonnet-4-5",max_tokens=200,messages=[{"role":"user","content":prompt}])
        return resp.content[0].text
    except Exception:
        verdict = str(player_row.get("CVI_VERDICT_FINAL","borderline"))
        salary  = float(player_row.get("SALARY_M",0) or 0)
        cvi_s   = float(player_row.get("CVI_SCORE_FINAL",50) or 50)
        avail   = float(player_row.get("AVAILABILITY_SCORE",50) or 50)
        market  = float(player_row.get("MARKET_SCORE",50) or 50)
        name    = player_row.get("PLAYER_NAME","This player")
        if verdict=="good":
            driver = "availability" if avail>75 else "market value"
            return f"{name}'s ${salary:.1f}M contract grades as strong value with a CVI of {cvi_s:.0f}. Strong {driver} score is the primary driver. The model probability confirms this as a high-confidence good contract."
        elif verdict=="bad":
            weakness = "availability" if avail<40 else "market value" if market<30 else "overall production"
            return f"{name}'s ${salary:.1f}M contract is flagged as an overpay with a CVI of {cvi_s:.0f}. Weak {weakness} is the primary concern. The model assigns low probability this contract delivers expected value."
        else:
            return f"{name}'s ${salary:.1f}M contract sits in borderline territory with a CVI of {cvi_s:.0f}. Mixed signals across the 9 features balance each other out. Monitor performance trajectory — this contract could trend either direction."


if __name__ == "__main__":
    app.run(debug=True, port=8050)