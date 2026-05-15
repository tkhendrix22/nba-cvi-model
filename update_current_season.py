# update_current_season.py
# Refreshes 2025-26 data by re-running notebook 05's scoring pipeline
# on the fresh BBRef scrape, then merging back into master_scored.csv

import requests
import pandas as pd
import numpy as np
import pickle
import time
import os
import unicodedata
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "cvi_logistic_model.pkl")
MASTER_PATH = os.path.join(DATA_DIR, "processed", "master_scored.csv")

CURRENT_SEASON = "2025-26"
SEASON_YEAR    = 2026

print("=" * 60)
print("NBA CVI — Current Season Update")
print(f"Target: {CURRENT_SEASON}")
print("=" * 60)

# ── Step 1: Load existing master ──────────────────────────────────────────────
print("\n[1/7] Loading existing master...")
master     = pd.read_csv(MASTER_PATH)
historical = master[master["SEASON"] != CURRENT_SEASON].copy()
old_current = master[master["SEASON"] == CURRENT_SEASON].copy()
print(f"  Historical rows: {len(historical)}")
print(f"  Old 2025-26 rows: {len(old_current)}")

# ── Step 2: Scrape fresh stats ────────────────────────────────────────────────
print("\n[2/7] Scraping fresh BBRef stats...")

def fix_name(name):
    if pd.isna(name): return name
    return (unicodedata.normalize("NFKD", str(name))
            .encode("ascii","ignore").decode("utf-8").strip())

NAME_FIXES = {
    "Nikola JokiA"      : "Nikola Jokic",
    "Luka DonAiA"       : "Luka Doncic",
    "Bogdan BogdanoviA" : "Bogdan Bogdanovic",
    "Nikola JoviA"      : "Nikola Jovic",
    "Alperen AengA14n"  : "Alperen Sengun",
    "Dennis SchrAder"   : "Dennis Schroder",
    "Jaren Jackson Jr." : "Jaren Jackson",
    "Gary Payton II"    : "Gary Payton",
    "Michael Porter Jr.": "Michael Porter",
    "PJ Washington"     : "P.J. Washington",
}

def scrape_bbref(url, label):
    headers = {"User-Agent": "Mozilla/5.0"}
    resp    = requests.get(url, headers=headers, timeout=30)
    time.sleep(4)
    tables  = pd.read_html(resp.text)
    df      = tables[0].copy()
    df.columns = [str(c).strip() for c in df.columns]
    df      = df[df.iloc[:,0] != df.columns[0]].copy()
    df      = df[df["Rk"].notna()].copy()
    rename  = {}
    for col in df.columns:
        if col == "Player": rename[col] = "PLAYER_NAME"
        if col in ["Tm","Team"]: rename[col] = "TEAM"
    df = df.rename(columns=rename)
    df["PLAYER_NAME"] = df["PLAYER_NAME"].apply(fix_name).replace(NAME_FIXES)
    skip = ["PLAYER_NAME","TEAM","Pos","Age"]
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["SEASON"] = CURRENT_SEASON
    print(f"  {label}: {len(df)} rows")
    return df

def dedup_traded(df):
    if "TEAM" not in df.columns:
        return df
    traded = df[df["TEAM"].isin(["2TM","3TM"])][
        ["PLAYER_NAME","SEASON"]
    ].drop_duplicates()
    traded["IS_TRADED"] = True
    df = df.merge(traded, on=["PLAYER_NAME","SEASON"], how="left")
    df = df[(df["IS_TRADED"]!=True) | (df["TEAM"].isin(["2TM","3TM"]))].copy()
    df = df.drop(columns=["IS_TRADED"], errors="ignore")
    return df.drop_duplicates(subset=["PLAYER_NAME","SEASON"], keep="first")

per_game = scrape_bbref(
    f"https://www.basketball-reference.com/leagues/NBA_{SEASON_YEAR}_per_game.html",
    "Per-game"
)
time.sleep(3)
adv = scrape_bbref(
    f"https://www.basketball-reference.com/leagues/NBA_{SEASON_YEAR}_advanced.html",
    "Advanced"
)

per_game = dedup_traded(per_game)
adv_keep = [c for c in ["PLAYER_NAME","TEAM","SEASON","BPM","VORP",
                         "WS","USG%","TS%","PER","OBPM","DBPM","WS/48"]
            if c in adv.columns]
adv = dedup_traded(adv[adv_keep])

fresh = per_game.merge(adv, on=["PLAYER_NAME","TEAM","SEASON"], how="left")
print(f"  Merged unique players: {len(fresh)}")

# ── Step 3: Carry forward all existing scores for unchanged players ────────────
print("\n[3/7] Merging existing scores...")

# For players already in our master, keep ALL their computed scores
# Only the raw stats (G, BPM etc) get updated from fresh scrape
# This preserves CVI_SCORE_FINAL, MODEL_PROB, feature scores etc.

# Columns to update from fresh scrape (raw stats)
raw_stat_cols = ["G","GS","MP","FG","FGA","FG%","3P","3PA","3P%",
                 "2P","2PA","2P%","eFG%","FT","FTA","FT%",
                 "ORB","DRB","TRB","AST","STL","BLK","TOV","PF","PTS",
                 "BPM","VORP","WS","USG%","TS%","PER","OBPM","DBPM","WS/48"]
raw_stat_cols = [c for c in raw_stat_cols if c in fresh.columns]

# Merge fresh raw stats onto existing current season data
updated = old_current.copy()

# Drop old raw stat columns and replace with fresh ones
cols_to_drop = [c for c in raw_stat_cols if c in updated.columns]
updated = updated.drop(columns=cols_to_drop)
updated = updated.merge(
    fresh[["PLAYER_NAME"] + raw_stat_cols],
    on="PLAYER_NAME", how="right"
)

# For new players (like Tatum returning), fill from old_current where possible
# then from fresh for everything else
updated["SEASON"] = CURRENT_SEASON

print(f"  Updated players: {len(updated)}")
print(f"  Players with existing CVI scores: {updated['CVI_SCORE_FINAL'].notna().sum()}")
print(f"  New players needing scoring: {updated['CVI_SCORE_FINAL'].isna().sum()}")

# ── Step 4: Score new players using existing feature pipeline ─────────────────
print("\n[4/7] Scoring new/returning players...")

new_players = updated[updated["CVI_SCORE_FINAL"].isna()].copy()
print(f"  Players to score: {len(new_players)}")

if len(new_players) > 0:
    # For new players, copy salary info from old_current if available
    salary_cols = ["SALARY_M","SALARY_TOTAL_M","YEARS_REMAINING","SALARY_TIER",
                   "DISPLAY_TIER","APRON_STATUS","PAYROLL_SHARE_PCT",
                   "TEAM_PAYROLL_M","LAST_TEAM","POSITION","POSITION_GROUP","AGE",
                   "SALARY_CAP_PCT","WIN_IMPACT_SCORE","AVAILABILITY_SCORE",
                   "MARKET_SCORE","AGE_SCORE","CAP_EFFICIENCY_SCORE","ROLE_FIT_SCORE",
                   "APRON_SCORE","PAYROLL_CONTEXT_SCORE","ACCOLADES_SCORE",
                   "PRESSURE_SCORE","UNDERVALUED_FLAG","OVERPAID_MAX_FLAG",
                   "PLAYMAKER_FLAG","ROOKIE_SCALE_FLAG",
                   "CVI_SCORE","CVI_PLAYER_SCORE","CVI_GAP","CVI_VERDICT",
                   "CVI_SCORE_FINAL","CVI_PLAYER_SCORE_FINAL","CVI_GAP_FINAL",
                   "CVI_VERDICT_FINAL","MODEL_PROB","MODEL_VERDICT"]
    salary_cols = [c for c in salary_cols if c in old_current.columns]

    # Check if these new players exist in historical data for context
    new_names = new_players["PLAYER_NAME"].tolist()
    print(f"  New players: {new_names[:10]}")

    # For players like Tatum who have historical data,
    # use their most recent season's feature scores as a starting point
    # then update availability and win impact from fresh stats
    for idx, player_row in new_players.iterrows():
        pname = player_row["PLAYER_NAME"]
        hist  = historical[historical["PLAYER_NAME"] == pname]
        if len(hist) > 0:
            most_recent = hist.sort_values("SEASON").iloc[-1]
            for col in salary_cols:
                if col in most_recent.index and pd.isna(updated.loc[idx, col] if col in updated.columns else np.nan):
                    if col in updated.columns:
                        updated.loc[idx, col] = most_recent[col]

    print(f"  ✓ Historical data merged for returning players")

# ── Step 5: Update availability scores for ALL current players ────────────────
print("\n[5/7] Updating availability scores from fresh G column...")

if "G" in updated.columns:
    scaler = MinMaxScaler((0,100))
    valid  = updated["G"].notna()
    if valid.sum() > 1:
        gp_pct = (updated.loc[valid,"G"] / 82).clip(0,1).values.reshape(-1,1)
        updated.loc[valid,"AVAILABILITY_SCORE"] = scaler.fit_transform(gp_pct).flatten().round(1)
        print(f"  ✓ Availability updated for {valid.sum()} players")

# ── Step 6: Apply model to players missing MODEL_PROB ────────────────────────
print("\n[6/7] Applying model to players missing scores...")

feature_cols = [
    "WIN_IMPACT_SCORE","AVAILABILITY_SCORE","MARKET_SCORE",
    "AGE_SCORE","CAP_EFFICIENCY_SCORE","ROLE_FIT_SCORE",
    "APRON_SCORE","PAYROLL_CONTEXT_SCORE","ACCOLADES_SCORE",
    "PRESSURE_SCORE","UNDERVALUED_FLAG","OVERPAID_MAX_FLAG","PLAYMAKER_FLAG",
]

needs_model = updated["MODEL_PROB"].isna()
print(f"  Players needing model scoring: {needs_model.sum()}")

if needs_model.sum() > 0 and os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    X = updated.loc[needs_model, [c for c in feature_cols
                                   if c in updated.columns]].copy()
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 50.0
        else:
            X[col] = X[col].fillna(50.0)

    probs = model.predict_proba(X.values)[:,1]
    updated.loc[needs_model,"MODEL_PROB"] = (probs * 100).round(1)

    def model_verdict(p):
        if p >= 65:   return "good"
        elif p >= 45: return "borderline"
        else:         return "bad"

    updated.loc[needs_model,"MODEL_VERDICT"] = updated.loc[
        needs_model,"MODEL_PROB"
    ].apply(model_verdict)

    print(f"  ✓ Model applied")
elif not os.path.exists(MODEL_PATH):
    print(f"  ⚠ Model not found at {MODEL_PATH}")

# ── Step 7: Save ──────────────────────────────────────────────────────────────
print("\n[7/7] Saving updated master...")

final = pd.concat([historical, updated], ignore_index=True)
final.to_csv(MASTER_PATH, index=False)

current_check = final[final["SEASON"] == CURRENT_SEASON]
print(f"  ✓ Total rows: {len(final)}")
print(f"  2025-26 players: {len(current_check)}")
print(f"  With CVI scores: {current_check['CVI_SCORE_FINAL'].notna().sum()}")
print(f"  With model prob: {current_check['MODEL_PROB'].notna().sum()}")

print(f"\nTop 10 by CVI — 2025-26:")
print(current_check[current_check["SALARY_M"]>=15][[
    "PLAYER_NAME","TEAM","SALARY_M","CVI_SCORE_FINAL","MODEL_PROB"
]].sort_values("CVI_SCORE_FINAL", ascending=False).head(10).to_string(index=False))

tatum = current_check[current_check["PLAYER_NAME"]=="Jayson Tatum"]
if len(tatum) > 0:
    print(f"\n✓ Jayson Tatum:")
    print(tatum[["PLAYER_NAME","TEAM","G","BPM",
                 "CVI_SCORE_FINAL","MODEL_PROB","CVI_VERDICT_FINAL"]].to_string(index=False))
else:
    print("\n⚠ Tatum still missing")

print("\n✓ Update complete")