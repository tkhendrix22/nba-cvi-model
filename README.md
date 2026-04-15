# NBA Contract Value Index (CVI)

A machine learning project that evaluates NBA player contracts using
a logistic regression classifier. Given a player's stats, salary,
age, team context, and accolades — the model outputs a contract
worthiness score (0–100) and a verdict: good, borderline, or bad.

Built as a data science portfolio project simulating the kind of
analytical tool an NBA front office might use to evaluate roster
decisions, trade targets, and free agent signings.

---

## How It Works

The model calculates 9 feature scores per player-season and combines
them into two composite CVI scores:

- **CVI Score** — team perspective (includes cap and apron context)
- **CVI Player Score** — player perspective (individual value only)
- **CVI Gap** — difference between the two (negative = good player, bad team situation)

### Feature Scorecard

| # | Feature | Weight | Description |
|---|---------|--------|-------------|
| 1 | Win impact | 18.1% | BPM + VORP + WS, peer-normalized within salary tier |
| 2 | Availability | 15.2% | 3-year weighted rolling games played % |
| 3 | Market comparison | 13.3% | OLS wage regression — underpaid vs overpaid |
| 4 | Age / trajectory | 11.4% | Gaussian decay curve by position + years remaining |
| 5 | Cap efficiency | 10.5% | BPM tier percentile minus salary tier percentile |
| 6 | Role / minutes fit | 9.5% | Actual vs expected minutes for salary tier |
| 7 | Apron & tax impact | 8.6% | Contract structure risk under NBA CBA rules |
| 8 | Team payroll context | 8.6% | Real Spotrac apron data + team payroll share |
| 9 | Accolades | 4.8% | Career awards with 5-year recency decay |

### Salary Tier Structure

| Tier | Criteria |
|------|----------|
| Franchise | Max contract + top 12 BPM + age ≤ 32 + 40+ games |
| Max | $30M+ but not franchise tier |
| Star | $15M–$30M |
| Role | Under $15M |

---

## Sample Results (2025-26)

| Player | Salary | CVI Score | Verdict |
|--------|--------|-----------|---------|
| Shai Gilgeous-Alexander | $38.3M | 79.9 | Good |
| Nikola Jokić | $55.2M | 71.0 | Good |
| Tyrese Maxey | $38.0M | 58.1 | Borderline |
| Luka Dončić | $54.1M | 54.1 | Borderline |
| Khris Middleton | $33.9M | 31.9 | Bad |
| Paul George | $51.7M | 33.2 | Bad |

---

## Project Structure
nba-cvi-model/

├── notebooks/

│   ├── 01_data_collection.ipynb      ← BBRef stats scraping (2021-26)

│   ├── 02_salary_data.ipynb          ← ESPN salary + BBRef contracts

│   ├── 03_feature_engineering.ipynb  ← CVI feature calculation

│   ├── 04_eda_visuals.ipynb          ← Exploratory analysis (in progress)

│   └── 05_model_training.ipynb       ← Logistic regression (coming)

├── src/                              ← Utility functions (coming)

├── data/
│   ├── raw/                          ← Source CSVs (gitignored)

│   └── processed/                   ← Feature CSVs (gitignored)

├── requirements.txt

└── README.md


---

## Data Sources

| Source | Data | URL |
|--------|------|-----|
| Basketball-Reference | Per-game + advanced stats | basketball-reference.com |
| ESPN | Historical salary data | espn.com/nba/salaries |
| Basketball-Reference | Contract data | basketball-reference.com/contracts |
| Spotrac | Real apron tracker | spotrac.com/nba/apron-tracker |

> **Note:** Data files are gitignored due to size. Run notebooks 01–03
> in order to regenerate all CSVs from scratch. Total runtime ~45 minutes
> due to API rate limiting delays.

---


### Requirements
- nba_api
- pandas
- numpy
- scikit-learn
- plotly
- requests
- beautifulsoup4
- html5lib
- lxml
- tqdm
- statsmodels
- xgboost
- shap
- anthropic
- jupyterlab

---

## Roadmap

### V1 (current)
- [x] Data collection (BBRef + ESPN + Spotrac)
- [x] 9-feature CVI scoring engine
- [x] Dual scoring (team + player perspective)
- [x] 4-tier salary classification
- [ ] EDA visuals (notebook 04)
- [ ] Logistic regression model (notebook 05)
- [ ] Playoff stats feature
- [ ] Interactive scoring app

### V2 (planned)
- [ ] Marketability score (Google Trends + social media)
- [ ] Jersey sales score (NBA Store rankings)
- [ ] Contract type flags (player option, team option)
- [ ] Injury risk model
- [ ] Salary prediction model (what should a player make?)
- [ ] Full historical Spotrac apron data
- [ ] Automated accolades scraping

---

## Author

Built by Troy Hendrickson — NBA analytics + machine learning portfolio project.

Connect: [[LinkedIn](https://www.linkedin.com/in/troy-hendrickson/)] 

## Live Demo
🏀 **[View the live app →](https://your-render-url.onrender.com)**
