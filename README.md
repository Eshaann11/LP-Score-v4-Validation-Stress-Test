
# LP Score v4 — Validation & Stress-Test

## Contents
- `LP_Score_v4_Validation_Report.html` — 2–3 page visual report with figures and key findings
- `lp_score_v4_analysis.py` — Reproducible Python script to regenerate outputs
- `anomalies.csv` — Wallet/pool-level anomalies (wallet_id, pool_id, reason, metric, value, threshold)
- `figures/` — PNG figures used in the report

## Quick Start
```bash
python lp_score_v4_analysis.py --csv dex-temp-db.score_v4.csv --out lp_score_v4_outputs
```

The script will produce the same folder structure and figures.

## Environment
- Python 3.x
- Libraries: pandas, numpy, matplotlib
