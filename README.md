# UK Political Polling Tracker

A full UK electoral forecasting pipeline: Kalman-filter poll aggregator → MRP seat model → XGBoost swing benchmark → Monte Carlo seat projections.

## Project structure

```
uk-polling-model/
├── src/
│   ├── scraper.py           # Wikipedia poll scraper
│   ├── aggregator.py        # Kalman filter + EM house effects
│   ├── census.py            # ONS Nomis API + IPF joint distribution
│   ├── mrp.py               # MixedLM per party + poststratification
│   ├── ml_model.py          # XGBoost constituency swing benchmark
│   ├── seat_projector.py    # Monte Carlo 10k draws → seat distributions
│   ├── generate_outputs.py  # Writes all JSON + PNG artefacts
│   └── pipeline.py          # Orchestrates all steps end-to-end
├── data/
│   ├── bes_2024.csv         # BES microdata — download separately (see below)
│   ├── results_2024.csv     # HoC Library 2024 results — download separately
│   ├── census_2021.csv      # Written by census.py (auto-generated)
│   └── aggregator_state.pkl # Kalman filter state (auto-generated)
├── output/                  # Generated artefacts (copied to website data/)
├── .github/workflows/
│   └── polling_pipeline.yml # GitHub Actions weekly automation
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# Phase 1 — poll tracker (no BES data needed)
python -m src.scraper       # Scrape Wikipedia polling table
python -m src.aggregator    # Run Kalman filter
python -m src.generate_outputs  # Write polling_predictions.json + vote_share_trend.png

# Phase 2 — census (run once; static data)
python -m src.census        # Download ONS Nomis + run IPF

# Phase 3+ — seat models (requires BES + results data)
python -m src.mrp           # Fit MRP + poststratify
python -m src.ml_model      # Train XGBoost swing model
python -m src.seat_projector # Monte Carlo projections
python -m src.generate_outputs  # Write full output bundle

# Or run the full pipeline in one step:
python -m src.pipeline
python -m src.pipeline --full   # Force full refit from scratch
```

## Data you need to provide manually

### 1. BES 2024 microdata (`data/bes_2024.csv`)

Apply (free) at [britishelectionstudy.com](https://www.britishelectionstudy.com/data-objects/panel-study-data/).
Download the 2024 post-election panel wave (Wave 25) as CSV and save as `data/bes_2024.csv`.

Key columns used: respondent age, sex, education level, ethnicity, vote intention, recalled 2024 vote,
Brexit vote, and Westminster constituency code (`pcon`). If your BES file uses different column names,
update `BES_COL_MAP` in `src/mrp.py`.

### 2. HoC Library 2024 results (`data/results_2024.csv`)

Download from the [House of Commons Library](https://commonslibrary.parliament.uk/research-briefings/cbp-10009/).
The file should contain constituency-level results for 2017, 2019 and 2024, plus:
- `pct_leave` — % Leave at the 2016 EU referendum
- `pct_graduate` — % with degree-level education
- `median_income` — median household income
- `urban_rural` — urban/rural classification

Expected column names are listed in `src/ml_model.py` (`PARTY_COLS`). Adjust if needed.

## GitHub Actions automation

The pipeline runs automatically every Sunday at 09:00 UTC. See `.github/workflows/polling_pipeline.yml`.

### PAT setup (one-time)

The workflow pushes outputs to the `SamHusbands21.github.io` website repo. To authorise this:

1. Go to **GitHub → Settings → Developer Settings → Personal Access Tokens → Fine-grained tokens**
2. Create a new token:
   - Resource owner: `SamHusbands21`
   - Repository access: Only `SamHusbands21.github.io`
   - Permissions: **Contents → Read and Write**
3. Copy the token value
4. Go to the **`uk-polling-model` repo → Settings → Secrets and variables → Actions**
5. Add a new secret: **Name** = `WEBSITE_DEPLOY_TOKEN`, **Value** = the token you copied

The workflow will then automatically commit updated JSON and PNG files to `data/` in the website repo after each run.

## Model architecture

```
Wikipedia polls  ──► scraper.py ──► aggregator.py (Kalman filter + EM)
                                         │
                                   ┌─────┴──────┐
                          national estimates    covariance matrix
                                   │                  │
ONS Census ──► census.py ──► mrp.py (MRP) ──┐         │
BES 2024                         │          │         │
                           constituency   seat_projector.py
results_2024.csv ──► ml_model.py  shares  (Monte Carlo 10k)
                             │    │         │
                             └────┘         │
                                   generate_outputs.py
                                   │
                        ┌──────────┴──────────┐
               polling_predictions.json    PNG charts
               (with history arrays)      (trend, seats, comparison)
```

## Scotland & Wales

- **Scotland (59 seats):** SNP model restricted to Scottish BES respondents and calibrated separately.
  Non-SNP models include an `is_scotland` fixed effect. Scottish seats identified by `S`-prefix constituency codes.
- **Wales (32 seats):** Plaid Cymru model restricted to Welsh respondents. Non-PC models include `is_wales`.
  Welsh seats identified by `W`-prefix constituency codes.
- **Northern Ireland (18 seats):** Excluded — different party system (Sinn Féin, DUP, Alliance etc.).
  Total modelled seats: 632.

## Limitations

- Wikipedia scraping may break if the table structure changes
- MRP binary models per party don't constrain shares to sum to 1 (row-normalised post-hoc)
- XGBoost trained on only ~1,300 rows (2 elections × 650 seats)
- 2021 Census demographics may be stale by the time of the next election
- BES subsamples for Scotland/Wales are small; Scottish/Welsh regional models carry higher uncertainty
