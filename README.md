# AOV Ladder Threshold

Interactive Streamlit app for picking upsell thresholds (T1, T2) on top of a 90-day pharmacy sales distribution.

## Quickstart (developer)
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt -r requirements-dev.txt`
3. Copy `.env.example` to `.env`, fill in Redshift creds.
4. `python build_data.py`   # writes data/*.parquet, data/*.json
5. `streamlit run app.py`

`requirements.txt` covers the runtime app only (streamlit + numpy/pandas/plotly/pyarrow). `requirements-dev.txt` adds psycopg2 and python-dotenv, which are needed by `build_data.py` but not the deployed app.

## Deploy (production)
- Push to git (data/ is committed; .env is not).
- Streamlit Cloud / Railway / Vercel — no env vars needed at runtime.

## Architecture
- `build_data.py` — runs once locally, queries Redshift, writes 5 small summary files.
- `app.py` — pure Streamlit UI. Reads only `data/`. Never touches Redshift.
- All threshold metrics computed live from `data/value_distribution.parquet`.

## Data freshness
Re-run `build_data.py` whenever you want to refresh. The app surfaces `topline.generated_at`.
