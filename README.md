# Tiltmeter-Dashboard

Upload one or more Tiltmeter sensor CSVs and it will, per sensor:

1. Parse the date column and report any rows it couldn't parse.
2. Plot the raw trend with a rolling N-day average and min/max annotations.
3. Validate `A` and 'B' against configurable bounds (via `pandera`).
4. Remove out-of-bounds rows and re-plot the cleaned trend.
5. Pull monthly readings (start date, then the same calendar date each
   successive month) via as-of lookup.
6. Look up the reading at **any custom dates you pick** (e.g. the three
   dates for a quarterly report) — nearest reading at or before each date,
   with the gap in days shown.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

This opens the dashboard in your browser (usually `http://localhost:8501`).

## Usage

1. In the sidebar, upload one or more Tiltmeter CSVs (e.g. `Tilt_S1.csv`, `Tilt_S2.csv`, ...).
   Each file gets its own tab.
2. Adjust the column names, units, rolling-average window, and validation
   bounds in the sidebar as needed 
3. Within each sensor's tab, review the raw plot, validation results,
   cleaned plot, monthly readings table, and custom date lookup.
4. Use the download buttons to export the cleaned data, monthly readings,
   or date lookup results as CSV.

## Notes

- Bad/unparseable dates are dropped automatically (matching the notebook's
  `errors='coerce'` behavior) and reported as a metric at the top of each tab.
- The custom date lookup uses "as-of" matching: it returns the last reading
  at or before your chosen date. Set a "max acceptable gap" if you want the
  app to blank out results where the nearest reading is too stale.
