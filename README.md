# Ember

Forecasting extreme fire-weather days in Canada with a small Qiskit quantum reservoir.
Built for Qiskit Fall Fest Waterloo 2026.

- Plan: `docs/ember-plan.tex`
- Data: CWFIS daily station Fire Weather Index codes, https://cwfis.cfs.nrcan.gc.ca/downloads/fwi_obs/
  (credit: Canadian Forest Service, CWFIS)

## Data

```bash
cd data/raw
for d in 2000s 2010s 2020s; do
  curl -O https://cwfis.cfs.nrcan.gc.ca/downloads/fwi_obs/cwfis_fwi${d}v3.0_ll.csv.bz2
done
cd ../..
./scripts/extract_stations.sh                    # -> data/raw/stations.csv (Yellowknife A, Kelowna, Jasper Warden)
python scripts/summarise_stations.py             # coverage, q95 threshold, 2023 extremes
```

## Layout

```
data/raw/        downloaded archives + extracted stations.csv (not committed)
data/processed/  per-station season tables
reservoir/       quantum reservoir circuits and features
baselines/       persistence, climatology, AR, ESN, controls
eval/            metrics and tables
hardware/        IBM runs and budget log
slides/          pitch
scripts/         data helpers
```

## IBM credentials

Save your own key once from a terminal with `QiskitRuntimeService.save_account(..., token=getpass.getpass())`.
Never put keys in notebooks or commit them.
