# Ember

Forecasting wildfire spread with a 5-qubit quantum reservoir.
Built for Qiskit Fall Fest Waterloo 2026.

From day 3 of a fire, Ember predicts how far the fire edge moves in each of 8 compass
sectors over the next 24 h and 72 h. The predictions give a predicted outline of the fire.

- Notebook (start here): `notebooks/ember.ipynb`
- Full method and results: `docs/ember-plan.tex`

## Result (2023 test fires, scored once)

| Model | MAE +24 h (km) | MAE +72 h (km) | Outline IoU +24 h |
|---|---|---|---|
| Persistence (today's advance continues) | 0.349 | 1.070 | **0.918** |
| FBP physics (Canadian spread-rate model) | **0.262** | **0.694** | 0.781 |
| Ridge on raw inputs | 0.348 | 0.920 | 0.766 |
| Classical reservoir (ESN, 20 units) | 0.350 | 0.934 | 0.763 |
| **Quantum reservoir** | 0.360 | 0.943 | 0.757 |
| Random circuit (10 seeds) | 0.361 | 0.946 | 0.761 |

- The quantum reservoir is about level with a classical reservoir and a random circuit, and
  slightly behind ridge on the raw inputs. No quantum advantage is claimed.
- No model wins on every measure. FBP has the lowest mean error. Persistence has the best
  outline overlap, because most sectors do not move. On growing sectors, every learned model
  beats both FBP and persistence.
- On real hardware (`ibm_fez`, one fire, 88 circuits, 10,000 shots, 238 s QPU time) the
  forecasts match the exact simulation to within about 0.02 km.

## How it works

1. **Daily fire shape.** Satellite hotspots are grouped by final fire perimeter. Each day,
   the edge distance from the day-1 centre is measured in 8 sectors. Rules: the day-1 seed is
   the largest cluster of detections within 2 km of each other; the fire grows through chains
   of detections within 2 km; the edge can advance at most 30 km per day.
2. **Inputs.** For each sector: the wind component toward the sector, ISI, today's advance,
   BUI and relative humidity, over the last 5 days.
3. **Quantum reservoir.** 5 qubits in a chain. Each day's inputs are encoded as RY rotations,
   followed by CZ gates and fixed random rotations. 20 expectation values (Z, ZZ, X) are
   measured.
4. **Read-out.** Ridge regression from the features to the +24 h and +72 h advance.
   Predictions are clipped at 0.
5. **Evaluation.** Train 2018–2021, choose settings on 2022, test once on 2023. Errors use
   95% fire-bootstrap intervals.

## Data

Natural Resources Canada, Canadian Forest Service: CWFIS fire perimeter and hotspot archives,
https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/. The raw data is not in this repository.

Download `<YEAR>_hotspots.zip` and `<YEAR>_perimeters.zip` for 2018–2023 into
`data/raw/cwfis_perimeters/`:

```bash
mkdir -p data/raw/cwfis_perimeters && cd data/raw/cwfis_perimeters
for y in 2018 2019 2020 2021 2022 2023; do
  curl -O https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/${y}_hotspots.zip
  curl -O https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/${y}_perimeters.zip
done
cd ../../..
```

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for y in 2018 2019 2020 2021 2022 2023; do .venv/bin/python -I scripts/build_sectors.py --year $y; done
.venv/bin/python -I scripts/run_experiments.py   # E1-E4, E7: validation choice, then one test score
.venv/bin/python -I scripts/run_u5b.py           # E5 shots, E9 sparse hotspots, hardware case file
.venv/bin/python -I scripts/make_figures.py      # figures/ and E8 case study
.venv/bin/python -I scripts/e6_hardware.py       # E6 from saved hardware features
.venv/bin/python -I scripts/build_notebook.py    # notebooks/ember.ipynb
.venv/bin/python -m pytest -q
```

The notebook runs from the committed `results/`, `figures/` and `hardware/` files, so it does
not need the raw data.

## Layout

```
ember/          package: data (hotspots, sectors), reservoir, baselines, metrics, experiments, hardware
scripts/        pipeline scripts (above) and checks
tests/          pytest suite
results/        metrics and experiment tables (CSV/JSON)
figures/        result figures
hardware/       ZF015 hardware case, job log, measured hardware features
notebooks/      ember.ipynb
docs/           method and results (LaTeX)
config.yaml     settings; locked by hash in results/frozen_config.json
```

## Hardware

The hardware run is the Yellowknife fire (ZF015-23), 10–20 August 2023: 11 days × 8 sectors =
88 circuits, 20 CZ gates each, no SWAPs. See `hardware/README.md`. To submit your own job,
save your IBM Quantum key once from a terminal with
`QiskitRuntimeService.save_account(..., token=getpass.getpass())`.
Never put keys in notebooks or commit them.
