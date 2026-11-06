# Ember

**Can a small quantum computer help forecast where a wildfire goes next?**

Ember was built for Qiskit Fall Fest Waterloo 2026. Give it a wildfire that has burned for at
least 3 days, and it predicts how far the fire's edge will move in the next 24 hours and the
next 72 hours, in 8 directions. The forecasting model uses a 5-qubit quantum circuit. We tested
it against simple and classical methods on fires it had never seen, and ran it on a real IBM
quantum computer.

## Short answer

- The quantum model is about as good as similar classical models. It does not beat them, so
  we do not claim a quantum advantage.
- Canada's official fire-spread model (FBP) has the smallest average error. It knows the fuel
  type (forest, grass and so on); our model does not.
- On a real IBM quantum computer, the average error came out close to a perfect simulator
  (within about 0.02 km). Single forecasts were noisier: 0.07 km (24 h) and 0.24 km (72 h) apart
  on average.

## The problem

2023 was Canada's worst fire season on record: 17.2 million hectares burned (CIFFC, Canada
Report 2023). In August, about 20,000 people were ordered to leave the city of Yellowknife as a
fire approached. To plan an evacuation, you need to know where the fire edge will be in the
next day or three.

## How Ember describes a fire

Satellites spot burning points ("hotspots") several times a day. The Canadian Forest Service
publishes these hotspots, plus each fire's final outline at the end of the season. Ember turns
them into a simple daily shape:

1. **Group hotspots into fires**, using each fire's final outline.
2. **Find the fire centre.** On the first day, take the largest group of hotspots that are
   within 2 km of each other. Its middle is the fire centre.
3. **Cut the area into 8 slices** around the centre, like a pizza: N, NE, E, SE, S, SW, W, NW.
4. **Measure the edge every day.** In each slice, the edge is the distance from the centre to
   the furthest point the fire has reached so far. The edge never moves back.

```
         N
    NW   |   NE
      \  |  /
  W ---  *  --- E      * = fire centre
      /  |  \          each slice: distance from * to the fire edge, in km
    SW   |   SE
         S
```

Two rules keep the shape clean. A new hotspot only counts if it is within 2 km of fire that is
already burning, so a separate fire nearby is not mixed in. And the edge can move at most
30 km in one day.

**What Ember predicts:** for each slice, how many km the edge will move in the next 24 h and the
next 72 h. For example: "the north edge moves 0.4 km by tomorrow".

## What the model looks at

For each slice, the last 5 days of:

| Input | What it means |
|---|---|
| Wind toward the slice | How strongly the wind blows toward that direction |
| ISI (Initial Spread Index) | A Canadian fire-weather number: how fast a fire could spread, from wind and how dry the thin surface fuel (needles, grass) is |
| Today's advance | How far the edge moved in that slice today |
| BUI (Buildup Index) | A Canadian fire-weather number: how much dry fuel there is deeper down, in soil and logs |
| Humidity | How much moisture is in the air |

All five come with the hotspot data.

## The quantum part, in plain words

A **qubit** is the basic unit of a quantum computer. You can picture it as an arrow that can
point in any direction, and a quantum operation turns the arrow. Ember uses **5 qubits, one per
input**.

For each of the 5 days, oldest first:

1. **Turn** each qubit by an amount set by that day's input. A bigger value gives a bigger turn.
2. **Link** neighbouring qubits, so that each qubit's state depends on its neighbours. This mixes
   the five inputs together.
3. **Scramble** with a fixed set of random turns. These are chosen once and never change.

After the 5 days, the qubits are **measured**. A quantum measurement gives a random result, so
the circuit is run many times (each run is a "shot") and the results are averaged. Ember
measures each qubit and each pair of qubits, which gives **20 numbers**.

Finally, a simple weighted sum (**ridge regression**) turns those 20 numbers into the two
forecasts. Only the weights of this sum are learned from data; the circuit itself never
changes.

This setup is called a **quantum reservoir**. Think of the circuit as a pond: the inputs are
stones thrown in, and the ripples are the 20 numbers. The pond is never trained. You only learn
how to read the ripples.

## How we tested it

**Fair split by year.** We learned from 2018–2021 (378 fires), chose the settings on 2022
(102 fires), then locked everything and scored 2023 (392 fires) once. That way the test fires
could not influence any choice.

**What we compared against:**

| Method | What it does |
|---|---|
| Persistence | Assumes the edge keeps moving at today's speed. The simplest forecast; anything useful must beat it |
| Mean | Always predicts the average advance |
| FBP physics | Canada's official Fire Behaviour Prediction spread rate, turned into km per slice using the wind direction |
| Ridge on raw inputs | The same weighted sum, fed the inputs directly, with no circuit |
| Classical reservoir (ESN) | The same "pond" idea, but with a small random classical network instead of qubits |
| Random circuit | Our circuit with random links between qubits, to check whether our design matters |

**How we scored:**

- **Average miss (MAE):** on average, how many km the forecast is off. Lower is better.
- **Outline overlap (IoU):** how much the predicted 8-slice shape overlaps the real one. 1 means
  a perfect match.
- **Average miss on growing slices:** the average miss counting only slices where the edge
  really moved. Most slices do not move on most days, so this shows how models do when it
  matters.
- Ranges are 95% intervals, found by resampling whole fires.

## Results (2023 test fires)

| Method | Average miss, +24 h (km) | Average miss, +72 h (km) | Outline overlap, +24 h | Average miss on growing slices, +24 h (km) |
|---|---|---|---|---|
| Persistence | 0.349 | 1.070 | **0.918** | 1.705 |
| FBP physics | **0.262** | **0.694** | 0.781 | 1.593 |
| Ridge on raw inputs | 0.348 | 0.920 | 0.766 | **1.425** |
| Classical reservoir (ESN, 20 units) | 0.350 | 0.934 | 0.763 | 1.429 |
| **Quantum reservoir** | 0.360 | 0.943 | 0.757 | 1.456 |
| Random circuit (average of 10) | 0.361 | 0.946 | 0.761 | 1.446 |

What this means:

- **Quantum close to classical.** The quantum reservoir is close to the classical reservoir (clearly
  worse at 24 h, no clear gap at 72 h) and the random circuit, and slightly behind feeding the inputs straight into the weighted sum.
- **It beats persistence only at 72 h.** At 24 h, "same speed as today" is already hard to beat.
- **No method wins every score.** FBP has the smallest average miss. Persistence draws the best
  outline, because most slices do not move and it predicts exactly 0 for them. On slices that
  are really growing, every learned model beats both FBP and persistence.

## Real quantum hardware

We ran the Yellowknife fire (ZF015, 10–20 August 2023) on IBM's `ibm_fez` quantum computer:
11 days × 8 slices = 88 circuits, 10,000 shots each, 238 seconds of quantum computer time. The
weighted sum learned on the perfect simulator was applied unchanged to the hardware results.

| Source | Average miss, +24 h (km) | Average miss, +72 h (km) |
|---|---|---|
| Perfect simulator | 0.272 | 1.001 |
| Noisy simulator (model of `ibm_fez`) | 0.277 | 1.006 |
| **Real `ibm_fez`** | 0.258 | 0.979 |
| Persistence | 0.301 | 1.228 |

Real quantum computers make small errors (noise). Here the noise barely changed the average error,
but single forecasts moved by 0.07 km (24 h) and 0.24 km (72 h) on average. This is one fire, so it
shows the average error survives the noise. It does not show that the
hardware is better.

## Limits

- Only one fire was run on real hardware.
- The model does not know the fuel type, which is FBP's main advantage.
- Fire shapes are rebuilt from satellite pixels 375 m to 1 km wide, so edges are rough.
- The 30 km/day rule can let two fires that merge look like one fire growing in steps.

## Try it

The notebook `notebooks/ember_explained.ipynb` walks through the whole project, with short
explanations at each step. **It runs on its own:** you only need this one file. The saved results, hardware measurements and figures are stored
inside it, so it needs no other files, no data download and no IBM account. Install four
packages, then run all cells:

```bash
pip install numpy==2.5.4 pandas==3.1.1 qiskit==2.6.1 ipython
```

You can also just open it: the outputs are already saved in it.

To rebuild the notebook from this repository (it is then run in an empty folder to check
that it needs no other files):

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -I scripts/build_notebook_explained.py   # notebooks/ember_explained.ipynb
```

## Rebuild everything from the raw data

1. Download the fire data (about 0.8 GB) from the Canadian Forest Service:

   ```bash
   mkdir -p data/raw/cwfis_perimeters && cd data/raw/cwfis_perimeters
   for y in 2018 2019 2020 2021 2022 2023; do
     curl -O https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/${y}_hotspots.zip
     curl -O https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/${y}_perimeters.zip
   done
   cd ../../..
   ```

2. Run the pipeline:

   ```bash
   for y in 2018 2019 2020 2021 2022 2023; do .venv/bin/python -I scripts/build_sectors.py --year $y; done
   .venv/bin/python -I scripts/run_experiments.py   # choose settings on 2022, then score 2023 once
   .venv/bin/python -I scripts/run_u5b.py           # shot-count and missing-data checks; hardware case file
   .venv/bin/python -I scripts/make_figures.py      # figures/ and the Yellowknife map
   .venv/bin/python -I scripts/e6_hardware.py       # hardware comparison from the saved hardware results
   .venv/bin/python -I scripts/e10_rebuild_check.py --year 2023  # E10 rebuilt shape vs CWFIS outline
   .venv/bin/python -I scripts/e13_build_drivers.py # terrain, fuel and weather inputs for the larger versions
   .venv/bin/python -I scripts/e14_fair_baselines.py   # 5, 7 and 10 qubits vs ESN and ridge, 5 seeds each
   .venv/bin/python -I scripts/plot_qubit_comparison.py # figures/qubit_comparison.png
   .venv/bin/python -I scripts/export_qubit_windows.py  # 7/10-qubit inputs for the notebook
   .venv/bin/python -I scripts/build_notebook_explained.py   # notebooks/ember_explained.ipynb
   .venv/bin/python -m pytest -q                    # tests
   ```

## Repository layout

```
ember/          the code: fire shapes, quantum reservoir, comparison methods, scores, hardware
scripts/        the pipeline steps above, and checks
tests/          automatic tests
results/        result tables (CSV/JSON)
figures/        result charts and the Yellowknife map
hardware/       the Yellowknife hardware case, job log and measured hardware results
notebooks/      ember_explained.ipynb (walkthrough)
slides/         presentation
docs/           full method and results (LaTeX)
config.yaml     settings, locked once the 2022 choices were made
```

## Running your own hardware job

You need your own IBM Quantum account. Save your key once from a terminal with
`QiskitRuntimeService.save_account(..., token=getpass.getpass())`. Never put keys in notebooks
or commit them. See `hardware/README.md`.

## Credits

- Data: Natural Resources Canada, Canadian Forest Service, CWFIS fire perimeter and hotspot
  archives, https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/
- 2023 season figure: CIFFC, Canada Report 2023. Yellowknife evacuation: CBS News, 16 August 2023.
- The reservoir design builds on PronoiaQ (Quantathon 2026).
