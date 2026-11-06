#!/usr/bin/env python3
"""Build and execute the explained, self-contained notebook (notebooks/ember_explained.ipynb).

Every input the notebook reads (saved results, hardware arrays, figures) is
embedded in it as one compressed, checksummed data cell, and the reservoir code
is copied in from ember/reservoir.py. The notebook is executed in an empty
temporary directory, so it is proven to run with no other files.

Run with .venv/bin/python -I scripts/build_notebook_explained.py from the repository root.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from textwrap import dedent
from time import perf_counter

from jupyter_client import KernelManager
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "ember_explained.ipynb"
EMBEDDED = [
    "results/fire_counts.json", "results/e10_2023.json", "results/e3_val.csv",
    "results/frozen_config.json",
    "results/metrics.csv", "results/qrc_comparisons.csv", "results/e4_seeds.csv",
    "results/e2_wind.csv", "results/e5_shots.csv", "results/e9_sparse.csv",
    "results/e6_meta.json", "results/e6_predictions.csv", "results/e6_hardware.csv",
    "results/qrc_z_readout.json", "results/e8_eta.json",
    "hardware/e6_job_metrics.json", "hardware/jobs.json",
    "hardware/e6_hw_features.npy", "hardware/zf015_case.npz",
    "figures/e1_test_mae.png", "figures/e4_seeds.png", "figures/e2_wind.png",
    "figures/e6_hardware.png", "figures/e8_zf015_map.png",
    "figures/qubit_comparison.png", "results/e13_metrics.csv", "results/e13_paired.csv",
]


def bundle_literal():
    """One gzip+base64 string holding every embedded file and its SHA-256."""
    files = {}
    for name in EMBEDDED:
        data = (ROOT / name).read_bytes()
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(),
                       "b64": base64.b64encode(data).decode("ascii")}
    packed = gzip.compress(json.dumps(files, sort_keys=True).encode(), mtime=0)
    return base64.b64encode(packed).decode("ascii")


def reservoir_source():
    """ember/reservoir.py without its module docstring and __future__ import."""
    text = (ROOT / "ember" / "reservoir.py").read_text()
    body = text.split("from __future__ import annotations\n", 1)[1]
    return body.strip()


def build_notebook():
    cells = []

    def markdown(source):
        cells.append(nbformat.v4.new_markdown_cell(dedent(source).strip()))

    def code(source, *, cell_id=None):
        cell = nbformat.v4.new_code_cell(dedent(source).strip())
        if cell_id is not None:
            cell.id = cell_id
        cells.append(cell)

    markdown(r"""
        # Ember

        **Can a small quantum computer help forecast where a wildfire goes next?**

        ### Problem

        - 2023 was Canada's worst fire season on record: 17.2 million ha burned (CIFFC, Canada Report 2023).
        - Yellowknife: ~20,000 people ordered to leave as fire ZF015 approached (August 2023).
        - Evacuation planning needs to know **where the fire edge will be in 1–3 days**, and in
          **which direction** it is moving.

        ### Proposed solution

        **Ember:** for a fire burning 3+ days, predict how far the edge moves in each of 8
        directions, +24 h and +72 h ahead, from free satellite hotspot data.

        ```
        satellite hotspots → 8 sector radii per day → 5 inputs × 5 days
                           → 5-qubit circuit (fixed) → 20 features → ridge (trained) → forecast
        ```

        | Choice | Why |
        |---|---|
        | 8 sectors | direction matters for evacuation; 8 numbers a day keep the data simple |
        | Quantum reservoir | fixed circuit, only ridge is trained → cheap; short circuit → runs on today's noisy hardware |
        | Honest test | settings frozen on 2022, scored once on unseen 2023 fires, against simple, classical and physics baselines |
        | Real hardware | one fire run on `ibm_fez` to check the forecasts survive noise |

        **Short answer**
        - Quantum ≈ classical models; no quantum advantage claimed.
        - FBP (Canada's physics model) has the smallest error.
        - Real `ibm_fez` hardware gives almost the same forecasts as a perfect simulator.

        **Run:** `pip install numpy==2.5.4 pandas==3.1.1 qiskit==2.6.1 ipython`, then run all.
        No other files, data download or IBM account needed.

        | Live in this notebook | Saved (too slow to rerun here) |
        |---|---|
        | Reservoir code, circuit simulation, simulator check, hardware forecasts recomputed from saved measurements | Training on all fires, result tables, figures, `ibm_fez` measurements |

        **Experiments**

        | ID | Tests | ID | Tests |
        |---|---|---|---|
        | E1 | 2023 test scores | E6 | real `ibm_fez` run |
        | E2 | by wind direction | E7 | reservoir + raw inputs |
        | E3 | settings search (2022) | E8 | Yellowknife distance |
        | E4 | random seeds | E9 | missing satellite days |
        | E5 | shots and noise | E10 | rebuilt shape vs official perimeter |

        **Words**

        | Word | Meaning |
        |---|---|
        | Sector | 1 of 8 directions, clockwise from north: 0 N, 1 NE, 2 E, 3 SE, 4 S, 5 SW, 6 W, 7 NW |
        | Radius $r_k(t)$ | km from fire centre to furthest burned point in sector $k$, end of day $t$ |
        | Lead $h$ | 1 day (+24 h) or 3 days (+72 h) |
        | MAE | average miss in km (lower better) |
        | Skill | $1-\mathrm{MAE}/\mathrm{MAE}_\text{persistence}$ (above 0 beats persistence) |
        | IoU | shape overlap ÷ union (1 = perfect) |
    """)
    cells.append(nbformat.v4.new_code_cell(
        "# Embedded data: a compressed copy of the saved result files (results/, hardware/,\n"
        "# figures/), so this notebook needs no other files. The next cell checks every\n"
        "# file against its SHA-256 checksum.\n"
        f'BUNDLE = "{bundle_literal()}"',
        metadata={"jupyter": {"source_hidden": True}, "collapsed": True}))
    cells[-1].id = "embedded-data"

    markdown("""
        ### Setup

        Unpacks the hidden `BUNDLE` above into `FILES` (name → bytes, in memory only) and checks
        each file's SHA-256. Helpers read one file each.
    """)
    code("""
        import base64
        import gzip
        import hashlib
        import io
        import json

        import numpy as np
        import pandas as pd
        from IPython.display import Image, display

        # Unpack the bundle: {file name: bytes}, each checked against its SHA-256.
        FILES = {}
        for name, item in json.loads(gzip.decompress(base64.b64decode(BUNDLE))).items():
            data = base64.b64decode(item["b64"])
            assert hashlib.sha256(data).hexdigest() == item["sha256"], f"corrupted: {name}"
            FILES[name] = data

        pd.set_option("display.max_rows", 50)
        pd.set_option("display.max_columns", 12)
        pd.set_option("display.float_format", lambda value: f"{value:.6f}")

        # Readers that take a file name and return its contents from FILES.
        def read_json(name):
            return json.loads(FILES[name])

        def read_csv(name, **options):
            return pd.read_csv(io.BytesIO(FILES[name]), **options)

        def load_array(name):
            return np.load(io.BytesIO(FILES[name]), allow_pickle=False)

        def show_figure(name):
            display(Image(data=FILES["figures/" + name], format="png", width=950))

        def mae(y, yhat):
            \"\"\"Mean absolute error over the supplied targets.\"\"\"
            return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))

        print(f"{len(FILES)} embedded files verified; distances and errors are in km.")
    """)

    markdown(r"""
        ## Data and sector geometry

        **Fire → 8 numbers a day**
        1. Fire centre: middle of the largest day-1 hotspot group (linked within 2 km).
        2. Cut into 8 sectors like a pizza.
        3. Radius $r_k(t)$ = furthest burned point so far; never shrinks.
        4. New hotspots count only within 2 km of the fire; max 30 km growth a day.

        **Target:** advance from the end of day $t$:
        $$y_k^{(h)}(t)=r_k(t+h)-r_k(t),\qquad h\in\{1,3\}.$$
        First forecast on fire day 3. The model sees days $t-4$ to $t$.

        **Inputs** (per sector, per day; weather = daily median over the fire's hotspots)

        | Qubit | Input | Meaning |
        |---|---|---|
        | 0 | `w_k` | wind toward the sector (km/h) |
        | 1 | `isi` | Initial Spread Index: how fast fire could spread |
        | 2 | `dd_km` | today's advance, $r_k(t)-r_k(t-1)$ (km) |
        | 3 | `bui` | Buildup Index: dry fuel deeper down |
        | 4 | `rh` | relative humidity (%) |

        **Years:** train 2018–2021 · choose settings 2022 · test 2023 once.
    """)
    code("""
        counts = pd.DataFrame.from_dict(read_json("results/fire_counts.json"), orient="index")
        counts.index.name = "year"
        display(counts[["split", "fires", "sector_rows", "origin_days"]])
        # Fires per split: train 2018-2021, val 2022, test 2023.
        display(counts.groupby("split", sort=False)[["fires"]].sum())
    """)
    markdown("""
        ### E10: does the rebuilt shape match the real fire?

        IoU of each 2023 fire's final shape vs the official perimeter (median over fires).

        | Shape | Meaning |
        |---|---|
        | eight-point outline | polygon through the 8 radii (what the model sees) |
        | buffered hotspots | every hotspot as a 375 m disc (raw data coverage) |

        0.65 vs 0.83 → 8 sectors are coarse but follow the fire.
    """)
    code("""
        rebuild = read_json("results/e10_2023.json")
        print(f"E10 median IoU, eight-point outline: {rebuild['median_iou']:.6f}")
        print(f"E10 median IoU, buffered hotspots: {rebuild['median_iou_buffered']:.6f}")
    """)

    markdown(r"""
        ## Quantum reservoir

        **Idea:** a fixed random circuit turns the input history into features; only the
        read-out on top is trained.

        **One window** = one sector, 5 inputs × 5 days, scaled to $[0,1]$ (min/max from train).
        Start in $|00000\rangle$, then each day, oldest first:

        | Step | Gate | Role |
        |---|---|---|
        | Encode | $R_Y(\gamma\,g_i\,u_i)$ on qubit $i$ | today's input turns its qubit |
        | Link | CZ on 0–1, 1–2, 2–3, 3–4 | mix neighbours |
        | Scramble | $R_Z(b_i)\,R_Y(a_i)$ | fixed random turns |

        State carries day to day (memory). Read **20 features**: 5 $\langle Z_i\rangle$ +
        10 $\langle Z_iZ_j\rangle$ + 5 $\langle X_i\rangle$ → ridge → forecast (clipped ≥ 0).

        | Part | Status |
        |---|---|
        | gains $g_i\in[0.5,1.5]$, angles $a_i,b_i$ | fixed, seed 7 |
        | window length, $\gamma$, input order, $\lambda$ | chosen on 2022 (E3) |
        | ridge weights | trained on 2018–2021 |

        Next cell: full reservoir code (= `ember/reservoir.py`). Feature modes: `exact`
        (Qiskit), `numpy` (fast copy, used for training), `sampled` (finite shots).
    """)
    cells.append(nbformat.v4.new_code_cell(reservoir_source()))
    cells[-1].id = "reservoir-code"

    markdown(r"""
        ### E3: settings search (2022 only)

        | Setting | Tried |
        |---|---|
        | `L` (days) | 3, 5, 7 |
        | `gamma` | $\pi/4$, $\pi/2$, $3\pi/4$ |
        | input order | `w_k, isi, dd_km, …` or `w_k, dd_km, isi, …` |
        | $\lambda$ | best of $10^{-6}$ … $10^{2}$ |

        18 settings, ranked by skill (mean of both leads). Winner: L 5, $\gamma=\pi/4$.
        All skills slightly below 0, top settings very close.
    """)
    code("""
        e3 = read_csv("results/e3_val.csv")
        e3["input_order"] = e3.input_map.map(lambda text: ", ".join(json.loads(text)))
        e3["gamma"] = (e3.gamma / np.pi).map(lambda value: f"{value:.2f}*pi")
        ranked = e3.sort_values("val_skill_mean", ascending=False).reset_index(drop=True)
        print(f"E3 tried {len(e3)} settings on 2022 validation; top five by mean skill:")
        display(ranked[["L", "gamma", "input_order", "lam_1", "val_mae_1",
                        "lam_3", "val_mae_3", "val_skill_mean"]].head(5))
    """)

    markdown("""
        ### Frozen settings and one circuit

        Loads the E3 winner and 88 saved Yellowknife (ZF015) windows (11 days × 8 sectors),
        then draws the circuit for the first window.
    """)
    code("""
        # E3 winner, frozen before any test fire was scored.
        frozen = read_json("results/frozen_config.json")
        choices = {key: frozen[key] for key in ("L", "gamma", "input_map", "lambdas", "reservoir_seed")}
        print(json.dumps(choices, indent=2))
        reservoir = ChainReservoir(gamma=frozen["gamma"], seed=frozen["reservoir_seed"])
        # 88 Yellowknife windows: (window, day, input), already scaled to [0, 1].
        with load_array("hardware/zf015_case.npz") as case:
            W = case["windows"].copy()
            case_rows = pd.DataFrame({
                "date": pd.to_datetime(case["dates"]),
                "sector": case["sectors"],
            })
        assert W.shape == (88, 5, 5)
        print("ZF015 window archive shape:", W.shape)
        print(reservoir.circuit(W[0]).draw("text", fold=110))
        names = feature_names(reservoir.n)
        assert len(names) == 20
        print("Feature names (20):", ", ".join(names))
    """)
    markdown("""
        ### Check: fast NumPy simulator = Qiskit exact

        Must agree within $10^{-10}$ (training used NumPy).
    """)
    code("""
        numpy_features = reservoir.feature_matrix(W[:2], mode="numpy")
        exact_features = reservoir.feature_matrix(W[:2], mode="exact")
        max_diff = float(np.max(np.abs(numpy_features - exact_features)))
        print(f"NumPy / exact feature max difference: {max_diff:.12e}")
        assert max_diff <= 1e-10
    """)

    markdown(r"""
        ## Baselines

        | `model` | What | Why |
        |---|---|---|
        | `qrc` | **our quantum reservoir** | |
        | `persistence` | $h$ × today's advance | simplest; must beat it |
        | `mean` | average training advance | floor |
        | `fbp` | physics spread rate × burn time × wind share × $h$ | knows fuel type; we don't |
        | `raw` | ridge on raw inputs, no circuit | does the circuit help? |
        | `esn5`, `esn20` | classical random network (5 / 20 units) + ridge | same idea, no qubits |
        | `haar` | random two-qubit gates instead of CZ | does our design matter? |
        | `qrc_raw` (E7) | reservoir features + raw inputs | do they combine? |
    """)

    markdown("""
        ## Test 2023 results

        E1: all models frozen, scored once on 392 test fires.

        - `mae [lo, hi]`: km, 95% interval (resampling whole fires)
        - `mae_growing`: MAE on sectors that really moved
        - Paired table: `qrc` minus other model; **negative favours `qrc`**; clear if interval misses 0
    """)
    code("""
        metrics = read_csv("results/metrics.csv")
        test = metrics.loc[metrics.split.eq("test")].copy()
        result_table = test[["model", "lead", "skill", "iou", "mae_growing"]].copy()
        result_table["lead"] *= 24  # days -> hours
        result_table = result_table.rename(columns={"lead": "lead_h"})
        result_table.insert(2, "mae [mae_lo, mae_hi]", [
            f"{row.mae:.6f} [{row.mae_lo:.6f}, {row.mae_hi:.6f}]"
            for row in test.itertuples()
        ])
        display(result_table.sort_values(["lead_h", "model"]).reset_index(drop=True))
        comparisons = read_csv("results/qrc_comparisons.csv")
        comparisons["lead_h"] = comparisons.pop("lead") * 24
        print("Paired MAE differences: reservoir minus reference; negative favours the reservoir.")
        display(comparisons)
        show_figure("e1_test_mae.png")
    """)
    markdown("""
        **Takeaways**
        - `qrc` 0.360 / 0.943 km ≈ `esn20`, `haar`.
        - `raw` ridge beats `qrc` by 0.01–0.02 km (clear).
        - `qrc` beats persistence only at +72 h.
        - Best MAE: `fbp`. Best IoU: `persistence` (most sectors don't move). Growing sectors: learned models win.
    """)

    markdown("""
        ## Controls

        | ID | Question | Answer |
        |---|---|---|
        | E4 | Lucky random circuit? | No: seeds vary by ~0.01–0.02 km |
        | E2 | Holds by wind direction? | Yes: all miss most downwind; `qrc` ≈ `esn20` in each |
        | E5 | How many shots needed? | 100 shots → 1.21 km; 10,000 → within 0.05 km of exact |
        | E9 | Missing satellite days? | Degrades like `esn20`, less than persistence |
    """)
    code("""
        seeds = read_csv("results/e4_seeds.csv")
        seeds = seeds.loc[seeds.split.eq("test")]
        print("E4 — fixed-seed test MAE (km)")
        display(seeds.groupby(["model", "lead"]).agg(
            seeds=("seed", "nunique"), mae_mean=("mae", "mean"), mae_sd=("mae", "std")
        ))
        show_figure("e4_seeds.png")
    """)
    code("""
        wind = read_csv("results/e2_wind.csv")
        print("E2 — test MAE by wind bin (km); lead is in days")
        display(wind.pivot(index=["model", "wind_bin"], columns="lead", values="mae"))
        show_figure("e2_wind.png")
    """)
    code("""
        shots = read_csv("results/e5_shots.csv")
        print("E5 — sampled / exact MAE (km) on the same subset; lead is in days")
        display(shots.pivot(index=["shots", "noise"], columns="lead", values=["mae", "mae_exact", "n_rows"]))
    """)
    code("""
        sparse = read_csv("results/e9_sparse.csv")
        print("E9 — test MAE (km) as hotspot days are removed; lead is in days")
        display(sparse.pivot(index=["drop_frac", "model"], columns="lead", values="mae"))
    """)

    markdown("""
        ## Real hardware (E6)

        Yellowknife fire, `ibm_fez`: 88 circuits × 10,000 shots, Z basis → 15 Z/ZZ features (saved).

        **No IBM connection here.** The next cells replay it:
        1. Job record.
        2. Measured features × saved ridge weights (trained on simulation only) → forecast →
           MAE; must match the saved score within $10^{-9}$ km.
    """)
    code("""
        # The job record saved when the circuits ran on ibm_fez.
        job_metrics = read_json("hardware/e6_job_metrics.json")
        meta = read_json("results/e6_meta.json")
        jobs = read_json("hardware/jobs.json")
        job = next(row for row in jobs if row["job_id"] == job_metrics["job_id"])
        assert meta["job_id"] == job["job_id"]
        assert meta["backend"] == job["backend"]
        assert meta["shots"] == job["budget"]["shots"]
        display(pd.DataFrame([{
            "job_id": job_metrics["job_id"],
            "backend": meta["backend"],
            "shots_per_circuit": meta["shots"],
            "circuits": len(W),
            "qpu_seconds": job_metrics["usage"]["qpu_charge_time_seconds"],
        }]))
    """)
    code("""
        # E6 live recomputation from measured features and frozen weights.
        readout = read_json("results/qrc_z_readout.json")
        measured = load_array("hardware/e6_hw_features.npy")
        assert measured.shape == (len(W), len(readout["feature_names"]))
        assert readout["feature_names"] == names[:measured.shape[1]]
        assert np.isfinite(measured).all()
        assert np.all((W >= 0) & (W <= 1))
        assert len(readout["scaler"]["lo"]) == W.shape[-1]
        saved_predictions = read_csv("results/e6_predictions.csv", parse_dates=["date"])
        hardware_scores = read_csv("results/e6_hardware.csv")
        checks = []
        for lead in (1, 3):
            # Forecast = measured features x saved weights + intercept, clipped at 0.
            fit = readout["readouts"][str(lead)]
            predicted = np.maximum(
                measured @ np.asarray(fit["coefficients"]) + fit["intercept"],
                readout["prediction_clip_min"],
            )
            # Match each forecast to its true advance by date and sector.
            targets = saved_predictions.loc[saved_predictions.lead.eq(lead), ["date", "sector", "y"]]
            joined = case_rows.assign(yhat=predicted).merge(
                targets, on=["date", "sector"], how="left", sort=False, validate="one_to_one"
            )
            keep = joined.y.notna()
            reference = hardware_scores.loc[
                hardware_scores.source.eq("hardware") & hardware_scores.lead.eq(lead)
            ].iloc[0]
            assert int(keep.sum()) == int(reference.n_rows)
            recomputed = mae(joined.loc[keep, "y"], joined.loc[keep, "yhat"])
            difference = abs(recomputed - reference.mae)
            print(f"E6 +{lead * 24} h absolute MAE difference: {difference:.12e} km")
            assert difference < 1e-9
            checks.append({"lead_h": lead * 24, "recomputed_mae": recomputed,
                           "saved_mae": reference.mae, "absolute_difference": difference})
        display(pd.DataFrame(checks))
        display(hardware_scores)
        show_figure("e6_hardware.png")
    """, cell_id="e6-recompute")
    markdown("""
        **Takeaway:** hardware 0.258 / 0.979 km vs exact 0.272 / 1.001 km; both beat
        persistence. Forecasts survive noise. One fire, so not evidence hardware is better.
        (`feature_mae` = average gap from exact features.)
    """)

    markdown("""
        ## ZF015 / Yellowknife case

        ~20,000 people evacuated, August 2023. Map: rebuilt outlines, forecasts, official perimeter.

        Town lies in sector 3 (SE), but the closest fire approach is the south lobe, sector 4,
        ~25 km. Arrival-time estimates (constant speed) give 38–1,355 days → not reported.
    """)
    code("""
        location = read_json("results/e8_eta.json")
        print(f"{location['town']}: {location['distance_km']:.1f} km from the fire centre, sector {location['sector']}.")
        show_figure("e8_zf015_map.png")
    """)

    markdown(r"""
        ## More qubits and inputs (after freezing)

        Exploratory, after the study was frozen. Chart: 2022 validation.

        | Version | Adds |
        |---|---|
        | 16 qubits (ring) | 2 qubits per sector, neighbouring sectors linked |
        | 7 qubits | upslope + conifer share |
        | 10 qubits | + future regional wind, ISI, rain\* |

        \*Observed later weather = assumes a perfect forecast (fire's own hotspots removed).
        Table: 2023 test, post-hoc (2023 already used above).
    """)
    code("""
        show_figure("qubit_comparison.png")
        extra = read_csv("results/e13_metrics.csv")
        extra = extra.loc[extra.split.eq("test")]
        reference = metrics.loc[metrics.split.eq("test") & metrics.model.isin(["fbp", "persistence"])]
        rows = []
        for version, label in (("base", "5 qubits (submitted)"), ("A", "7 qubits (+terrain +fuel)"),
                               ("B", "10 qubits (+future weather)")):
            for kind, name in (("qrc", "quantum"), ("esn", "ESN"), ("raw", "ridge")):
                for row in extra.loc[extra.model.eq(f"{kind}_{version}")].itertuples():
                    rows.append({"version": label, "model": name, "lead_h": row.lead * 24, "mae": row.mae})
        for row in reference.itertuples():
            rows.append({"version": "reference", "model": row.model, "lead_h": row.lead * 24, "mae": row.mae})
        print("2023 test MAE (km), post-hoc")
        display(pd.DataFrame(rows).pivot_table(index=["version", "model"], columns="lead_h",
                                               values="mae", sort=False))
        paired = read_csv("results/e13_paired.csv")
        paired = paired.loc[paired.split.eq("test")].assign(lead_h=lambda d: d.lead * 24)
        print("Paired 2023 MAE differences: larger quantum model minus reference; negative favours it.")
        display(paired[["model", "reference", "lead_h", "diff", "lo", "hi"]].reset_index(drop=True))
    """)
    markdown("""
        **Takeaway:** only future weather helps (10 q beats 5 q). Ridge on the same inputs still
        wins. More qubits don't close the gap; perfect forecasts don't exist → best case.
    """)

    markdown("""
        ## Conclusions and limits

        - Quantum ≈ classical reservoirs; slightly behind raw ridge. No quantum advantage.
        - FBP: lowest MAE. Persistence: best IoU. Growing sectors: learned models win.
        - Beats persistence only at +72 h.
        - Robust to seeds (E4), wind (E2), missing days (E9).
        - Real hardware within ~0.02 km of exact (one fire).
        - Limits: one hardware case · 30 km/day cap can merge fires · no fuel-type input ·
          375 m–1 km satellite pixels.
    """)
    markdown("""
        ### Reproduce

        With the source data (README: 'Rebuild everything from the raw data'), run from the
        repository root with `.venv/bin/python -I`:

        1. `scripts/build_sectors.py`
        2. `scripts/run_experiments.py`
        3. `scripts/run_u5b.py`
        4. `scripts/make_figures.py`
        5. `scripts/e6_hardware.py`
        6. `scripts/build_notebook_explained.py` (this notebook) and `scripts/build_notebook.py` (short version)
    """)
    return nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    })


def main():
    start = perf_counter()
    notebook = build_notebook()
    NOTEBOOK.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, NOTEBOOK)
    # Use the invoking environment even when the installed kernelspec says 'python'.
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-I", "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    executor = ExecutePreprocessor(timeout=600, kernel_name="python3")
    # Execute in an empty directory: the notebook must not need any other file.
    with tempfile.TemporaryDirectory() as empty:
        try:
            executor.preprocess(notebook, {"metadata": {"path": empty}}, km=manager)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)
    for cell in notebook.cells:
        if cell.id == "e6-recompute":
            for output in cell.outputs:
                if output.output_type == "stream":
                    print(output.text, end="")
    print(f"Notebook executed in {perf_counter() - start:.2f} s")
    print(f"notebooks/ember_explained.ipynb: {NOTEBOOK.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
