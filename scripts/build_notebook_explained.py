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
    "results/metrics.csv", "results/qrc_comparisons.csv",
    "results/e6_meta.json", "results/e6_predictions.csv", "results/e6_hardware.csv",
    "results/qrc_z_readout.json", "results/e8_eta.json",
    "hardware/e6_job_metrics.json", "hardware/jobs.json",
    "hardware/e6_hw_features.npy", "hardware/zf015_case.npz",
    "figures/e1_test_mae.png",
    "figures/e6_hardware.png", "figures/e8_zf015_map.png",
    "figures/qubit_comparison.png", "results/e14_metrics.csv", "results/e14_selected.csv",
    "results/e14_paired.csv", "results/e14_meta.json", "results/qubit_windows.npz",
    "results/qubit_comparison_val.csv",
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
        - Quantum is close to classical reservoirs (worse than `esn20` at +24 h, no clear gap at +72 h); no quantum advantage claimed.
        - FBP (Canada's physics model) has the smallest error.
        - Real `ibm_fez` hardware: average error close to a perfect simulator; single forecasts noisier.

        **Run:** `pip install numpy==2.5.4 pandas==3.1.1 qiskit==2.6.1 ipython`, then run all.
        No other files, data download or IBM account needed.

        | Live in this notebook | Saved (too slow to rerun here) |
        |---|---|
        | Reservoir code, circuit simulation, simulator check, hardware forecasts recomputed from saved measurements | Training on all fires, result tables, figures, `ibm_fez` measurements |

        **Experiments**

        | ID | Tests | ID | Tests |
        |---|---|---|---|
        | E1 | 2023 test scores | E7 | reservoir + raw inputs |
        | E3 | settings search (2022) | E8 | Yellowknife distance |
        | E6 | real `ibm_fez` run | E10 | rebuilt shape vs CWFIS outline |

        Robustness checks (E2 wind, E4 seeds, E5 shots, E9 missing days) are saved in `results/`, not shown here.

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

        IoU of each 2023 fire's final shape vs the CWFIS season-end outline (buffered hotspots; median over fires).

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
        (Qiskit), `numpy` (fast copy, used for training), `sampled` (finite shots; 15 Z/ZZ features, needs `qiskit_aer`).
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
        # haar (random gates instead of CZ) is a design check, left out here.
        test = metrics.loc[metrics.split.eq("test") & metrics.model.ne("haar")].copy()
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
        - `qrc` 0.360 / 0.943 km vs `esn20` 0.350 / 0.934: `esn20` clearly better at +24 h, no clear gap at +72 h.
        - `raw` ridge beats `qrc` by 0.01–0.02 km (clear).
        - `qrc` beats persistence only at +72 h.
        - Best MAE: `fbp`. Best IoU: `persistence` (most sectors don't move). Growing sectors: learned models win.
    """)

    markdown(r"""
        ## More qubits and inputs (after freezing)

        Exploratory, after the study was frozen. Chart: 2022 validation.

        | Version | Adds |
        |---|---|
        | 7 qubits | upslope + conifer share |
        | 10 qubits | + future regional wind, ISI, rain\* |

        \*Observed later weather = assumes a perfect forecast (fire's own hotspots removed).
        Fair budget: quantum and ESN each try 3 settings × 5 seeds; best 5-seed mean on 2022 kept.
        Blue bars = seed 7, trained in this notebook below; lines = range over 5 seeds; ESN bars = 5-seed mean.
        Seed 7 is the best of 5 at 10 q; the fair comparison uses seed means (table). 2023 columns are post-hoc.

        **Same reservoir code, more qubits:** `ChainReservoir(n=7)` and `ChainReservoir(n=10)`.

        | Qubits | Inputs |
        |---|---|
        | q0–q4 | wind → sector, ISI, today's advance, BUI, humidity (as base) |
        | q5–q6 | upslope toward sector, conifer share |
        | q7–q9 | future wind → sector, ISI, rain at the lead (10 q only; one model per lead) |

        Terrain is read at each day's edge; future values are the same on all 5 days. Next cell builds both
        circuits and checks NumPy = Qiskit; the cell after trains and forecasts them.
    """)
    code("""
        # 7- and 10-qubit versions: same chain, same frozen gamma and seed, more qubits.
        rng = np.random.default_rng(0)
        for n in (7, 10):
            big = ChainReservoir(n=n, gamma=frozen["gamma"], seed=frozen["reservoir_seed"])
            window = rng.uniform(size=(frozen["L"], n))  # stand-in inputs in [0, 1]
            qc = big.circuit(window)
            diff = float(np.max(np.abs(big.features_numpy(window) - big.features_exact(window))))
            print(f"{n} qubits: {qc.count_ops()['cz']} CZ, depth {qc.depth()}, "
                  f"{len(feature_names(n))} features, max |NumPy - Qiskit| = {diff:.1e}")
            assert diff < 1e-10
        print("One day of the 10-qubit circuit:")
        print(big.circuit(window[-1:]).draw("text", fold=120))
    """, cell_id="more-qubit-code")
    markdown("""
        ### Train and forecast: 5, 7, 10 qubits (seed 7, $\\gamma=\\pi/4$)

        Saved inputs (`results/qubit_windows.npz`, from `scripts/export_qubit_windows.py`):
        2018–21 train, 2022 val, 88 Yellowknife windows, already scaled.
        Steps: reservoir features → ridge on train → $\\lambda$ picked on 2022 → forecast. ~1 min.
        Check: 2022 MAE must equal the E14 seed-7 row. Blue bars in the chart below = these 2022 numbers. Yellowknife = one fire, descriptive.
    """)
    code("""
        GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

        def ridge_fit(X, y, lam):
            # Same as ember/readout.py: centre, then solve (X'X + lam I) w = X'y.
            x_mean, y_mean = X.mean(axis=0), y.mean()
            Xc = X - x_mean
            coef = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (y - y_mean))
            return lambda Z: np.maximum(Z @ coef + (y_mean - x_mean @ coef), 0.0)  # no negative advance

        with load_array("results/qubit_windows.npz") as saved:
            daily, static, part = saved["daily"].astype(float), saved["static"].astype(float), saved["part"]
            targets = {1: saved["y1"].astype(float), 3: saved["y3"].astype(float)}
        e14_val = read_csv("results/e14_metrics.csv").query(
            "family == 'qrc' and seed == 7 and config == 'gamma=0.25pi' and split == 'val'")

        def inputs(version, h):
            if version == "base5":
                return daily[:, :, :5]                       # wind, ISI, advance, BUI, humidity
            if version == "A7":
                return daily                                 # + upslope, conifer share
            future = static[:, :3] if h == 1 else static[:, 3:]  # future wind, ISI, rain at lead h
            return np.concatenate([daily, np.repeat(future[:, None], daily.shape[1], axis=1)], axis=2)

        results = []
        for version in ("base5", "A7", "B10_h"):
            for h in (1, 3):
                W_v = inputs(version, h)
                X = ChainReservoir(n=W_v.shape[2], gamma=frozen["gamma"],
                                   seed=frozen["reservoir_seed"]).feature_matrix(W_v, mode="numpy")
                y = targets[h]
                ok = ~np.isnan(y)
                train, val, yk = ok & (part == 0), ok & (part == 1), ok & (part == 2)
                val_mae = {lam: np.mean(np.abs(y[val] - ridge_fit(X[train], y[train], lam)(X[val])))
                           for lam in GRID}
                lam = min(GRID, key=val_mae.get)             # first lowest, as select_lambda
                forecast = ridge_fit(X[train], y[train], lam)
                saved_val = e14_val.query("version == @version and lead == @h").mae.item()
                assert abs(val_mae[lam] - saved_val) < 1e-6, (version, h)
                results.append({"version": version, "qubits": W_v.shape[2],
                                "features": X.shape[1], "lead_h": 24 * h, "lambda": lam,
                                "val 2022 MAE": round(val_mae[lam], 3), "E14 saved": round(saved_val, 3),
                                "Yellowknife MAE": round(float(np.mean(np.abs(y[yk] - forecast(X[yk])))), 3),
                                "Yellowknife n": int(yk.sum())})
        display(pd.DataFrame(results))
    """, cell_id="more-qubit-train")
    code("""
        show_figure("qubit_comparison.png")
        bars = read_csv("results/qubit_comparison_val.csv")
        for row in results:  # chart blue bars = this notebook's seed-7 run
            h = row["lead_h"] // 24
            bar = bars.query("qubits == @row['qubits'] and lead == @h").quantum.item()
            assert abs(bar - row["val 2022 MAE"]) < 5e-4, row
        print("Chart blue bars match the seed-7 run above.")
        version_labels = {"base5": "5 qubits (base)", "A7": "7 qubits (+terrain +fuel)",
                  "B10_h": "10 qubits (+future weather)"}
        family_names = {"qrc": "quantum", "esn": "ESN", "raw": "ridge"}
        e14_selected = read_csv("results/e14_selected.csv")

        def span(row, split):
            mean, lo, hi = row[f"{split}_mean"], row[f"{split}_min"], row[f"{split}_max"]
            return f"{mean:.3f}" if row.n_seeds == 1 else f"{mean:.3f} [{lo:.3f}-{hi:.3f}]"

        e14_table = pd.DataFrame({"version": e14_selected.version.map(version_labels), "model": e14_selected.family.map(family_names),
                              "lead_h": e14_selected.lead * 24, "setting": e14_selected.config,
                              "val 2022": e14_selected.apply(span, axis=1, split="val"),
                              "2023 (post-hoc)": e14_selected.apply(span, axis=1, split="test")})
        print("MAE (km): mean [min-max] over 5 seeds, for the setting chosen on 2022")
        display(e14_table.sort_values("lead_h", kind="stable").reset_index(drop=True))
        e14_paired = read_csv("results/e14_paired.csv")
        e14_paired = e14_paired.loc[e14_paired.split.eq("test")].assign(
            version=lambda d: d.version.map(version_labels), lead_h=lambda d: d.lead * 24,
            comparison=lambda d: d.model.map(family_names) + " minus " + d.reference.map(family_names))
        print("Paired 2023 differences in seed-averaged error (km), 95% fire bootstrap; "
              "negative favours the first model.")
        display(e14_paired[["version", "lead_h", "comparison", "diff", "lo", "hi"]].reset_index(drop=True))
    """)
    markdown("""
        **Takeaway (seed means, fair):** ridge on raw inputs is best or within 0.0015 km of best everywhere. 5 q and 7 q: quantum loses
        to ridge and tuned ESN (both leads, both years). 10 q: ties ridge on 2022; on 2023 loses
        to ridge (+0.006 / +0.027 km) and beats ESN by a little (−0.003 / −0.008 km), the only
        quantum edge left. Seed spread is wide (10 q, 2022, +24 h: 0.276–0.307 km). Both picked
        the smallest setting tried (γ = π/4, input scale 0.3); a wider grid might move them.
        Perfect forecasts don't exist → 10 q is a best case.
    """)

    markdown("""
        ## Real hardware (E6)

        Yellowknife fire, `ibm_fez`: 88 circuits × 10,000 shots, Z basis → 15 Z/ZZ features (saved).
        Why 10,000 shots: in simulation, +24 h MAE 0.41 km vs 0.36 exact (100 shots: 1.21 km).

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
        persistence. Average error survives noise; single forecasts differ from exact by 0.07 / 0.24 km on
        average (max 0.57 / 1.37). One fire, so not evidence hardware is better.
        (`feature_mae` = average gap from exact features.)
    """)

    markdown("""
        ## ZF015 / Yellowknife case

        ~20,000 people evacuated, August 2023. Map: rebuilt outlines, forecasts, CWFIS season-end outline.

        Town lies in sector 3 (SE), but the closest fire approach is the south lobe, sector 4,
        ~25 km. Arrival-time estimates (constant speed) give 38–1,355 days → not reported.
    """)
    code("""
        location = read_json("results/e8_eta.json")
        print(f"{location['town']}: {location['distance_km']:.1f} km from the fire centre, sector {location['sector']}.")
        show_figure("e8_zf015_map.png")
    """)

    markdown("""
        ## Conclusions and limits

        - Quantum close to classical reservoirs (worse than `esn20` at +24 h); slightly behind raw ridge. No quantum advantage.
        - FBP: lowest MAE. Persistence: best IoU. Growing sectors: learned models win.
        - Beats persistence only at +72 h.
        - Real hardware: MAE within 0.014 / 0.022 km of exact; single forecasts differ by 0.07 / 0.24 km on average (one fire).
        - More qubits (post-hoc): 10 q ties ridge on 2022 and slightly beats tuned ESN; needs a perfect weather forecast.
        - Limits: one hardware case · no forecast for a fire's first 2 days · 30 km/day cap can merge fires ·
          no fuel-type input (base model) · 375 m–1 km satellite pixels.
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
        6. `scripts/e10_rebuild_check.py --year 2023` (E10 shape check)
        7. `scripts/e13_build_drivers.py` (terrain, fuel, weather inputs for the more-qubit versions)
        8. `scripts/e14_fair_baselines.py`, then `scripts/plot_qubit_comparison.py`
        9. `scripts/export_qubit_windows.py` (7/10-qubit inputs trained here)
        10. `scripts/build_notebook_explained.py` (this notebook)
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
