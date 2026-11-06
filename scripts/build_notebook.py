#!/usr/bin/env python3
"""Build and execute the self-contained submission notebook.

Every input the notebook reads (saved results, hardware arrays, figures) is
embedded in it as one compressed, checksummed data cell, and the reservoir code
is copied in from ember/reservoir.py. The notebook is executed in an empty
temporary directory, so it is proven to run with no other files.

Run with .venv/bin/python -I scripts/build_notebook.py from the repository root.
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
NOTEBOOK = ROOT / "notebooks" / "ember.ipynb"
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

        Give Ember a wildfire that has burned for at least 3 days. For each of 8 directions
        around the fire, it predicts how many km the fire edge will move in the next 24 h and
        the next 72 h. The forecasting model is a 5-qubit quantum circuit (a *quantum
        reservoir*) followed by a simple weighted sum (*ridge regression*). We compare it with
        simple and classical methods on 2023 fires it never saw, and we ran it once on a real
        IBM quantum computer (`ibm_fez`).

        Data: CWFIS hotspots and fire perimeters, Natural Resources Canada, Canadian Forest Service.

        **Short answer.** The quantum model is about as good as similar classical models and
        does not beat them, so we claim no quantum advantage. Canada's physics-based spread model
        (FBP) has the smallest average error. On real hardware the forecasts came out almost the
        same as on a perfect simulator.

        ### How to run it

        This notebook runs on its own. It needs no other files, no data download and no IBM
        account. Install four packages, then run all cells (it takes a few seconds):

        `pip install numpy==2.5.4 pandas==3.1.1 qiskit==2.6.1 ipython`

        **What runs live and what is saved.** Training on all fires takes too long for a
        notebook, so the result tables, the hardware measurements and the figures were saved
        by the pipeline scripts and are stored in the next (collapsed) cell. The notebook then:

        - runs the full reservoir code and simulates the circuit live on real fire windows;
        - checks that two independent simulators give the same numbers;
        - recomputes the real-hardware forecasts from the saved `ibm_fez` measurements and checks
          they match the saved score. It does not connect to IBM.

        ### Sections

        | Section | Question it answers |
        |---|---|
        | Data and sector geometry | How does a fire become 8 numbers a day? What is predicted? |
        | Quantum reservoir | What is the circuit, which parts are learned, how were settings chosen? |
        | Baselines | What do we compare against? |
        | Test 2023 results | How good is it on unseen fires? |
        | Controls | Is the result robust to seeds, wind, shot count and missing data? |
        | Real hardware (E6) | Does it still work on a real, noisy quantum computer? |
        | ZF015 / Yellowknife case | What does one real fire look like? |
        | Conclusions and limits | What can and can't we claim? |

        ### Experiment numbers

        The project numbers its experiments E1–E10. They appear in file names and headings.

        | ID | What it tests | Where |
        |---|---|---|
        | E1 | Main 2023 test scores for every model | Test 2023 results |
        | E2 | Scores split by wind direction (downwind / crosswind / upwind) | Controls |
        | E3 | Search for the reservoir settings, on 2022 only | Quantum reservoir |
        | E4 | Different random circuits (seeds) | Controls |
        | E5 | Finite shots and simulated hardware noise | Controls |
        | E6 | Real `ibm_fez` run on one fire | Real hardware |
        | E7 | Reservoir features plus the raw inputs together | Test 2023 results (`qrc_raw` row) |
        | E8 | Yellowknife distance and direction | ZF015 / Yellowknife case |
        | E9 | Randomly removed satellite days | Controls |
        | E10 | Does the rebuilt fire shape match the official perimeter? | Data and sector geometry |

        ### Words used below

        | Word | Meaning |
        |---|---|
        | Sector / slice | One of 8 directions around the fire centre, numbered clockwise from north: 0 N, 1 NE, 2 E, 3 SE, 4 S, 5 SW, 6 W, 7 NW |
        | Radius $r_k(t)$ | Distance (km) from the fire centre to the furthest burned point in sector $k$ by the end of day $t$ |
        | Origin | The day $t$ a forecast is made from (end of that day) |
        | Lead $h$ | How far ahead: $h=1$ day (+24 h) or $h=3$ days (+72 h) |
        | Window | The inputs for one sector over the last 5 days, $t-4$ to $t$ |
        | MAE | Mean absolute error: average km the forecast misses by. Lower is better |
        | Skill | $1-\mathrm{MAE}_\text{model}/\mathrm{MAE}_\text{persistence}$. Above 0 beats persistence |
        | IoU | Overlap of two shapes divided by their union. 1 is a perfect match |
        | Train / val / test | 2018–2021 fit the weights, 2022 chooses settings, 2023 is scored once |
    """)
    cells.append(nbformat.v4.new_code_cell(
        "# Embedded data: a compressed copy of the saved result files (results/, hardware/,\n"
        "# figures/), so this notebook needs no other files. The next cell checks every\n"
        "# file against its SHA-256 checksum.\n"
        f'BUNDLE = "{bundle_literal()}"',
        metadata={"jupyter": {"source_hidden": True}, "collapsed": True}))
    cells[-1].id = "embedded-data"

    markdown("""
        ### Setup: unpack the saved files

        The cell above holds one long string, `BUNDLE`: every saved file, compressed and
        base64-encoded, each with its SHA-256 checksum. The next cell unpacks it into the
        dictionary `FILES` (file name → bytes) in memory. Nothing is written to disk. Each file
        is checked against its checksum, so a damaged copy stops the notebook.

        The small helpers read one file from `FILES`: `read_json`, `read_csv`, `load_array`
        (NumPy arrays), `show_figure` (saved PNG charts) and `mae`.
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

        ### From satellite hotspots to 8 numbers a day

        Satellites detect burning pixels ("hotspots") several times a day. For each fire:

        1. **Fire centre.** On day 1, take the largest group of hotspots linked within 2 km of
           each other. Its middle is the fire centre.
        2. **8 sectors.** Cut the area around the centre into 8 equal slices, like a pizza.
        3. **Daily radius.** In each sector, the radius $r_k(t)$ is the distance from the centre
           to the furthest point burned so far. It never shrinks.
        4. **Clean-up rules.** A new hotspot only counts if it is within 2 km of fire already
           burning, so a separate nearby fire is not mixed in. A radius can grow by at most
           30 km in one day.

        ### What is predicted

        A forecast is made at the end of day $t$ (the *origin*). The target in sector $k$ is how
        far the edge moves from now:
        $$y_k^{(h)}(t)=r_k(t+h)-r_k(t),\qquad h\in\{1,3\}\text{ days}.$$
        The first origin is fire day 3. The model sees the last 5 days, $t-4$ to $t$; days
        before day 1 are filled with zero advance and the first day's weather.

        ### The 5 inputs (per sector, per day)

        | Name | Meaning |
        |---|---|
        | `w_k` | Wind blowing toward this sector (km/h). Negative means the wind blows away from it |
        | `isi` | Initial Spread Index: Canadian fire-weather number for how fast fire could spread |
        | `dd_km` | Today's advance in this sector, $r_k(t)-r_k(t-1)$ (km) |
        | `bui` | Buildup Index: how much dry fuel there is deeper down (soil, logs) |
        | `rh` | Relative humidity (%) |

        The weather numbers come with the hotspot data. Each is the median over that fire's
        hotspots that day; a day with no reading repeats the previous day.

        ### Year split

        Years never mix: weights are fitted on 2018–2021, settings are chosen on 2022, and
        2023 is scored once at the end. In the table below, `sector_rows` counts
        (fire, day, sector) rows and `origin_days` counts (fire, day) forecast origins.
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

        Before trusting the targets, E10 compares each 2023 fire's final 8-sector shape with
        the official season-end perimeter, using IoU (overlap ÷ union, 1 = perfect).

        - **Eight-point outline:** the polygon through the 8 final radii, our model's view of
          the fire.
        - **Buffered hotspots:** every hotspot drawn as a 375 m disc (about one pixel) and
          merged. This shows how well the raw satellite data covers the fire.

        A median IoU of about 0.65 for 8 points (against 0.83 for the raw hotspots) means the
        8-sector shape is coarse but follows the real fire. The gap is the cost of describing
        a complex shape with 8 numbers.
    """)
    code("""
        rebuild = read_json("results/e10_2023.json")
        print(f"E10 median IoU, eight-point outline: {rebuild['median_iou']:.6f}")
        print(f"E10 median IoU, buffered hotspots: {rebuild['median_iou_buffered']:.6f}")
    """)

    markdown(r"""
        ## Quantum reservoir

        ### The idea

        A *reservoir* is a fixed, random system that turns an input history into many
        features. It is never trained. Only a simple read-out on top is learned. Picture a
        pond: the inputs are stones thrown in, the ripples are the features. You don't train
        the pond; you only learn how to read the ripples.

        ### One window through the circuit

        One window is one sector's 5 inputs over 5 days, each scaled to $[0,1]$ (the minimum and
        maximum come from training rows only). Qubit $i$ reads input $i$. The circuit starts in
        $|00000\rangle$ and, for each day from oldest to newest:

        1. **Encode:** turn qubit $i$ by $R_Y(\gamma\,g_i\,u_i)$, where $u_i$ is that day's
           input, $\gamma$ is the encoding scale and $g_i\in[0.5,1.5]$ a fixed random gain.
        2. **Link:** CZ gates on neighbours 0–1, 1–2, 2–3, 3–4 mix the qubits.
        3. **Scramble:** fixed random turns $R_Z(b_i)\,R_Y(a_i)$ on every qubit.

        The state carries from one day to the next, so older days still shape the final state:
        that is the reservoir's memory. Each window restarts at $|00000\rangle$.

        At the end we read **20 features**: $\langle Z_i\rangle$ for each qubit (5),
        $\langle Z_iZ_j\rangle$ for each pair (10) and $\langle X_i\rangle$ for each qubit (5).
        Ridge regression maps them to the forecast, separately for +24 h and +72 h. Negative
        forecasts are clipped to 0 because a radius never shrinks.

        | Part | Fixed or learned |
        |---|---|
        | Gains $g_i$, scramble angles $a_i, b_i$ | Fixed: drawn once from seed 7 |
        | Window length, $\gamma$, input order, ridge penalty $\lambda$ | Chosen once on 2022 (E3), then frozen |
        | Ridge weights | Learned on 2018–2021 |

        The next cell is the complete reservoir code, the same as `ember/reservoir.py`.
        `feature_matrix` computes the features in three ways: `exact` (Qiskit statevector),
        `numpy` (a fast pure-NumPy copy used for all training) and `sampled` (finite shots).
    """)
    cells.append(nbformat.v4.new_code_cell(reservoir_source()))
    cells[-1].id = "reservoir-code"

    markdown(r"""
        ### E3: how the settings were chosen

        E3 tried 18 settings and scored each on 2022 validation fires only; test fires never
        reached it.

        | Setting | Tried |
        |---|---|
        | `L`, days per window | 3, 5, 7 |
        | `gamma`, encoding scale $\gamma$ | $\pi/4$, $\pi/2$, $3\pi/4$ |
        | Input order | `w_k, isi, dd_km, bui, rh` or `w_k, dd_km, isi, bui, rh` |
        | `lam_1`, `lam_3`, ridge penalty | best of $10^{-6}$ to $10^{2}$, per setting and lead |

        Input order matters because CZ only links neighbouring qubits, so the order decides
        which inputs mix directly. Each setting is ranked by skill against persistence,
        averaged over the two leads. Skill is used rather than raw MAE because a different
        window length gives a different set of windows; persistence on the same rows makes the
        comparison fair.

        The top five are shown below. All six best settings use 5-day windows. Every skill is
        slightly negative: even the best setting is a little behind persistence on 2022. The
        top settings are very close, so the exact choice matters little.
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
        ### The frozen settings and one real circuit

        The next cell loads the E3 winner, builds the reservoir, and loads 88 saved windows
        from the Yellowknife fire ZF015 (11 days × 8 sectors), already scaled to $[0,1]$.
        It then prints the circuit for the first window and the 20 feature names in read-out
        order. In the drawing, each day is one block of input `Ry` gates, the CZ chain (dots
        joined by lines) and the fixed `Ry`/`Rz` gates.
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
        ### Check: two simulators agree

        All training uses the fast NumPy simulator. This cell computes the 20 features of two
        windows with it and with Qiskit's exact statevector, and stops if they differ by more
        than $10^{-10}$. A difference near $10^{-16}$ is ordinary rounding.
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

        Every model forecasts the same targets from the same rows. Names in brackets are the
        `model` names in the result tables.

        | Model | What it does | Why it is here |
        |---|---|---|
        | Persistence (`persistence`) | Predicts $h$ times today's advance | Simplest forecast; anything useful must beat it |
        | Mean (`mean`) | Predicts the average training advance, per lead | Floor for any learned model |
        | FBP physics (`fbp`) | Canada's Fire Behaviour Prediction spread rate turned into km per sector: $h\cdot\mathrm{ros}\cdot T_b/1000\cdot\max(0,w_k/\mathrm{ws})$ | Physics reference; it knows the fuel type, our inputs do not |
        | Ridge on raw inputs (`raw`) | The same ridge read-out fed the 5-day input window directly, no circuit | Does the circuit add anything? |
        | Classical reservoir (`esn5`, `esn20`) | A small random tanh network (5 or 20 units) plus ridge | Same reservoir idea without qubits |
        | Random circuit (`haar`) | Our circuit with random two-qubit gates instead of CZ | Does our circuit design matter? |
        | Reservoir + raw (`qrc_raw`, E7) | Ridge on reservoir features and raw inputs together | Do the two combine well? |

        Our model is `qrc`. In FBP, ros is the rate of spread (m/min) from the hotspot data,
        $T_b$ is burning minutes per day fitted on training fires, and $w_k/\mathrm{ws}$ is the
        share of the wind blowing toward the sector (calm wind gives zero).
    """)

    markdown("""
        ## Test 2023 results

        These are the main results (E1): every model, frozen, scored once on the 392 test fires
        of 2023, using exact noiseless circuit features.

        **How to read the first table.** One row per model and lead (`lead_h` 24 or 72).
        `mae` is the average miss in km, with its 95% interval in brackets. `skill` is against
        persistence (above 0 is better). `iou` is the overlap between the predicted and real
        8-sector shapes (higher is better). `mae_growing` is the MAE counting only sectors that
        really moved. Intervals come from resampling whole fires 1,000 times (fire bootstrap).
        The `haar` row averages 10 random circuits; its interval is the average of their
        intervals, not an ensemble interval.

        **How to read the second table.** Paired differences: `qrc` MAE minus another model's
        MAE on the same rows. Negative favours `qrc`. If the interval
        (`difference_lo` to `difference_hi`) does not cross 0, the gap is clear.
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
        ### What the numbers say

        - **Quantum ≈ classical.** `qrc` misses by 0.360 km at +24 h and 0.943 km at +72 h,
          close to the classical reservoir (`esn20`) and the random circuit (`haar`).
        - **The circuit does not beat plain ridge.** `qrc` is about 0.01 km (+24 h) and
          0.02 km (+72 h) behind ridge on the raw inputs, and both intervals sit above 0.
        - **It beats persistence only at +72 h,** by about 0.13 km with a clear interval.
          At +24 h, "same speed as today" is hard to beat.
        - **No model wins every score.** FBP has the smallest MAE. Persistence has the best
          IoU, because most sectors don't move and it predicts exactly 0 for them. On growing
          sectors every learned model beats both FBP and persistence.
    """)

    markdown("""
        ## Controls

        Each control asks whether the main result holds when one thing changes. All keep the
        frozen settings and read the saved tables.

        ### E4: does the random circuit matter?

        The fixed gains and angles come from a random seed. E4 rebuilds the reservoir with 5
        seeds (`qrc`) and the random-gate circuit with 10 seeds (`haar`), and scores each on
        2023. A small standard deviation (`mae_sd`, about 0.01 km at +24 h and 0.02 km at +72 h) means the result is not one
        lucky draw.
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
    markdown("""
        ### E2: downwind, crosswind and upwind sectors

        Fires grow mostly downwind, so E2 splits the test rows by how the wind meets each
        sector: downwind (wind toward the sector makes up more than half the wind speed),
        upwind (less than minus half), crosswind (in between, or calm). Columns are leads in
        days. Every model misses most downwind, where the edge moves most. FBP is lowest in
        every group, and `qrc` stays within about 0.02 km of `esn20` in each one.
    """)
    code("""
        wind = read_csv("results/e2_wind.csv")
        print("E2 — test MAE by wind bin (km); lead is in days")
        display(wind.pivot(index=["model", "wind_bin"], columns="lead", values="mae"))
        show_figure("e2_wind.png")
    """)
    markdown("""
        ### E5: finite shots and hardware noise

        A real quantum computer gives random measurement results, so each circuit is run many
        times (*shots*) and averaged. E5 takes 2,088 test windows (the 88 Yellowknife ones plus
        2,000 random ones) and computes the Z and ZZ features with 100 to 10,000 shots, either
        noise-free (`ideal`) or with a noise model of `ibm_fez` (`fake_fez`). `mae_exact` is
        the same rows with exact features.

        Few shots make the features noisy and the forecasts much worse (100 shots: 1.21 km
        against 0.36 km exact). At 10,000 shots the gap shrinks to under 0.05 km, which is why
        the hardware run used 10,000 shots.
    """)
    code("""
        shots = read_csv("results/e5_shots.csv")
        print("E5 — sampled / exact MAE (km) on the same subset; lead is in days")
        display(shots.pivot(index=["shots", "noise"], columns="lead", values=["mae", "mae_exact", "n_rows"]))
    """)
    markdown("""
        ### E9: missing satellite days

        Clouds and smoke hide fires from satellites. E9 removes a random 15%, 30% or 45% of
        each fire's hotspot days (never the first), carries the last values forward, rebuilds
        the sectors and rescores. The rows change with each drop level, so compare models
        within one level. The reservoir degrades about as much as the classical reservoir and
        less than persistence.
    """)
    code("""
        sparse = read_csv("results/e9_sparse.csv")
        print("E9 — test MAE (km) as hotspot days are removed; lead is in days")
        display(sparse.pivot(index=["drop_frac", "model"], columns="lead", values="mae"))
    """)

    markdown("""
        ## Real hardware (E6)

        We ran the Yellowknife fire (ZF015, 10–20 August 2023) on IBM's `ibm_fez`: 88 circuits
        (11 days × 8 sectors), 10,000 shots each, measured in the Z basis. The measured counts
        were turned into the 15 Z and ZZ features and saved. (X features need a second,
        rotated measurement, so this read-out uses only Z and ZZ.)

        **This notebook does not contact IBM.** It replays the saved measurements:

        1. The first cell shows the job record (job ID, backend, shots, quantum-computer time).
        2. The second cell feeds the measured features into the saved ridge weights, which were
           trained on exact simulation and never saw hardware data, clips at 0, matches each
           forecast to its true advance by date and sector, and recomputes the MAE. It stops
           if the result differs from the saved score by more than $10^{-9}$ km.

        The last table compares four sources on the same 88 windows: exact simulation, the
        `fake_fez` noise model, real `ibm_fez`, and persistence. `feature_mae` is the average
        gap between each source's features and the exact ones. This is one fire with 88
        related windows, so the comparison is descriptive.
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
        ### What the hardware shows

        The hardware features differ from the exact ones (larger `feature_mae` than
        `fake_fez`), but the forecasts barely change: real `ibm_fez` misses by 0.258 km at
        +24 h and 0.979 km at +72 h, against 0.272 and 1.001 km for the exact simulator, and
        all three beat persistence on this fire. This shows the forecasts survive real noise.
        It does not show that hardware is better: it is one fire, so the small gain may be
        chance.
    """)

    markdown("""
        ## ZF015 / Yellowknife case

        In August 2023 about 20,000 people were ordered to leave Yellowknife as fire ZF015
        approached. The map shows the dated 8-sector outlines rebuilt from hotspots, the
        forecasts, and the official season-end perimeter.

        The cell prints the distance from the fire centre to Yellowknife and the sector that
        contains it (E8). The town lies in sector 3 (SE), but the fire's closest approach is
        through its south lobe, sector 4, about 25 km away. A "days until it reaches town"
        estimate that assumes one sector keeps a constant speed gives tens to over a thousand
        days here, so it is not meaningful and is not reported.
    """)
    code("""
        location = read_json("results/e8_eta.json")
        print(f"{location['town']}: {location['distance_km']:.1f} km from the fire centre, sector {location['sector']}.")
        show_figure("e8_zf015_map.png")
    """)

    markdown("""
        ## Conclusions and limits

        - On 2023 test fires the quantum reservoir is about level with classical reservoirs
          and a random circuit, slightly behind ridge on the raw inputs. No quantum advantage
          is claimed.
        - No model wins every measure: FBP physics has the lowest mean error; persistence
          has the best outline overlap (most sectors do not move); on growing sectors
          all learned models beat FBP and persistence.
        - The reservoir beats persistence only at +72 h.
        - The result does not depend on a lucky circuit (E4), stays close to the classical reservoir in every wind group (E2),
          and degrades gently when satellite days are missing (E9).
        - On real `ibm_fez` hardware the forecasts survive the noise: MAE within about
          0.02 km of the exact simulation on one fire (88 related windows; descriptive only).
        - Limits: one hardware case; the 30 km/day cap can let two merging fires look like
          one fire growing in steps; the inputs exclude fuel type, which FBP uses; fire shapes
          come from satellite pixels 375 m to 1 km wide, so edges are rough.
    """)
    markdown("""
        ### Reproduce

        To rebuild the full experiment, provide the source data described in the
        README section 'Rebuild everything from the raw data', then run these scripts in order from the repository root:

        1. `scripts/build_sectors.py`
        2. `scripts/run_experiments.py`
        3. `scripts/run_u5b.py`
        4. `scripts/make_figures.py`
        5. `scripts/e6_hardware.py`
        6. `scripts/build_notebook.py`

        Run each with `.venv/bin/python -I`. The E6 script reads the saved hardware
        features. `scripts/build_notebook.py` rebuilds this notebook: it embeds the
        saved inputs and runs it in an empty folder to check that it needs no other files.
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
    print(f"notebooks/ember.ipynb: {NOTEBOOK.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
