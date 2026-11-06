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
    "results/fire_counts.json", "results/e10_2023.json", "results/frozen_config.json",
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

        From day 3 of a wildfire, Ember predicts how far each of eight compass-sector
        edges will advance, in kilometres, at +24 h and +72 h. A five-qubit quantum
        reservoir turns recent weather and spread history into fixed features; a
        ridge read-out learns the forecasts. Data: CWFIS hotspots and fire perimeters,
        Canadian Forest Service.

        **This notebook runs on its own.** It needs no other files, no data download
        and no IBM account. The saved results, hardware measurements and figures are
        embedded in the next (collapsed) cell, and the reservoir code is included below.
        It needs only these Python packages:

        `pip install numpy==2.5.4 pandas==3.1.1 qiskit==2.6.1 ipython`

        Then run all cells.
    """)
    cells.append(nbformat.v4.new_code_cell(
        "# Embedded data: a compressed copy of the saved result files (results/, hardware/,\n"
        "# figures/), so this notebook needs no other files. The next cell checks every\n"
        "# file against its SHA-256 checksum.\n"
        f'BUNDLE = "{bundle_literal()}"',
        metadata={"jupyter": {"source_hidden": True}, "collapsed": True}))
    cells[-1].id = "embedded-data"
    code("""
        import base64
        import gzip
        import hashlib
        import io
        import json

        import numpy as np
        import pandas as pd
        from IPython.display import Image, display

        FILES = {}
        for name, item in json.loads(gzip.decompress(base64.b64decode(BUNDLE))).items():
            data = base64.b64decode(item["b64"])
            assert hashlib.sha256(data).hexdigest() == item["sha256"], f"corrupted: {name}"
            FILES[name] = data

        pd.set_option("display.max_rows", 50)
        pd.set_option("display.max_columns", 12)
        pd.set_option("display.float_format", lambda value: f"{value:.6f}")

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

        Years stay separate: train on 2018–2021, choose settings on 2022 validation,
        and score 2023 test fires once. Counts below come from `results/fire_counts.json`.
        Each fire has eight sectors, numbered clockwise from north.

        The day-1 seed is the largest cluster of detections linked within 2 km.
        Later growth follows chains within 2 km of the existing fire; daily radial
        advance is capped at 30 km. The target in sector $k$ is
        $$y_k^{(h)}(t)=r_k(t+h)-r_k(t),\qquad h\in\{1,3\}\text{ days}.$$
        E10 checks the rebuilt final outline against the official final perimeter.
    """)
    code("""
        counts = pd.DataFrame.from_dict(read_json("results/fire_counts.json"), orient="index")
        counts.index.name = "year"
        display(counts[["split", "fires", "sector_rows", "origin_days"]])
        display(counts.groupby("split", sort=False)[["fires"]].sum())
        rebuild = read_json("results/e10_2023.json")
        print(f"E10 median IoU, eight-point outline: {rebuild['median_iou']:.6f}")
        print(f"E10 median IoU, buffered hotspots: {rebuild['median_iou_buffered']:.6f}")
    """)

    markdown(r"""
        ## Quantum reservoir

        E3 selected the window length, encoding scale, input map and ridge penalties
        on validation. The circuit gates stay fixed; only the read-out is trained.
        Training rows set the input min/max scaling to $[0,1]$.
        Each window starts in $|00000\rangle$ and carries its state through the days:
        input-dependent RY rotations, a CZ chain, then fixed RY/RZ rotations.
        The saved ZF015 windows are already scaled. The next cell is the complete
        reservoir code (the same as `ember/reservoir.py`). After it come the frozen
        choices, the first window's circuit, and all observable names in read-out order.
    """)
    cells.append(nbformat.v4.new_code_cell(reservoir_source()))
    cells[-1].id = "reservoir-code"
    code("""
        frozen = read_json("results/frozen_config.json")
        choices = {key: frozen[key] for key in ("L", "gamma", "input_map", "lambdas", "reservoir_seed")}
        print(json.dumps(choices, indent=2))
        reservoir = ChainReservoir(gamma=frozen["gamma"], seed=frozen["reservoir_seed"])
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
    code("""
        numpy_features = reservoir.feature_matrix(W[:2], mode="numpy")
        exact_features = reservoir.feature_matrix(W[:2], mode="exact")
        max_diff = float(np.max(np.abs(numpy_features - exact_features)))
        print(f"NumPy / exact feature max difference: {max_diff:.12e}")
        assert max_diff <= 1e-10
    """)

    markdown(r"""
        ## Baselines

        - Persistence: $h$ times today's sector advance.
        - Mean: the mean target advance in training rows, separately for each lead.
        - FBP physics: $h\cdot\mathrm{ros}\cdot T_b/1000\cdot\max(0,w_k/\mathrm{ws})$;
          $T_b$ is burning minutes per day fitted on train, ros is in m/min, and calm wind gives zero.
        - Raw-window ridge: ridge regression on the same raw input history.
        - ESN 5/20: fully connected tanh reservoirs with 5 or 20 units and a ridge read-out.
        - Random (Haar) circuit: the same encoding and read-out with random two-qubit entanglers.
        - Reservoir + raw: ridge on reservoir features together with the raw window.
    """)

    markdown("""
        ## Test 2023 results

        E1 uses exact noiseless features. MAE and growing-sector MAE are in km;
        growing means the target advance is positive. Skill is relative to persistence
        (positive is better). IoU measures predicted versus observed outline overlap
        (higher is better). Intervals are 95% fire-bootstrap intervals.
        The Haar row averages per-seed metrics and interval endpoints; it is not an
        ensemble confidence interval. E7 is the reservoir + raw row.
    """)
    code("""
        metrics = read_csv("results/metrics.csv")
        test = metrics.loc[metrics.split.eq("test")].copy()
        result_table = test[["model", "lead", "skill", "iou", "mae_growing"]].copy()
        result_table["lead"] *= 24
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
        ## Controls

        E4 varies fixed circuit seeds; the table gives mean and sample standard
        deviation of test MAE. E2 splits sectors by wind direction. E5 varies local
        simulator shots and noise on the saved subset, with its exact MAE alongside.
        E9 drops hotspot days and carries the last value forward. These checks use
        the committed tables and keep the selected configuration fixed.
    """)
    code("""
        seeds = read_csv("results/e4_seeds.csv")
        seeds = seeds.loc[seeds.split.eq("test")]
        print("E4 — fixed-seed test MAE (km)")
        display(seeds.groupby(["model", "lead"]).agg(
            seeds=("seed", "nunique"), mae_mean=("mae", "mean"), mae_sd=("mae", "std")
        ))
        show_figure("e4_seeds.png")

        wind = read_csv("results/e2_wind.csv")
        print("E2 — test MAE by wind bin (km); lead is in days")
        display(wind.pivot(index=["model", "wind_bin"], columns="lead", values="mae"))
        show_figure("e2_wind.png")
    """)
    code("""
        shots = read_csv("results/e5_shots.csv")
        print("E5 — sampled / exact MAE (km) on the same subset; lead is in days")
        display(shots.pivot(index=["shots", "noise"], columns="lead", values=["mae", "mae_exact", "n_rows"]))

        sparse = read_csv("results/e9_sparse.csv")
        print("E9 — test MAE (km) as hotspot days are removed; lead is in days")
        display(sparse.pivot(index=["drop_frac", "model"], columns="lead", values="mae"))
    """)

    markdown("""
        ## Real hardware (E6)

        The saved job ran Z-basis circuits on ibm_fez. The table below compares its
        features with exact simulation, local FakeFez simulation and persistence.
        All feature sources use the same frozen, exact-trained Z-only read-out.
        This is one fire with 88 correlated sector windows: the comparison is descriptive.

        For the local recomputation, targets come from `results/e6_predictions.csv`
        and are joined by date and sector to `hardware/zf015_case.npz` archive order.
        The saved scaler maps raw inputs before circuit encoding; the hardware file
        already contains measured Z/ZZ features, so these go directly into the saved
        ridge coefficients and intercept, followed by the saved lower clipping bound.
    """)
    code("""
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
            fit = readout["readouts"][str(lead)]
            predicted = np.maximum(
                measured @ np.asarray(fit["coefficients"]) + fit["intercept"],
                readout["prediction_clip_min"],
            )
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
        ## ZF015 / Yellowknife case

        The map compares dated reconstructed outlines and forecasts with the official
        season-end perimeter. Yellowknife's centre distance and sector below come from
        `results/e8_eta.json`. The single-sector time-to-reach is not meaningful here:
        the closest approach is through the south lobe, sector 4, about 25 km away.
        A single-sector time-to-reach is not reported.
    """)
    code("""
        location = read_json("results/e8_eta.json")
        print(f"{location['town']}: {location['distance_km']:.1f} km from the fire centre, sector {location['sector']}.")
        show_figure("e8_zf015_map.png")
    """)

    markdown("""
        ## More qubits and inputs (after freezing)

        After the study was frozen, we tried larger circuits. These runs are exploratory and
        do not change the results above. The chart shows 2022 validation MAE (lower is better);
        larger circuits that only copy the same 5 inputs onto extra qubits are left out.

        - **16 qubits (ring):** 2 qubits per sector, all 8 sectors linked to their neighbours.
        - **7 qubits (+terrain +fuel):** adds upslope toward the sector and conifer share.
        - **10 qubits (+future weather\\*):** also adds regional wind toward the sector, ISI and
          rain for the forecast days (+24 h: the next day; +72 h: the next three days).
          \\*This uses the weather that was later observed, so it assumes a perfect weather
          forecast; the fire's own hotspots are removed from it.

        Each group compares the quantum reservoir (seed 7; the thin line spans other seeds, where run), a
        classical reservoir (ESN) of the same size, and ridge on the same inputs. The table
        below gives the 2023 test scores. 2023 had already been used by the frozen study, so
        these are a post-hoc second look, not a new test.
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
        Future weather gives the only clear gain: the 10-qubit reservoir beats the submitted
        5-qubit one on 2023 (paired differences below 0). Ridge on the same 10 inputs still
        does better (paired differences above 0), so adding qubits does not close the gap to
        ridge. Terrain and fuel do not help the quantum reservoir. A perfect weather forecast
        is not available in practice, so the 10-qubit result is a best case.
    """)

    markdown("""
        ## Conclusions and limits

        - On 2023 test fires the quantum reservoir is about level with classical reservoirs
          and a random circuit, slightly behind ridge on the raw inputs. No quantum advantage is claimed.
        - No model wins every measure: FBP physics has the lowest mean error; persistence
          has the best outline overlap (most sectors do not move); on growing sectors
          all learned models beat FBP and persistence.
        - The reservoir beats persistence only at +72 h.
        - On real ibm_fez hardware the forecasts survive the noise: MAE within about
          0.02 km of the exact simulation on one fire (88 correlated windows; descriptive only).
        - Limits: one hardware case; the 30 km/day cap can creep for merged fires;
          inputs exclude fuel type, which FBP uses.
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
