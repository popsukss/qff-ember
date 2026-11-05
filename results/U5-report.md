---
title: U5 experiment results
created: 2026-11-05
updated: 2026-11-05
agent: implementation
status: complete
---

U5 attempt 2 is implemented, with no git commit. All reservoir features are Exact noiseless expectation values computed with the NumPy statevector twin (mode='numpy').

## Files and execution

Code: `ember/experiments.py`, `scripts/run_experiments.py`.
Required outputs: `results/e3_val.csv`, `results/frozen_config.json`, `results/metrics.csv`, `results/e4_seeds.csv`, `results/e2_wind.csv`.
Additional evidence: `results/qrc_comparisons.csv`, `results/run_summary.json`, `results/verification.txt`, `results/test_mae.png`, `results/test_mae.html`, this report.
The ignored `results/run_experiments.log` contains the complete printed table and runtime locally.
Run command: `.venv/bin/python -I scripts/run_experiments.py`.
Runtime: **33.032 seconds**, measured from script start through completed metric/diagnostic computation and CSV writes, before summary serialization and final printing.

## Frozen E3 choice

All 18 train/validation configurations were evaluated. Selected L = 5, gamma = 0.7853981633974483 (pi/4), input map = ['w_k', 'isi', 'dd_km', 'bui', 'rh'], QRC lambda for leads 1 and 3 = 1e-6.
Mean validation skill across leads = **-0.070434910**; the best validation configuration still loses to persistence on that mean criterion.

| L | gamma | input_map | lam_1 | val_mae_1 | lam_3 | val_mae_3 | val_skill_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.785398 | ["w_k", "isi", "dd_km", "bui", "rh"] | 0.000001 | 0.302704 | 0.000001 | 0.766907 | -0.070435 |

Config SHA256: `9672868c35fa057b69fd5678cb010793f5b626710ccfe787faede371908c2c50`.
Frozen JSON SHA256 (present on every metrics row): `a0e215ee828bcb1a5d918d1db2f220c86a8d9ce538fd655a265c022c7a03caba`.

## Main test table

MAE and growing-row MAE are in km; skill is fractional relative MAE improvement over persistence. CIs use 1,000 fire bootstrap draws, seed 0, percentile 95% intervals, pooling rows in each sampled fire. Growing rows mean target > 0. IoU is averaged equally over complete eight-sector fire-day outlines using r(t)+y and r(t)+clipped prediction. Each lead drops its own missing targets. No zero-fill for incomplete outlines.

| model | lead | mae | mae_lo | mae_hi | skill | mae_growing | iou | n_rows | n_fires | lam |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | 1 | 0.348986 | 0.306802 | 0.388743 | 0.000000 | 1.705278 | 0.917901 | 77312 | 392 | NA |
| mean | 1 | 0.387043 | 0.359699 | 0.412127 | -0.109049 | 1.495680 | 0.749887 | 77312 | 392 | NA |
| fbp | 1 | 0.261797 | 0.232591 | 0.288843 | 0.249836 | 1.593038 | 0.781474 | 77312 | 392 | NA |
| raw | 1 | 0.348129 | 0.319236 | 0.374964 | 0.002455 | 1.424595 | 0.766076 | 77312 | 392 | 0.000001 |
| esn5 | 1 | 0.357794 | 0.330522 | 0.383042 | -0.025237 | 1.473719 | 0.757984 | 77312 | 392 | 0.000001 |
| esn20 | 1 | 0.350293 | 0.322225 | 0.376369 | -0.003746 | 1.429400 | 0.762824 | 77312 | 392 | 10.000000 |
| qrc | 1 | 0.359898 | 0.330036 | 0.387195 | -0.031268 | 1.455600 | 0.757289 | 77312 | 392 | 0.000001 |
| qrc_raw | 1 | 0.347457 | 0.317997 | 0.374092 | 0.004380 | 1.420634 | 0.765207 | 77312 | 392 | 0.000001 |
| persistence | 3 | 1.069785 | 0.933333 | 1.195464 | 0.000000 | 3.636969 | 0.832396 | 74000 | 392 | NA |
| mean | 3 | 0.999911 | 0.923320 | 1.069277 | 0.065316 | 2.430635 | 0.654111 | 74000 | 392 | NA |
| fbp | 3 | 0.694097 | 0.609309 | 0.771518 | 0.351181 | 2.689586 | 0.709615 | 74000 | 392 | NA |
| raw | 3 | 0.919704 | 0.839866 | 0.990446 | 0.140291 | 2.365133 | 0.672190 | 74000 | 392 | 0.000001 |
| esn5 | 3 | 0.946481 | 0.870765 | 1.017272 | 0.115261 | 2.413824 | 0.663596 | 74000 | 392 | 0.000001 |
| esn20 | 3 | 0.934348 | 0.857983 | 1.004096 | 0.126603 | 2.364160 | 0.668843 | 74000 | 392 | 10.000000 |
| qrc | 3 | 0.942642 | 0.863688 | 1.014536 | 0.118849 | 2.389401 | 0.662983 | 74000 | 392 | 0.000001 |
| qrc_raw | 3 | 0.920834 | 0.843282 | 0.990523 | 0.139235 | 2.360360 | 0.670616 | 74000 | 392 | 0.010000 |
| haar | 1 | 0.360574 | 0.332342 | 0.386820 | -0.033203 | 1.446067 | 0.760518 | 77312 | 392 | 2.320000 |
| haar | 3 | 0.945931 | 0.868405 | 1.016706 | 0.115775 | 2.380573 | 0.666119 | 74000 | 392 | 1.420000 |

Haar: **Arithmetic mean of per-seed metrics and CI endpoints over 10 entangler seeds; lam is the mean selected penalty, not an ensemble penalty.** These endpoints do not form an uncertainty interval for a prediction ensemble or capture between-seed uncertainty; E4 reports that spread separately.

## Does QRC beat the requested references?

At lead 1, QRC has higher MAE than persistence, raw and ESN20. At lead 3, QRC has lower MAE than persistence, but higher MAE than raw and ESN20. QRC never has the best test MAE; FBP has the lowest MAE at both leads. Persistence has the highest outline IoU at both leads.

Paired differences use the same fire draws for each model pair. Negative difference favors QRC. Lead-3 improvement over persistence has a CI wholly below zero. QRC's losses to raw at both leads and ESN20 at lead 1 have CIs wholly above zero. The lead-1 persistence comparison and lead-3 ESN20 comparison include zero. These are unadjusted pairwise intervals, not simultaneous intervals.

| lead | reference | qrc_minus_reference_mae | difference_lo | difference_hi |
| --- | --- | --- | --- | --- |
| 1 | persistence | 0.010912 | -0.004148 | 0.025118 |
| 1 | raw | 0.011769 | 0.009486 | 0.014071 |
| 1 | esn20 | 0.009605 | 0.005998 | 0.012999 |
| 3 | persistence | -0.127143 | -0.185652 | -0.067169 |
| 3 | raw | 0.022938 | 0.016652 | 0.028777 |
| 3 | esn20 | 0.008295 | -0.002300 | 0.017653 |

## E4 seed spreads

Haar uses entangler seeds 0..9 with fixed reservoir gain/angle seed 7. QRC uses gain/angle seeds 7, 11, 13, 17, 19. Each seed selects its own readout lambdas on validation; seeds are never selected by test performance. SD below is the sample SD (ddof=1).

| model | lead | n_seeds | mae_mean | mae_sd | mae_min | mae_max | skill_mean | skill_sd | skill_min | skill_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| haar | 1 | 10 | 0.360574 | 0.007301 | 0.346514 | 0.367935 | -0.033203 | 0.020920 | -0.054297 | 0.007084 |
| haar | 3 | 10 | 0.945931 | 0.015459 | 0.920181 | 0.962055 | 0.115775 | 0.014451 | 0.100703 | 0.139845 |
| qrc | 1 | 5 | 0.362390 | 0.008139 | 0.354012 | 0.373012 | -0.038409 | 0.023321 | -0.068845 | -0.014401 |
| qrc | 3 | 5 | 0.949891 | 0.016248 | 0.938782 | 0.977933 | 0.112073 | 0.015188 | 0.085861 | 0.122457 |

## E2 test wind bins

Wind is measured at forecast day t. cos = w_k/ws; >0.5 is downwind, <-0.5 is upwind, otherwise crosswind (including ws=0). Skill uses persistence on the identical lead/bin rows.

| model | lead | wind_bin | mae | skill | n_rows |
| --- | --- | --- | --- | --- | --- |
| qrc | 1 | downwind | 0.457917 | -0.030040 | 25837 |
| qrc | 1 | crosswind | 0.334649 | -0.029260 | 25633 |
| qrc | 1 | upwind | 0.286944 | -0.035576 | 25842 |
| qrc | 3 | downwind | 1.069567 | 0.178599 | 24731 |
| qrc | 3 | crosswind | 0.908947 | 0.095103 | 24535 |
| qrc | 3 | upwind | 0.849157 | 0.058853 | 24734 |
| persistence | 1 | downwind | 0.444562 | 0.000000 | 25837 |
| persistence | 1 | crosswind | 0.325135 | 0.000000 | 25633 |
| persistence | 1 | upwind | 0.277086 | 0.000000 | 25842 |
| persistence | 3 | downwind | 1.302125 | 0.000000 | 24731 |
| persistence | 3 | crosswind | 1.004476 | 0.000000 | 24535 |
| persistence | 3 | upwind | 0.902258 | 0.000000 | 24734 |
| fbp | 1 | downwind | 0.405191 | 0.088562 | 25837 |
| fbp | 1 | crosswind | 0.208375 | 0.359112 | 25633 |
| fbp | 1 | upwind | 0.171419 | 0.381350 | 25842 |
| fbp | 3 | downwind | 0.919101 | 0.294153 | 24731 |
| fbp | 3 | crosswind | 0.607898 | 0.394810 | 24535 |
| fbp | 3 | upwind | 0.554627 | 0.385290 | 24734 |
| esn20 | 1 | downwind | 0.460195 | -0.035165 | 25837 |
| esn20 | 1 | crosswind | 0.322775 | 0.007260 | 25633 |
| esn20 | 1 | upwind | 0.267708 | 0.033843 | 25842 |
| esn20 | 3 | downwind | 1.076083 | 0.173595 | 24731 |
| esn20 | 3 | crosswind | 0.898927 | 0.105078 | 24535 |
| esn20 | 3 | upwind | 0.827765 | 0.082563 | 24734 |

## Leakage controls and verification

Training: 2018--2021; validation: 2022; test: 2023. E3 is explicitly rejected if it receives any test row. Test files are loaded only after the E3 choice is written and successfully reloaded with its configuration-byte hash guard. Training windows alone fit the scaler. Training targets alone fit ridge, mean and per-lead FBP burn minutes. The accepted select_lambda helper selects penalties from train/val; all reported scores use nonnegative clipped predictions. QRC uses its frozen E3 penalties. Baseline and E4 penalties use validation independently of test. Test is evaluated once for each fixed model/seed; E2 and seed-7 E4 reuse the main-run results. No post-test model tuning or test-selected seed is used.

.venv/bin/python -I -m pytest tests/test_experiments.py -q -k 'not real'
exit 0: 13 tests passed.

.venv/bin/python -I -m pytest -q
exit 0: 92 tests passed (72 at 78%, 20 at 100%); two existing Qiskit SamplerV2 deprecation warnings from ember/hardware.py:80.

Synthetic adversarial audit:
PASS: perturbing only test inputs and targets leaves scaler fits, ridge fits, FBP burn-minute fits, lambda selection inputs and penalties, and all validation scores identical.
PASS: lead-specific missing-target filtering, no-growing-row MAE = NaN, strict wind threshold equality = crosswind, and calm wind = crosswind.


The scope-gate output is appended to `results/verification.txt`. No accepted module, schema, configuration, data, hardware code, or test file was edited.
