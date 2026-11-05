# Hardware preparation

Run the offline pilot from the project root:

```sh
.venv/bin/python -I scripts/hardware_pilot.py
```

The pilot samples four seeded random windows locally with Runtime SamplerV2 on
FakeFez. It prints CZ counts, depth, the 88-circuit budget and feature errors
against the exact reservoir. FakeFez uses bundled calibration data. The budget
reports two-qubit survival and shot uncertainty, not QPU minutes or a guarantee
that the job fits the Open Plan allowance.

The intended run is ZF015 (uid 242981), 10–20 August 2023 inclusive: 11 days ×
8 sectors, 88 circuits, 2000 shots each. Prepare a NumPy `.npz` archive with:

- `windows`: scaled inputs, shape `(88, 5, 5)`, oldest day first; columns are
  `w_k, isi, dd_km, bui, rh`. Use the frozen training-fitted scaler.
- `uid`: shape `(88,)`, all 242981.
- `dates`: shape `(88,)`, dates 2023-08-10 through 2023-08-20, each repeated eight times.
- `sectors`: shape `(88,)`, sectors 0 through 7 repeated for each date.

The archive must be ordered by date, then sector. Random pilot windows are never
used for submission. Explicit `--submit --windows PATH` requires an interactive
terminal and the exact word `SUBMIT`. A confirmed run creates one SamplerV2 job
in job mode with XY4 dynamical decoupling and gate twirling, then appends its ID,
budget, reservoir parameters, input checksum and circuit statistics to
`hardware/jobs.json`.

`fetch(job_id)` reuses the service from a confirmed submission in the same Python
process. For a later process, supply an already initialized service explicitly
as `fetch(job_id, service=service)`. Fetch checks completion before retrieving
results. Only `submit` constructs a Runtime service inside this module; importing
it and running the local pilot never initialize an account.
