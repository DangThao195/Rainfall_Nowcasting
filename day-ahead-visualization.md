# Day-ahead rolling visualization

## Goal

Add a separate command that reads one complete IMERG day and rolls the existing
6-to-4 ConvLSTM forward to produce all 48 half-hour frames of the following day.
The existing single-window visualizer must remain unchanged.

## Tasks

- [x] Define exact day selection and reject incomplete or discontinuous days.
- [x] Implement 48-step autoregressive rollout in a new module.
- [x] Render observed/predicted snapshots, daily accumulation, and a time series.
- [x] Save per-step and aggregate metrics to a separate JSON file.
- [x] Add tests, document the command, and run an end-to-end CPU check.

## Verification

- Unit tests cover day loading and rollout shape.
- Ruff, mypy, pytest, and the repository audit scripts pass.
- A smoke checkpoint produces all PNG and JSON artifacts on local CPU.
