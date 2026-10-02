# Local Test Visualization

## Goal
Add a CPU-friendly CLI that loads `best.pt` and renders ground truth, ConvLSTM
prediction, and `predicted - truth` error for the four IMERG forecast lead times.

## Tasks
- [x] Extract reusable checkpoint loading from the evaluation CLI. → Verify the smoke checkpoint reloads with its saved architecture.
- [x] Select a representative rainy test window or accept an explicit sample index. → Verify selection uses only target frames inside the Vietnam mask.
- [x] Render a shared-scale 4x2 truth/prediction figure and sample metrics JSON. → Verify both artifacts are created from local prepared data.
- [x] Document checkpoint placement and local CPU commands. → Verify paths match the Windows workspace.
- [x] Run tests, Ruff, MyPy, standard audits, and visually inspect the generated PNG. → Verify all checks pass.

## Done When
- [x] A downloaded `best.pt` can be tested without Colab or GPU.
- [x] The output clearly compares observed and predicted rainfall side by side for +30, +60, +90, and +120 minutes.
