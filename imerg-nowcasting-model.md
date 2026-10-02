# IMERG ConvLSTM Nowcasting

## Goal
Build a reproducible research pipeline that uses six 30-minute IMERG frames to predict the next four frames over the Vietnam grid.

## Tasks
- [x] Inspect the prepared NetCDF schema, normalization metadata, window indices, and land weights. → Verify shapes and split counts match `dataset_metadata.json`.
- [x] Implement a lazy NetCDF window dataset. → Verify one sample has input shape `(6, 1, 165, 80)` and target shape `(4, 1, 165, 80)`.
- [x] Implement the autoregressive ConvLSTM encoder-decoder and weighted rainfall loss. → Verify a CPU forward/backward pass preserves the requested output shape.
- [x] Implement training, validation, checkpointing, AMP, and deterministic seeding. → Verify a limited one-epoch smoke run writes a loadable best checkpoint.
- [x] Implement persistence and trained-model evaluation with MAE, RMSE, CSI, and FSS. → Verify both baselines emit machine-readable test metrics.
- [x] Document local smoke testing and full Colab/Kaggle GPU training commands. → Verify commands and paths match the repository layout.
- [x] Run tests, lint, type checks, and end-to-end smoke training/evaluation. → Verify all checks pass or document environment limitations.

## Done When
- [x] The repository can train, validate, and test the model without loading an entire split into memory.
- [x] A CPU smoke run succeeds and the same CLI automatically uses CUDA on Colab/Kaggle.
- [x] Test output compares ConvLSTM against persistence in physical units (`mm/30 min`).

## Notes
- Confirmed choices: research/academic scope, full training on Colab/Kaggle GPU, ConvLSTM plus persistence baseline.
- Prepared data uses six input steps, four forecast steps, a 30-minute interval, and train-only `log1p` z-score statistics.
