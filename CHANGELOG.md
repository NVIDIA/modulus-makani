# Changelog

## 0.3.0

**Breaking**
- Python >= 3.11, torch >= 2.9.1, torch-harmonics >= 0.9.0, PhysicsNeMo >= 1.3.0 (1.x and 2.x), DALI 2.x (`nvidia-dali-cuda120`), zarr >= 3. Dropped numba, pynvml, progressbar2, jsbeautifier. New `test`, `data_process`, `legacy_data_process` extras.
- Dataloaders emit dict batches (`input`, `target`, `*_dynamic_features`, `*_timestamp`; `(B, T, C, *grid)`). Timestamps are always present, and `return_timestamp` is gone.
- `Rollout` owns the multistep loop, and `StepWrapper` replaces `SingleStepWrapper`/`MultiStepWrapper`. The preprocessor is stateless between steps. `get_model(..., multistep=)` is removed. `continue_noise` is replaced by `Rollout(noise_state=...)`.
- Old DALI helpers (`data_loader_dali_2d`, `dali_es_helper_*`) were replaced by `dali_dataloader` + `sample_source`. Removed `img_utils`, `factorizations`, `drift_regularization`, `DistributedAFNO2Dv2`, and the fin/fout matmul split, which is now plain fork-join.
- SFNO: no MLP output bias before instance norms (dropped on load); the frequency `pos_embed` has a full `m = 0` column. Old checkpoints of the latter need conversion, and runs from before cannot resume their optimizer state.
- Invariants are stored in a single file; the model package renamed `land_sea_mask.nc` to `land_mask.nc` and normalizes its file names. Cosine zenith calculation uses an optimized vectorized routine.

**Data**
- Pluggable storage backends: makani HDF5, concatenated HDF5, makani zarr, ARCO/WB2 zarr, and native ICON mesh. Datasets declare and verify their grid. Shared channel mapping and metadata rework.
- Subsampling/resampling, NaN targets with masked losses and imputation, leap-year-safe concatenation, dataloader save/restore, GDS and O_DIRECT IO.
- Conservative remapping (raster and mesh to lat/lon), serial and spatially distributed.
- Converters: WB2 and NSF NCAR ERA5 merged into one tool with pluggable sources, NCAR downloader, h5_convert fixes, improved stats (Welford, power spectra).

**Models**
- Model package rework built on `Rollout`.
- Stochastic MLP with log-std spread, and a reworked noise module that is compilable and has a save/restore state.

**Training**
- Multistep weighting modes and multistep checkpointing. The ensemble is folded into the batch dimension. Validation streams one lead time at a time.
- Losses: MMD, kernel score, corrected and memory-efficient energy score, and relative losses. Fixes to CRPS gradients, the coherence loss, and the energy-score spread at E=1. Losses are AMP-safe.
- `torch.compile` and inductor support (step-level compilation), FP8/TE autocast options, weight-decay param groups, gradient-norm tracking, and an LR warmup start.
- Checkpoints carry a parameter-name checksum. Fixes to the distributed instance norm and to the parallel comm routines, which moved from PhysicsNeMo to TH/makani.

**Inference and tooling**
- Streaming rollout buffer (GDS, O_DIRECT), WB2-compliant output, and an inferencer built on `Rollout`.
- Visualization rewrite: op registry instead of eval, rank 0 only, progress bars on videos.
- Benchmark tool, profiling/NVTX fixes, pre-commit and a style CI, CI on pull requests, API docs, SECURITY.md, and broad test coverage, including distributed tests.
