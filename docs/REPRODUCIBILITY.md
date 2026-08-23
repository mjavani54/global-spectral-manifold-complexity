# Reproducibility guide

## Scope

The repository distinguishes verification from recomputation. Verification uses the frozen outputs under `results/`. Full recomputation requires raw inputs that are not committed.

## Stage 1: source-data audit

Required inputs:

- `data/raw/Datasets.zip`
- the earlier master-analysis archive containing `master_analysis_output/intrinsic_dimension/id_summary.csv`

Example:

```bash
python scripts/audit_paper2_inputs.py \
  --datasets-zip data/raw/Datasets.zip \
  --master-zip data/raw/master_analysis_output.zip \
  --output-dir build/audit
```

The audit checks archive integrity, array shapes, finite values, response bounds, sampled duplicates, covariance metrics, and design-response pairing sanity for the 2×2 and grating cases.

## Stage 2: response-manifold geometry

```bash
python scripts/paper2_geometry_analysis.py \
  --datasets-zip data/raw/Datasets.zip \
  --output-dir build/geometry \
  --common-dim 200 \
  --sample-sizes 1000 2500 5000 \
  --repeats 5 \
  --global-bootstrap 30 \
  --base-seed 240819
```

Cross-family spectra are resampled to 200 positions on each dataset's normalized `[0,1]` spectral coordinate. This equalizes output resolution; it does **not** align physical wavelength or frequency.

## Stage 3: neural-surrogate experiments

```bash
python scripts/paper2_dnn_learning_curves.py \
  --datasets-zip data/raw/Datasets.zip \
  --output-dir build/dnn \
  --mode all \
  --seeds 1 2 3 \
  --train-sizes 500 1000 2000 3000 4000 4800 \
  --capacity-train-size 4800 \
  --capacity-widths 32 64 128 256 512 \
  --fixed-width 512 \
  --hidden-layers 4 \
  --val-size 600 \
  --test-size 600 \
  --batch-size 256 \
  --max-epochs 600 \
  --patience 50 \
  --lr 0.001 \
  --weight-decay 0.000001 \
  --split-seed-base 991000 \
  --device auto
```

The script writes `dnn_results.csv` and `run_metadata.json`. Preserve both files from every GPU shard. The historical per-seed files used to create the frozen aggregate tables were not available in the supplied archive, so the exact aggregation cannot be independently reconstructed from this repository alone.

## Stage 4: postprocessing and robustness

```bash
python scripts/paper2_postprocess_dnn.py \
  --dnn-results build/dnn/dnn_results.csv \
  --geometry-results-dir build/geometry \
  --output-dir build/dnn_summaries

python scripts/paper2_robustness_analysis.py \
  --input build/dnn_summaries/geometry_learning_merged.csv \
  --output-dir build/robustness \
  --bootstrap-replicates 50000
```

Exact two-sided Spearman tests enumerate all `9!` outcome permutations. The bootstrap resamples the nine architecture-level cases and is descriptive because the cases are neither numerous nor a balanced random sample of device families.

## Stage 5: figures

Use `make figures` or the three explicit commands in the root README. Generated files go to `build/figures/`, leaving the frozen release figures unchanged.

## Expected nondeterminism

The geometry analysis is deterministic under the recorded seeds. PyTorch requests deterministic algorithms where available, but GPU models, driver versions, BLAS libraries, and floating-point reduction order can still prevent bitwise-identical DNN results.

