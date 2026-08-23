# Global spectral manifold complexity and neural-surrogate learning difficulty

Reproducibility repository for the manuscript:

> **Global spectral manifold complexity predicts neural-surrogate learning difficulty in nanophotonics**

The study compares response-manifold geometry with fixed-protocol neural-surrogate learning difficulty across nine nanophotonic datasets: seven SiO2/TiO2 multilayers (8–20 independently variable layers), one 2×2 HfO2-on-SiO2 metasurface, and one TiO2-on-SiO2 grating.

## What this repository supports

This release contains the analysis and training code, frozen aggregate results, publication figures, audit outputs, and the current Optics Express manuscript. It supports three levels of checking:

1. **Immediate validation:** verify the released tables and the manuscript-level numerical claims without obtaining the raw spectra.
2. **Figure regeneration:** rebuild all manuscript figures from the frozen CSV/JSON outputs.
3. **Full rerun:** recompute geometry and retrain the neural surrogates after placing the original data archive in `data/raw/`.

The third level is not self-contained. The raw spectral datasets, raw per-seed DNN logs, exact wavelength vectors, Lumerical `.fsp` project files, mesh settings, boundary conditions, and material-database versions were not available for inclusion. The aggregate three-seed tables are included, but they are not a substitute for the missing per-seed logs. See [Known limitations](docs/KNOWN_LIMITATIONS.md).

## Main findings represented by the frozen results

- PCA99 versus learning difficulty: Pearson `r = 0.894`, `p = 0.00115`; Spearman `rho = 0.900`, exact `p = 0.00203`; leave-one-case-out `Q² = 0.580`.
- TwoNN versus learning difficulty: Pearson `r = 0.029`, `p = 0.942`; Spearman `rho = 0.300`, exact `p = 0.437`.
- PCA99/TwoNN versus learning difficulty: Spearman `rho = 0.983`, exact `p = 4.96e-05`; leave-one-out rho range `0.976–1.000`; leave-one-case-out `Q² = -0.032`.

These are exploratory architecture-level results, not a universal sample-complexity law. The study contains only nine cases and has a 7/1/1 device-family imbalance.

## Repository structure

| Path | Contents |
| --- | --- |
| `manuscript/` | Current Optics Express manuscript and manuscript notes |
| `scripts/` | Input audit, geometry analysis, DNN training, postprocessing, robustness analysis, and figure generation |
| `results/audit/` | Read-only source-data audit outputs |
| `results/geometry/` | Global and local spectral-geometry results |
| `results/dnn_summaries/` | Frozen aggregate three-seed learning and capacity summaries |
| `results/robustness/` | Exact permutation, leave-one-out, bootstrap, and partial-correlation outputs |
| `figures/` | Final PNG and PDF manuscript figures |
| `data/` | Expected raw-data layout and file manifest; raw data are not committed |
| `tests/` | Release-integrity tests for the frozen results |
| `docs/` | Detailed reproduction instructions, data dictionary, limitations, and release checklist |

## Installation

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA training, install the PyTorch build appropriate for the local CUDA driver before installing the remaining requirements. The generic `torch` entry may otherwise install a CPU build.

## Validate the frozen release

```bash
python scripts/validate_release.py
python -m unittest discover -s tests -v
```

## Recreate figures from frozen tables

```bash
python scripts/make_paper2_geometry_figures.py \
  --results-dir results/geometry \
  --output-dir build/figures

python scripts/make_frozen_dnn_figures.py \
  --summary-dir results/dnn_summaries \
  --output-dir build/figures

python scripts/make_revised_manuscript_figures.py \
  --merged-csv results/dnn_summaries/geometry_learning_merged.csv \
  --robustness-dir results/robustness \
  --geometry-dir results/geometry \
  --output-dir build/figures
```

The final manuscript numbering is Fig. 1 multilayer scaling, Fig. 2 cross-architecture dimensions, Fig. 3 intrinsic-dimension stability, Fig. 4 learning curves, Fig. 5 geometry-learning robustness, Fig. 6 capacity curves, and Fig. S1 predictor robustness.

## Full analysis

Place the original dataset archive at `data/raw/Datasets.zip`. Its internal member names must match [the expected-data manifest](data/expected_files.csv).

```bash
python scripts/paper2_geometry_analysis.py \
  --datasets-zip data/raw/Datasets.zip \
  --output-dir build/geometry

python scripts/paper2_dnn_learning_curves.py \
  --datasets-zip data/raw/Datasets.zip \
  --output-dir build/dnn \
  --mode all \
  --seeds 1 2 3

python scripts/paper2_postprocess_dnn.py \
  --dnn-results build/dnn/dnn_results.csv \
  --geometry-results-dir build/geometry \
  --output-dir build/dnn_summaries

python scripts/paper2_robustness_analysis.py \
  --input build/dnn_summaries/geometry_learning_merged.csv \
  --output-dir build/robustness
```

The fixed split seeds are `991001`, `991002`, and `991003` for training seeds 1–3. Full DNN reruns can be expensive and may not reproduce GPU-level floating-point behavior bit for bit across hardware and software versions.

## Citation

Use [CITATION.cff](CITATION.cff). Update the coauthor list, repository URL, and publication DOI before creating a public archival release.

## License

The source code is released under the MIT License. The manuscript, figures, and research data are not covered by that license; see [LICENSE_SCOPE.md](LICENSE_SCOPE.md).

