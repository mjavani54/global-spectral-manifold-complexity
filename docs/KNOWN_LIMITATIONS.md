# Known limitations and unresolved provenance

1. **Nine architecture-level observations.** Seven are multilayers; only one 2×2 metasurface and one grating are included. Family-level generalization is not established.
2. **Normalized spectral coordinates.** Resampling controls output dimension but does not align the physical spectral intervals of the three device families.
3. **One surrogate family.** Learning difficulty is conditional on the documented four-hidden-layer MLP, optimizer, loss, split, early-stopping rule, and sample range.
4. **Missing raw DNN provenance.** Aggregate three-seed tables are available, but the historical per-seed `dnn_results.csv` files and `run_metadata.json` files are absent.
5. **Missing Lumerical projects.** The `.fsp` files, exact wavelength vectors, mesh settings, boundary conditions, and material-dispersion database versions were not supplied.
6. **Grating-coordinate ambiguity.** The dissertation describes five physical grating quantities, whereas the archived design matrix has four unlabeled columns. No one-to-one mapping is assigned.
7. **20-layer master-summary mismatch.** Recalculation from the audited 200,000-sample response array reproduces the PCA90/PCA95/PCA99/PCA995/PCA999 counts, but it does not exactly reproduce the earlier master-summary participation ratio, effective rank, or PC1 fraction. The recalculated values used in this release are recorded in `results/audit/audit_report.json`; the discrepancy should be resolved before claiming exact end-to-end provenance.
8. **One out-of-range 2×2 value.** One of 3,306,000 stored 2×2 response entries is 1.03293. It is retained because no physical or numerical basis for clipping was available.
9. **Threshold censoring.** Five cases do not reach `R² >= 0.99` within 4,800 training samples. Capacity curves are nonmonotonic for some difficult cases, so threshold summaries are secondary.

