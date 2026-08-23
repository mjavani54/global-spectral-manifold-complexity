# Raw-data placement

Raw design and response arrays are intentionally not committed to GitHub. The largest CSV files exceed practical source-control limits, and the complete original archive should be deposited in a research-data repository before public release.

Place the archive at:

```text
data/raw/Datasets.zip
```

The analysis scripts read CSV members directly from the ZIP. The required member names and shapes are listed in `expected_files.csv`. Arrays are stored as `features × samples`: design variables or spectral coordinates occupy rows, and samples occupy columns.

The source archive audited for the manuscript had SHA-256:

```text
daf81fb01b3ece9105038d9d47d955616e2e5a8dbbbedb96c99f84b658efa3fb
```

This checksum identifies the analyzed archive; it is not a download location.

## Before public release

1. Deposit the original raw archive in Zenodo, an institutional repository, or another stable data service.
2. Add its DOI and access terms to the root `README.md`, `CITATION.cff`, and manuscript Data Availability statement.
3. Do not add guessed wavelength vectors or guessed labels for the four archived grating coordinates.
4. If Lumerical project files are later recovered, deposit them separately with the applicable Ansys license restrictions and simulator-version metadata.

