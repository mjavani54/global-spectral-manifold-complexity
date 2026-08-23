# Contributing

Contributions should preserve scientific traceability.

1. Open an issue describing the scientific or software problem.
2. Do not overwrite frozen release tables without documenting the source data, command, software environment, and random seeds.
3. Put regenerated outputs in `build/` during development. Replace files under `results/` or `figures/` only as part of a documented release.
4. Run `python scripts/validate_release.py` and `python -m unittest discover -s tests -v` before submitting a pull request.
5. Do not commit raw data, credentials, Lumerical license information, or machine-specific absolute paths.

