.PHONY: validate test figures manifest

validate:
	python scripts/validate_release.py

test: validate
	python -m unittest discover -s tests -v

figures:
	python scripts/make_paper2_geometry_figures.py --results-dir results/geometry --output-dir build/figures
	python scripts/make_frozen_dnn_figures.py --summary-dir results/dnn_summaries --output-dir build/figures
	python scripts/make_revised_manuscript_figures.py --merged-csv results/dnn_summaries/geometry_learning_merged.csv --robustness-dir results/robustness --geometry-dir results/geometry --output-dir build/figures

manifest:
	python scripts/build_release_manifest.py

