#!/usr/bin/env python3
"""Validate repository structure and manuscript-level frozen numerical claims."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CASES = {
    "multilayer_8",
    "multilayer_10",
    "multilayer_12",
    "multilayer_14",
    "multilayer_16",
    "multilayer_18",
    "multilayer_20",
    "2x2",
    "grating",
}

REQUIRED = [
    "README.md",
    "CITATION.cff",
    "LICENSE",
    "LICENSE_SCOPE.md",
    "manuscript/Paper2_Optics_Express_Manuscript.docx",
    "results/audit/audit_report.json",
    "results/geometry/global_complexity_full.csv",
    "results/geometry/nonlinear_id_summary.csv",
    "results/dnn_summaries/learning_curve_summary.csv",
    "results/dnn_summaries/capacity_curve_summary.csv",
    "results/dnn_summaries/geometry_learning_merged.csv",
    "results/robustness/exact_spearman_tests.csv",
    "results/robustness/loocv_prediction_summary.csv",
]

EXPECTED_FIGURES = {
    "figure1_multilayer_geometry_scaling",
    "figure2_cross_architecture_dimensions",
    "figure3_intrinsic_dimension_stability_revised",
    "figure4_dnn_learning_curves",
    "figure5_geometry_learning_robustness",
    "figure6_dnn_capacity_curves",
    "figureS1_predictor_robustness",
}


def read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def assert_close(actual: float, expected: float, tolerance: float = 5e-10) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"Expected {expected}, found {actual}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest() -> None:
    path = ROOT / "release_manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        target = ROOT / entry["path"]
        if not target.is_file():
            raise AssertionError(f"Manifest target missing: {entry['path']}")
        if target.stat().st_size != entry["size_bytes"]:
            raise AssertionError(f"Manifest size mismatch: {entry['path']}")
        if sha256(target) != entry["sha256"]:
            raise AssertionError(f"Manifest hash mismatch: {entry['path']}")


def validate() -> dict[str, int | str]:
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            raise AssertionError(f"Required file missing: {relative}")

    merged = read_csv("results/dnn_summaries/geometry_learning_merged.csv")
    if {row["case"] for row in merged} != EXPECTED_CASES or len(merged) != 9:
        raise AssertionError("Merged table must contain exactly the nine documented cases")

    learning = read_csv("results/dnn_summaries/learning_curve_summary.csv")
    capacity = read_csv("results/dnn_summaries/capacity_curve_summary.csv")
    if len(learning) != 54:
        raise AssertionError(f"Expected 54 learning-curve rows, found {len(learning)}")
    if len(capacity) != 45:
        raise AssertionError(f"Expected 45 capacity-curve rows, found {len(capacity)}")
    if {int(row["seeds_completed"]) for row in learning + capacity} != {3}:
        raise AssertionError("Every frozen DNN summary point must contain three seeds")

    correlations = read_csv("results/dnn_summaries/geometry_learning_correlations.csv")
    pca = next(
        row
        for row in correlations
        if row["outcome"] == "learning_difficulty_1_minus_auc"
        and row["predictor"] == "common_pca99"
    )
    assert_close(float(pca["pearson_r"]), 0.8939712147735348)
    assert_close(float(pca["pearson_p"]), 0.0011504847697339973)
    assert_close(float(pca["spearman_rho"]), 0.9)

    exact = read_csv("results/robustness/exact_spearman_tests.csv")
    ratio = next(row for row in exact if row["column"] == "pca99_twonn_ratio")
    pca_exact = next(row for row in exact if row["column"] == "common_pca99")
    assert_close(float(ratio["spearman_rho"]), 0.9833333333333332)
    assert_close(float(ratio["exact_two_sided_p"]), 4.96031746031746e-05)
    assert_close(float(pca_exact["exact_two_sided_p"]), 0.0020282186948853615)

    loocv = read_csv("results/robustness/loocv_prediction_summary.csv")
    pca_cv = next(row for row in loocv if row["predictor"] == "PCA99 dimension")
    ratio_cv = next(row for row in loocv if row["predictor"] == "PCA99 / TwoNN gap")
    assert_close(float(pca_cv["loocv_q2"]), 0.5798439382117129)
    assert_close(float(ratio_cv["loocv_q2"]), -0.03200807737583666)

    audit = json.loads((ROOT / "results/audit/audit_report.json").read_text(encoding="utf-8"))
    twenty = next(row for row in audit["cases"] if row["case"] == "multilayer_20")
    if twenty["master_reproduced"] is not False:
        raise AssertionError("The documented 20-layer master-summary mismatch disappeared")
    if any(float(twenty["master_differences"][key]) != 0.0 for key in ("pca90", "pca95", "pca99", "pca995", "pca999")):
        raise AssertionError("The 20-layer PCA component counts no longer match the master summary")

    figure_stems = {path.stem for path in (ROOT / "figures").glob("*.png")}
    if figure_stems != EXPECTED_FIGURES:
        raise AssertionError(f"Unexpected PNG figure set: {sorted(figure_stems)}")
    for stem in EXPECTED_FIGURES:
        if not (ROOT / "figures" / f"{stem}.pdf").is_file():
            raise AssertionError(f"Missing PDF figure: {stem}.pdf")

    raw_files = [path for path in (ROOT / "data/raw").iterdir() if path.name != ".gitkeep"]
    if raw_files:
        raise AssertionError("Raw data must not be included in the GitHub release")

    forbidden = "/workspace/" + "scratch/" + "7131922b39f7"
    for path in (ROOT / "scripts").glob("*.py"):
        if forbidden in path.read_text(encoding="utf-8"):
            raise AssertionError(f"Machine-specific path remains in {path.name}")

    manuscript = ROOT / "manuscript/Paper2_Optics_Express_Manuscript.docx"
    with zipfile.ZipFile(manuscript) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
    for placeholder in ("[COAUTHORS]", "[Additional affiliations]", "[email]", "[Funding information"):
        if placeholder not in document_xml:
            raise AssertionError(f"Expected unresolved manuscript placeholder missing: {placeholder}")

    validate_manifest()
    return {
        "cases": len(merged),
        "learning_rows": len(learning),
        "capacity_rows": len(capacity),
        "figure_pairs": len(EXPECTED_FIGURES),
        "status": "passed",
    }


def main() -> None:
    try:
        summary = validate()
    except (AssertionError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Release validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
