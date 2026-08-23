from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_release import validate  # noqa: E402


class FrozenReleaseTest(unittest.TestCase):
    def test_release_integrity(self) -> None:
        summary = validate()
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["cases"], 9)
        self.assertEqual(summary["learning_rows"], 54)
        self.assertEqual(summary["capacity_rows"], 45)


if __name__ == "__main__":
    unittest.main()

