#!/usr/bin/env python3
"""Unit tests for sensitivity-analysis input parsing and summary statistics."""
from __future__ import annotations

import importlib.util
import pathlib
import unittest

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("run_sensitivity", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
sensitivity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sensitivity)


class SensitivityToolsTests(unittest.TestCase):
    def test_parse_positive_grid(self):
        self.assertEqual(sensitivity.parse_values("0.01, 0.02,0.1", "gamma"), [0.01, 0.02, 0.1])

    def test_parse_rejects_empty(self):
        with self.assertRaises(Exception):
            sensitivity.parse_values("", "gamma")

    def test_parse_rejects_nonfinite(self):
        with self.assertRaises(Exception):
            sensitivity.parse_values("0.01,nan", "gamma")

    def test_parse_rejects_zero(self):
        with self.assertRaises(Exception):
            sensitivity.parse_values("0,0.1", "gamma")

    def test_parse_rejects_duplicate(self):
        with self.assertRaises(Exception):
            sensitivity.parse_values("0.1,0.1", "gamma")

    def test_small_sample_student_t_values(self):
        self.assertAlmostEqual(sensitivity.t_critical_95(2), 12.706, places=3)
        self.assertAlmostEqual(sensitivity.t_critical_95(10), 2.262, places=3)
        self.assertLess(sensitivity.t_critical_95(100), sensitivity.t_critical_95(10))

    def test_confidence_interval_has_expected_center_and_width(self):
        low, high = sensitivity.ci95(10.0, 2.0, 5)
        self.assertAlmostEqual((low + high) / 2.0, 10.0)
        self.assertAlmostEqual(high - low, 2.0 * 2.776 * 2.0 / (5 ** 0.5), places=8)

    def test_single_run_interval_is_degenerate(self):
        self.assertEqual(sensitivity.ci95(10.0, 3.0, 1), (10.0, 10.0))

    def test_pareto_frontier_removes_dominated_rows(self):
        rows = [
            {"qvi_mean_pnl": 1.0, "qvi_mean_rms_inventory": 3.0},
            {"qvi_mean_pnl": 2.0, "qvi_mean_rms_inventory": 2.0},
            {"qvi_mean_pnl": 0.0, "qvi_mean_rms_inventory": 4.0},
            {"qvi_mean_pnl": 3.0, "qvi_mean_rms_inventory": 4.0},
        ]
        result = sensitivity.pareto_rows(rows)
        self.assertEqual(
            {(row["qvi_mean_pnl"], row["qvi_mean_rms_inventory"]) for row in result},
            {(2.0, 2.0), (3.0, 4.0)},
        )

    def test_pareto_frontier_preserves_equal_metrics(self):
        rows = [
            {"qvi_mean_pnl": 1.0, "qvi_mean_rms_inventory": 2.0},
            {"qvi_mean_pnl": 1.0, "qvi_mean_rms_inventory": 2.0},
        ]
        self.assertEqual(len(sensitivity.pareto_rows(rows)), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
