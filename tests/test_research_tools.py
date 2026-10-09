#!/usr/bin/env python3
"""Unit tests for sensitivity-analysis input parsing and summary statistics."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest
import xml.etree.ElementTree as ET

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("run_sensitivity", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
sensitivity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sensitivity)

CONV_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_convergence.py"
CONV_SPEC = importlib.util.spec_from_file_location("run_convergence", CONV_PATH)
assert CONV_SPEC is not None and CONV_SPEC.loader is not None
convergence = importlib.util.module_from_spec(CONV_SPEC)
CONV_SPEC.loader.exec_module(convergence)

RUN_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_research.py"
RUN_SPEC = importlib.util.spec_from_file_location("run_research", RUN_PATH)
assert RUN_SPEC is not None and RUN_SPEC.loader is not None
run_research = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(run_research)


class TerminalSnapshotTests(unittest.TestCase):
    def test_terminal_snapshot_uses_large_text_and_fits_canvas(self):
        root_dir = pathlib.Path(__file__).resolve().parents[1]
        data = json.loads((root_dir / "results" / "research_run.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temp_dir:
            output = pathlib.Path(temp_dir) / "results"
            output.mkdir()
            run_research.make_terminal_snapshot(output, data)
            root = ET.parse(output / "terminal_snapshot.svg").getroot()

        width = float(root.attrib["width"])
        texts = [node for node in root.iter() if node.tag.endswith("text")]
        body = [node for node in texts if float(node.attrib.get("y", "0")) >= 180]
        self.assertGreaterEqual(len(body), 10)
        self.assertIn("Stochastic Market Maker", "".join("".join(node.itertext()) for node in texts))
        for node in texts:
            y = float(node.attrib["y"])
            size = float(node.attrib.get("font-size", "0").replace("px", ""))
            self.assertGreaterEqual(size, 24 if y >= 180 else 18)
            estimated_width = len("".join(node.itertext())) * size * 0.62
            self.assertLessEqual(float(node.attrib["x"]) + estimated_width, width - 12)



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

    def test_convergence_counts_are_accepted(self):
        self.assertEqual(convergence.parse_counts("25,50,100,200"), [25, 50, 100, 200])

    def test_convergence_counts_reject_unsorted_values(self):
        with self.assertRaises(Exception):
            convergence.parse_counts("50,25,100")

    def test_convergence_counts_reject_duplicates(self):
        with self.assertRaises(Exception):
            convergence.parse_counts("25,25,50")

    def test_convergence_counts_reject_too_small(self):
        with self.assertRaises(Exception):
            convergence.parse_counts("1,25")

    def test_convergence_interval_shrinks_with_more_samples(self):
        small = convergence.ci95(1.0, 2.0, 25)
        large = convergence.ci95(1.0, 2.0, 100)
        self.assertGreater(small[1]-small[0], large[1]-large[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
