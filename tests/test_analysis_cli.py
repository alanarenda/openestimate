"""Check analysis CLI dispatch without running expensive dataset analyses."""

import contextlib
import io
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class AnalysisCLITests(unittest.TestCase):
    def assert_dispatch(self, tokens, expected):
        calls = []
        compare_module = types.ModuleType("compare_models")
        compare_module.compare_models = lambda datasets, output: calls.append(
            ("compare", datasets, output)
        )
        ablations_module = types.ModuleType("ablations")
        ablations_module.run_ablations = lambda datasets, output: calls.append(
            ("ablations", datasets, output)
        )
        argv = ["run_analysis.py", "--datasets", *tokens, "--output_dir", "test-output"]
        with patch.object(sys, "argv", argv), patch.dict(
            sys.modules,
            {"compare_models": compare_module, "ablations": ablations_module},
        ), contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(ROOT / "analysis" / "run_analysis.py"), run_name="__main__")
        self.assertEqual(
            calls,
            [("compare", expected, "test-output"), ("ablations", expected, "test-output")],
        )

    def test_single_dataset(self):
        self.assert_dispatch(["nhanes"], ["nhanes"])

    def test_comma_separated_datasets(self):
        self.assert_dispatch(
            ["glassdoor,nhanes,pitchbook"], ["glassdoor", "nhanes", "pitchbook"]
        )

    def test_space_separated_datasets(self):
        self.assert_dispatch(
            ["glassdoor", "nhanes", "pitchbook"], ["glassdoor", "nhanes", "pitchbook"]
        )

    def test_mixed_separators_and_whitespace(self):
        self.assert_dispatch(
            ["glassdoor, nhanes", "pitchbook"], ["glassdoor", "nhanes", "pitchbook"]
        )


if __name__ == "__main__":
    unittest.main()
