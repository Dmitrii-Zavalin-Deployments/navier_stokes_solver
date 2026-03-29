# tests/main_solver/test_guards.py

import runpy
import sys
from unittest.mock import patch
import pytest
from src.main_solver import _load_simulation_context

def test_load_context_missing_input_file():
    with patch("src.main_solver.Path.exists") as mock_exists:
        mock_exists.return_value = False
        with pytest.raises(FileNotFoundError, match="Input file missing"):
            _load_simulation_context("non_existent_input.json")

def test_load_context_missing_config_file():
    with patch("src.main_solver.Path.exists") as mock_exists:
        mock_exists.side_effect = [True, False]
        with pytest.raises(FileNotFoundError, match="config.json required"):
            _load_simulation_context("valid_input.json")

def test_cli_entrypoint_no_args():
    with patch("sys.argv", ["src/main_solver.py"]), \
         patch("builtins.print") as mock_print:
        with pytest.raises(SystemExit) as e:
            runpy.run_module("src.main_solver", run_name="__main__")
        assert e.value.code == 1
        mock_print.assert_any_call("Usage: python src/main_solver.py <input_json_path>")

def test_cli_entrypoint_success():
    with patch("src.main_solver.run_solver", return_value="mock.zip"), \
         patch("sys.argv", ["src/main_solver.py", "valid.json"]), \
         patch("builtins.print") as mock_print:
        with pytest.raises(SystemExit) as e:
            runpy.run_module("src.main_solver", run_name="__main__")
        assert e.value.code == 0
        mock_print.assert_any_call("Pipeline complete. Artifacts archived at: mock.zip")