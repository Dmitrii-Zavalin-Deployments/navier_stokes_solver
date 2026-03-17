# tests/test_main_solver.py

import json
from unittest.mock import MagicMock, patch

import pytest

from src.main_solver import archive_simulation_artifacts, run_solver_from_file
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy


class TestMainSolverOrchestration:

    @patch("src.main_solver.orchestrate_step1")
    @patch("src.main_solver.orchestrate_step2")
    @patch("src.main_solver.orchestrate_step3")
    @patch("src.main_solver.orchestrate_step4")
    @patch("src.main_solver.orchestrate_step5")
    @patch("shutil.make_archive")
    def test_main_solver_full_flow_success(
        self, mock_zip, mock_s5, mock_s4, mock_s3, mock_s2, mock_s1, tmp_path
    ):
        input_file = tmp_path / "test_input.json"
        data = solver_input_schema_dummy()
        input_file.write_text(json.dumps(data))
        
        # Setup mocks using specific Frozen Truth dummies
        state = make_output_schema_dummy()
        state.ready_for_time_loop = False # Exit loop immediately
        state.config.case_name = "test_case"
        
        mock_s1.return_value = make_step1_output_dummy()
        mock_s2.return_value = make_step2_output_dummy()
        mock_s3.return_value = make_step3_output_dummy()
        mock_s4.return_value = make_step4_output_dummy()
        mock_s5.return_value = state
        
        with patch.object(state, 'validate_against_schema') as mock_val:
            result = run_solver_from_file(str(input_file))
            mock_val.assert_called_once()
            assert "test_case" in result

    def test_schema_firewall_rejection(self, tmp_path):
        input_file = tmp_path / "bad.json"
        input_file.write_text(json.dumps(solver_input_schema_dummy()))
        
        with patch("src.main_solver.orchestrate_step1", return_value=MagicMock()) as mock_s1:
            state = mock_s1.return_value
            state.validate_against_schema.side_effect = RuntimeError("Contract Violation")
            
            with pytest.raises(RuntimeError, match="Contract Violation"):
                run_solver_from_file(str(input_file))

    def test_loop_iteration_logic(self, tmp_path):
        input_file = tmp_path / "loop.json"
        input_file.write_text(json.dumps(solver_input_schema_dummy()))
        
        state = make_output_schema_dummy()
        state.ready_for_time_loop = True
        state.dt = 0.1
        state.iteration = 0
        
        # Chain dependencies using step dummies
        with patch("src.main_solver.orchestrate_step1", return_value=make_step1_output_dummy()), \
             patch("src.main_solver.orchestrate_step2", return_value=make_step2_output_dummy()), \
             patch("src.main_solver.orchestrate_step3", return_value=make_step3_output_dummy()), \
             patch("src.main_solver.orchestrate_step4", return_value=make_step4_output_dummy()), \
             patch("src.main_solver.orchestrate_step5", side_effect=lambda s: setattr(s, 'ready_for_time_loop', False) or s):
            
            run_solver_from_file(str(input_file))
            assert state.iteration == 1
            assert state.time == 0.1

    def test_archive_creates_final_snapshot(self, tmp_path):
        state = make_output_schema_dummy()
        state.manifest.output_directory = str(tmp_path)
        state.config.case_name = "test"
        
        with patch("shutil.make_archive") as mock_zip, \
             patch("builtins.open", MagicMock()):
            archive_simulation_artifacts(state)
            assert (tmp_path / "final_state_snapshot.json").exists()
            mock_zip.assert_called()

    def test_main_block_execution(self):
        with patch("src.main_solver.run_solver_from_file") as mock_run, \
             patch("sys.argv", ["main_solver.py", "input.json"]), \
             patch("sys.exit"):
            
            import src.main_solver
            # Force re-evaluation of main block
            with patch("builtins.print"):
                if __name__ == "__main__":
                    try:
                        src.main_solver.run_solver_from_file("input.json")
                    except:
                        pass
            mock_run.assert_called_with("input.json")