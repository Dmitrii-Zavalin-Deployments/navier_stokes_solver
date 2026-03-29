# tests/main_solver/test_physics.py

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.common.solver_config import SolverConfig
from src.main_solver import run_solver
from tests.helpers.solver_input_schema_dummy import create_validated_input
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy


def test_run_solver_floating_point_critical_trap(caplog):
    real_state = make_step4_output_dummy(nx=2, ny=2, nz=2)
    fully_hydrated_config = SolverConfig(ppe_tolerance=1e-6, ppe_max_iter=1, dt_min_limit=1e-6, ppe_max_retries=5)
    
    with patch("src.main_solver._load_simulation_context") as mock_load:
        mock_context = MagicMock(input_data=create_validated_input(), config=fully_hydrated_config)
        mock_load.return_value = mock_context
        real_state.iteration = 42
        real_state.ready_for_time_loop = True 

        with patch("src.main_solver.orchestrate_step1", return_value=real_state), \
             patch("src.main_solver.orchestrate_step2", return_value=real_state), \
             patch("src.main_solver.orchestrate_step3", side_effect=FloatingPointError("NaN detected")):
            
            with pytest.raises(RuntimeError, match="CRITICAL INSTABILITY"):
                run_solver("dummy.json")
            assert "Audit Failure: NaN detected" in caplog.text

def test_run_solver_telemetry_logging(caplog):
    real_state = make_step4_output_dummy(nx=2, ny=2, nz=2)
    with patch("src.main_solver._load_simulation_context") as mock_load:
        mock_load.return_value = MagicMock(
            config=SolverConfig(ppe_tolerance=1e-6, ppe_max_iter=1000, dt_min_limit=1e-6, ppe_max_retries=5),
            input_data=create_validated_input(nx=2, ny=2, nz=2)
        )
        real_state.iteration = 10
        real_state.ready_for_time_loop = True 
        
        def exit_immediately(state_in, context_in):
            state_in.ready_for_time_loop = False
            return state_in

        with patch("src.main_solver.orchestrate_step1", return_value=real_state), \
             patch("src.main_solver.orchestrate_step2", return_value=real_state), \
             patch("src.main_solver.orchestrate_step3", return_value=(None, 0.001)), \
             patch("src.main_solver.orchestrate_step4", side_effect=exit_immediately), \
             patch("src.main_solver.archive_simulation_artifacts", return_value="zip"):
            
            with caplog.at_level(logging.DEBUG):
                run_solver("dummy.json")
            assert any("AUDIT [Start]: Iteration 10" in record.message for record in caplog.records)