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
    fully_hydrated_config = SolverConfig(ppe_tolerance=1e-6, ppe_max_iter=1, dt_min_limit=1e-6, ppe_max_retries=1)
    
    with patch("src.main_solver._load_simulation_context") as mock_load:
        mock_context = MagicMock(input_data=create_validated_input(), config=fully_hydrated_config)
        mock_load.return_value = mock_context
        real_state.iteration = 42
        real_state.ready_for_time_loop = True 
        it = iter([FloatingPointError('NaN'), FloatingPointError('Trap Reached')])

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
            config=SolverConfig(ppe_tolerance=1e-6, ppe_max_iter=1000, dt_min_limit=1e-6, ppe_max_retries=1),
            input_data=create_validated_input(nx=2, ny=2, nz=2)
        )
        real_state.iteration = 10
        real_state.ready_for_time_loop = True 
        it = iter([FloatingPointError('NaN'), FloatingPointError('Trap Reached')])
        
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

def test_run_solver_elastic_success_signal():
    """
    Forensic Audit: Validates Line 120 of src/main_solver.py.
    Ensures that a successful iteration triggers the 'stabilization(is_needed=False)' 
    signal to the Elasticity Engine.
    """
    # 1. Prepare real dummies for state and input
    real_state = make_step4_output_dummy(nx=2, ny=2, nz=2)
    real_input = create_validated_input()
    
    # 2. Explicitly create the config dummy (since state doesn't have it)
    # This matches the expected SolverConfig structure
    real_config = SolverConfig(
        ppe_tolerance=1e-6,
        ppe_max_iter=10,
        dt_min_limit=1e-6,
        ppe_max_retries=1
    )
    
    with patch("src.main_solver._load_simulation_context") as mock_load:
        mock_context = MagicMock()
        mock_load.return_value = mock_context
        
        # 3. Correctly hydrate the mock context
        mock_context.input_data = real_input
        mock_context.config = real_config
        
        # Ensure the loop triggers
        real_state.ready_for_time_loop = True 
        it = iter([FloatingPointError('NaN'), FloatingPointError('Trap Reached')])
        
        def exit_immediately(state_in, context_in):
            # Force the loop to terminate after the first pass
            state_in.ready_for_time_loop = False
            return state_in

        # 4. Patch the ElasticManager and Orchestrators
        with patch("src.main_solver.ElasticManager") as mock_elastic_cls, \
             patch("src.main_solver.orchestrate_step1", return_value=real_state), \
             patch("src.main_solver.orchestrate_step2", return_value=real_state), \
             patch("src.main_solver.orchestrate_step3", return_value=(None, 0.000001)), \
             patch("src.main_solver.orchestrate_step4", side_effect=exit_immediately), \
             patch("src.main_solver.archive_simulation_artifacts", return_value="mock.zip"):
            
            # Setup the mock instance behavior
            mock_elastic_instance = mock_elastic_cls.return_value
            mock_elastic_instance.dt = 0.01 
            
            # Execute the solver
            run_solver("dummy.json")
            
            # 5. VERIFY: The success signal must be sent to the elasticity engine
            mock_elastic_instance.stabilization.assert_called_with(is_needed=False)

def test_run_solver_floating_point_critical_trap(caplog):
    """
    Forensic Audit: Validates Line 145-147 of src/main_solver.py.
    Ensures that a FloatingPointError (NaN/Inf) triggers the NUMERICAL CRITICAL
    log and terminates the process immediately.
    """
    # 1. Setup real state/input to pass ValidatedContainer checks
    real_state = make_step4_output_dummy()
    real_input = create_validated_input()
    real_state.ready_for_time_loop = True
        it = iter([FloatingPointError('NaN'), FloatingPointError('Trap Reached')])

    # 2. Configure the Safety Ladder (src/common/elasticity.py:30)
    # These values ensure the linear interpolation in __init__ doesn't fail.
    safe_config = SolverConfig(
        ppe_max_iter=1,
        ppe_tolerance=1e-6,
        dt_min_limit=1e-6,      # Sets the ladder floor
        ppe_max_retries=1       # Sets the ladder steps
    )

    # 3. Mock the loader and orchestrator
    with patch("src.main_solver._load_simulation_context") as mock_load, \
         patch("src.main_solver.orchestrate_step1", return_value=real_state), \
         patch("src.main_solver.orchestrate_step2", return_value=real_state), \
         patch("src.main_solver.orchestrate_step3", side_effect=lambda *args, **kwargs: next(it)):
        
        mock_context = MagicMock()
        mock_load.return_value = mock_context
        mock_context.input_data = real_input
        mock_context.config = safe_config

        # 4. Trigger the trap
        with pytest.raises(FloatingPointError):
            run_solver("dummy.json")

        # 5. Verify the Forensic Log (aligned with main_solver.py:146)
        assert "NUMERICAL CRITICAL: Floating point trap sprung" in caplog.text

def test_run_solver_value_error_contract_violation(caplog):
    """
    Forensic Audit: Validates Line 149-151 of src/main_solver.py.
    """
    real_state = make_step4_output_dummy()
    real_input = create_validated_input()
    real_state.ready_for_time_loop = True
        it = iter([FloatingPointError('NaN'), FloatingPointError('Trap Reached')])

    safe_config = SolverConfig(
        ppe_max_iter=1,
        ppe_tolerance=1e-6,
        dt_min_limit=1e-6,
        ppe_max_retries=1
    )

    with patch("src.main_solver._load_simulation_context") as mock_load, \
         patch("src.main_solver.orchestrate_step1", return_value=real_state), \
         patch("src.main_solver.orchestrate_step2", return_value=real_state), \
         patch("src.main_solver.orchestrate_step3", side_effect=ValueError("Illegal Stencil State")):
        
        mock_context = MagicMock()
        mock_load.return_value = mock_context
        mock_context.input_data = real_input
        mock_context.config = safe_config

        with pytest.raises(ValueError, match="Illegal Stencil State"):
            run_solver("dummy.json")

        assert "🚫 CONTRACT VIOLATION: Illegal Stencil State" in caplog.text