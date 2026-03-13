# tests/integration/test_solver_lifecycle.py

import json
from pathlib import Path

from src.main_solver import run_solver


class TestSolverLifecycle:
    """
    INTEGRATION AUDITOR: Verifies the full pipeline flow 
    from Initialization (Step 1) to Archive (Step 5).
    """

    def test_full_solver_pipeline_integrity(self):
        """
        Rule 7 (STS): Atomic Verification of the end-to-end 
        solver lifecycle using the project root as the anchor.
        """
        project_root = Path(__file__).resolve().parent.parent.parent
        config_file = project_root / "config.json"
        input_file = project_root / "input_validated.json"
        
        # Save original state to restore after test
        orig_config = config_file.read_text() if config_file.exists() else None
        
        try:
            # 1. Write comprehensive test mocks that satisfy SolverInput.from_dict
            config_file.write_text(json.dumps({
                "ppe_tolerance": 1e-6, 
                "ppe_max_iter": 10, 
                "ppe_omega": 1.0
            }))
            
            # The mask must be a flat list of integers, not a dictionary
            # For a 4x4x4 grid, we provide 64 elements (4*4*4 = 64)
            input_file.write_text(json.dumps({
                "domain_configuration": {"type": "INTERNAL", "reference_velocity": [0.0, 0.0, 0.0]},
                "grid": {"nx": 4, "ny": 4, "nz": 4, "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0},
                "fluid_properties": {"density": 1.0, "viscosity": 0.1},
                "initial_conditions": {"velocity": [0.0, 0.0, 0.0], "pressure": 0.0},
                "simulation_parameters": {"time_step": 0.01, "total_time": 0.02, "output_interval": 1},
                "external_forces": {"force_vector": [0.0, 0.0, 0.0]},
                "mask": [0] * 64, 
                "boundary_conditions": {"conditions": []}
            }))
            
            # 2. Execution
            final_state = run_solver("input_validated.json")
            
            # 3. Assertions
            assert final_state.ready_for_time_loop is False
            assert len(final_state.manifest.saved_snapshots) > 0
            
            # 4. Archive Verification
            archive_path = project_root / "navier_stokes_test_case_output.zip"
            assert archive_path.exists(), "Archive service failed to produce output."
            
            # Cleanup
            archive_path.unlink()
            
        finally:
            # Restore original config if it existed
            if orig_config:
                config_file.write_text(orig_config)
            if input_file.exists():
                input_file.unlink()