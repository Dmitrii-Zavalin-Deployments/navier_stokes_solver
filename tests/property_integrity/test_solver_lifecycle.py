# tests/integration/test_solver_lifecycle.py

import pytest
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
            # 1. Write configuration matching the dummy's physics profile
            config_file.write_text(json.dumps({
                "ppe_tolerance": 1e-6, 
                "ppe_max_iter": 10, 
                "ppe_omega": 1.0
            }))
            
            # 2. Write mock data mirroring make_step1_output_dummy structure
            input_file.write_text(json.dumps({
                "domain_configuration": {"type": "INTERNAL", "reference_velocity": [0.0, 0.0, 0.0]},
                "grid": {"nx": 4, "ny": 4, "nz": 4, "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0},
                "fluid_properties": {"density": 1000.0, "viscosity": 0.001},
                "initial_conditions": {"velocity": [0.0, 0.0, 0.0], "pressure": 0.0},
                "simulation_parameters": {"time_step": 0.001, "total_time": 0.002, "output_interval": 1},
                "external_forces": {"force_vector": [0.0, 0.0, -9.81]},
                "mask": [1] * 64, # Matches the dummy's np.ones mask
                "boundary_conditions": {
                    "conditions": [
                        {"side": "x_min", "bc_type": "inflow", "value": {"u": 1.0, "v": 0.0, "w": 0.0, "p": 1.0}},
                        {"side": "x_max", "bc_type": "outflow", "value": {"p": 0.0}},
                        {"side": "y_min", "bc_type": "no-slip", "value": {"u": 0.0, "v": 0.0, "w": 0.0}}
                    ]
                }
            }))
            
            # 3. Execution
            final_state = run_solver("input_validated.json")
            
            # 4. Assertions
            assert final_state.ready_for_time_loop is False
            assert len(final_state.manifest.saved_snapshots) > 0
            
            # 5. Archive Verification
            archive_path = project_root / "navier_stokes_test_case_output.zip"
            assert archive_path.exists(), "Archive service failed to produce output."
            
            # Cleanup
            archive_path.unlink()
            
        finally:
            if orig_config:
                config_file.write_text(orig_config)
            if input_file.exists():
                input_file.unlink()