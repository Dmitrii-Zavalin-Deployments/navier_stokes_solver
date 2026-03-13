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
        project_root = Path(__file__).resolve().parent.parent.parent
        config_file = project_root / "config.json"
        input_file = project_root / "input_validated.json"
        
        orig_config = config_file.read_text() if config_file.exists() else None
        
        try:
            # 1. Define configuration explicitly
            config_dict = {
                "ppe_tolerance": 1e-6, 
                "ppe_max_iter": 10, 
                "ppe_omega": 1.0
            }
            config_file.write_text(json.dumps(config_dict))
            
            # 2. Write mock input data
            input_dict = {
                "domain_configuration": {"type": "INTERNAL", "reference_velocity": [0.0, 0.0, 0.0]},
                "grid": {"nx": 4, "ny": 4, "nz": 4, "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0},
                "fluid_properties": {"density": 1000.0, "viscosity": 0.001},
                "initial_conditions": {"velocity": [0.0, 0.0, 0.0], "pressure": 0.0},
                "simulation_parameters": {"time_step": 0.001, "total_time": 0.002, "output_interval": 1},
                "external_forces": {"force_vector": [0.0, 0.0, -9.81]},
                "mask": [0] * 64,
                "boundary_conditions": [
                    {"location": "x_min", "type": "inflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0, "p": 1.0}},
                    {"location": "x_max", "type": "outflow", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}},
                    {"location": "y_min", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}}
                ]
            }
            input_file.write_text(json.dumps(input_dict))
            
            # 3. Execution
            # If run_solver() internally uses config_file, this now succeeds 
            # because config_file exists and is valid.
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