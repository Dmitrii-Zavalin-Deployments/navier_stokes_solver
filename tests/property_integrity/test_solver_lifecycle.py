import json
from pathlib import Path

from src.main_solver import BASE_DIR, run_solver


class TestSolverLifecycle:
    """
    SYSTEM AUDITOR: High-Fidelity Smoke Test.
    Updated for Rule 4/5 Compliance: Supports Elasticity Engine and Explicit Config.
    Ensures the 'Plumbing' is ready for production-grade execution.
    """

    def test_production_plumbing_gatekeeper(self):
        # 1. Target: The exact production filenames required by main_solver
        test_filename = "input_test.json"
        config_filename = "config.json"
        
        input_path = Path(BASE_DIR) / test_filename
        config_path = Path(BASE_DIR) / config_filename

        test_data_dir = Path(BASE_DIR) / "data" / "testing-input-output"
        production_output_dir = Path(BASE_DIR) / "output"
        
        # 2. Payload A: Numerical Configuration (Required for Elasticity)
        # These parameters prevent the AttributeError: 'SolverConfig' object has no attribute 'dt'
        config_data = {
            "ppe_tolerance": 1e-5,
            "ppe_atol": 1e-7,
            "ppe_max_iter": 50,
            "ppe_omega": 1.2
        }

        # 3. Payload B: Physical Input (Schema-compliant)
        nx, ny, nz = 4, 4, 4
        input_data = {
            "domain_configuration": {"type": "INTERNAL"},
            "grid": {
                "x_min": 0.0, "x_max": 1.0,
                "y_min": 0.0, "y_max": 1.0,
                "z_min": 0.0, "z_max": 1.0,
                "nx": nx, "ny": ny, "nz": nz
            },
            "fluid_properties": {"density": 1.0, "viscosity": 0.01},
            "initial_conditions": {"velocity": [0.0, 0.0, 0.0], "pressure": 0.0},
            "simulation_parameters": {
                "time_step": 0.0001,
                "total_time": 0.0002,
                "output_interval": 1
            },
            "boundary_conditions": [
                {
                    "location": "x_min", "type": "inflow",
                    "values": {"u": 1.0, "v": 0.0, "w": 0.0}
                },
                {
                    "location": "x_max", "type": "outflow",
                    "values": {"p": 0.0}
                }
            ],
            "mask": [0] * (nx * ny * nz),
            "external_forces": {"force_vector": [0.0, -9.81, 0.0]}
        }

        output_dir = Path(BASE_DIR) / "data" / "testing-input-output"

        try:
            # 4. Inject: Create the files in the repo root
            input_path.write_text(json.dumps(input_data))
            config_path.write_text(json.dumps(config_data))

            # 5. Execute: Call the upgraded production entry point
            zip_path_str = run_solver(test_filename)
            zip_path = Path(zip_path_str)

            # 6. Assert: Verify the plumbing success
            assert zip_path.exists(), "GATEKEEPER FAIL: ZIP not created."
            assert zip_path.parent == output_dir, f"GATEKEEPER FAIL: Wrong output dir. Expected {output_dir}"
            assert zip_path.stat().st_size > 0, "GATEKEEPER FAIL: ZIP is empty."

        finally:
            # 7. PURGE: Universal Cleanup (Rule 2: Zero Debt)
            if input_path.exists():
                input_path.unlink()
            
            if config_path.exists():
                config_path.unlink()
            
            if output_dir.exists():
                for artifact in output_dir.glob("*.zip"):
                    artifact.unlink()
                    print(f"\n[Sanitization] Purged artifact: {artifact.name}")
            
            if production_output_dir.exists():
                shutil.rmtree(production_output_dir)
                print(f"\n[Sanitization] Purged rogue directory: {production_output_dir.name}")
            
            # 8. FINAL AUDIT: Assert the folder is 100% clean
            leftover_zips = list(output_dir.glob("*.zip"))
            assert len(leftover_zips) == 0, f"CLEANUP FAILURE: Found {len(leftover_zips)} leftover artifacts."

            assert not production_output_dir.exists(), f"CLEANUP FAILURE: {production_output_dir} still exists."
            assert not test_data_dir.exists(), f"CLEANUP FAILURE: {test_data_dir} still exists."