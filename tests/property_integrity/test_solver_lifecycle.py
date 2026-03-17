# tests/property_integrity/test_solver_lifecycle.py

import json
from pathlib import Path

from src.main_solver import BASE_DIR, run_solver


class TestSolverLifecycle:
    """
    SYSTEM AUDITOR: Verifies the full pipeline flow using a 'Smoke Test'.
    Injects a tiny valid input into the real repo root, runs the full solver,
    verifies the ZIP in the real output folder, and cleans up immediately.
    """

    def test_live_plumbing_smoke_test(self):
        # 1. Configuration (Salted name to ensure no production collision)
        test_filename = "input_SMOKE_TEST_INTEGRITY.json"
        # We use the real BASE_DIR from the solver's own logic
        input_path = Path(BASE_DIR) / test_filename
        
        # Tiny 4x4x4 grid for a sub-second execution (Rule 9 compliant)
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
                "time_step": 0.001,
                "total_time": 0.002,  # Minimal time steps
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

        zip_path = None
        try:
            # 2. Inject: Write directly to repo root where solver expects it
            input_path.write_text(json.dumps(input_data))

            # 3. Execute: Run the real production pipeline
            # This uses the real config.json and real schema/ folder
            zip_path_str = run_solver(test_filename)
            zip_path = Path(zip_path_str)

            # 4. Verify: Ensure the plumbing actually connected to the output folder
            assert zip_path.exists(), "ERROR: Solver finished but ZIP was not created."
            assert zip_path.suffix == ".zip", "ERROR: Output is not a ZIP file."
            assert "testing-input-output" in str(zip_path), "ERROR: ZIP saved in wrong directory."

        finally:
            # 5. Leave No Trace: Cleanup both input and output artifacts
            if input_path.exists():
                input_path.unlink()
            
            if zip_path and zip_path.exists():
                zip_path.unlink()
                print(f"\n[Audit] Cleanup successful: Removed {zip_path.name}")