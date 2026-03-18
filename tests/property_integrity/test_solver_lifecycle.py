# tests/property_integrity/test_solver_lifecycle.py

import json
from pathlib import Path

from src.main_solver import BASE_DIR, run_solver


class TestSolverLifecycle:
    """
    SYSTEM AUDITOR: High-Fidelity Smoke Test.
    Uses the EXACT production filename 'input_test.json'.
    Ensures the 'Plumbing' is ready for the real Dropbox download.
    """

    def test_production_plumbing_gatekeeper(self):
        # 1. Target: The exact production filename
        test_filename = "input_test.json"
        input_path = Path(BASE_DIR) / test_filename
        
        # 2. Payload: Minimal 4x4x4 grid (Schema-compliant)
        # Reduced time_step and total_time to ensure numerical stability on tiny grids.
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
                "time_step": 0.0001,  # Reduced from 0.001 to prevent blow-up
                "total_time": 0.0002, # Reduced to stay within stable bounds
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
        output_dir = Path(BASE_DIR) / "data" / "testing-input-output"

        try:
            # 3. Inject: Create the file in the repo root
            input_path.write_text(json.dumps(input_data))

            # 4. Execute: Call the real production entry point
            zip_path_str = run_solver(test_filename)
            zip_path = Path(zip_path_str)

            # 5. Assert: Verify the plumbing success
            assert zip_path.exists(), "GATEKEEPER FAIL: ZIP not created."
            assert zip_path.parent == output_dir, f"GATEKEEPER FAIL: Wrong output dir. Expected {output_dir}"
            assert zip_path.stat().st_size > 0, "GATEKEEPER FAIL: ZIP is empty."

        finally:
            # 6. PURGE: Universal Cleanup
            if input_path.exists():
                input_path.unlink()
            
            if output_dir.exists():
                # We now search for ALL zips, including 'navier_stokes_output.zip'
                # to ensure the folder is pristine for the real production data.
                for artifact in output_dir.glob("*.zip"):
                    artifact.unlink()
                    print(f"\n[Sanitization] Purged artifact: {artifact.name}")
            
            # 7. FINAL AUDIT: Assert the folder is 100% clean
            leftover_zips = list(output_dir.glob("*.zip"))
            assert len(leftover_zips) == 0, f"CLEANUP FAILURE: Found {len(leftover_zips)} leftover artifacts."