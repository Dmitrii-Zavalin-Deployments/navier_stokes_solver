# tests/property_integrity/test_heavy_elasticity_lifecycle.py

import json
import logging
from pathlib import Path

import pytest

from src.main_solver import BASE_DIR, run_solver


class TestHeavyElasticityLifecycle:
    def test_numerical_stabilization_and_terminal_failure(self, caplog):
        """
        Validates the new Elasticity Range-Sweep logic:
        1. Triggers ArithmeticError via extreme initial velocity.
        2. Verifies the solver attempts to stabilize by reducing dt.
        3. Verifies terminal failure when the dt_floor is reached.
        """
        input_filename = "integration_input.json"
        config_filename = "config.json"
        
        input_path = Path(BASE_DIR) / input_filename
        config_path = Path(BASE_DIR) / config_filename
        
        # Define 10 steps in config to match ElasticManager._runs
        config_data = {
            "dt_min_limit": 0.0001,
            "ppe_tolerance": 1e-4, 
            "ppe_max_iter": 50, 
            "ppe_omega": 1.7
        }

        # Force immediate explosion with 1e10 velocity
        nx, ny, nz = 4, 4, 4
        input_data = {
            "domain_configuration": {"type": "INTERNAL"},
            "grid": {
                "x_min": 0.0, "x_max": 1.0, 
                "y_min": 0.0, "y_max": 1.0, 
                "z_min": 0.0, "z_max": 1.0, 
                "nx": nx, "ny": ny, "nz": nz
            },
            "fluid_properties": {"density": 1.0, "viscosity": 0.001},
            "initial_conditions": {"velocity": [1e10, 1e10, 1e10], "pressure": 1.0},
            "simulation_parameters": {"time_step": 0.5, "total_time": 10.0, "output_interval": 1},
            "boundary_conditions": [
                {"location": "x_min", "type": "inflow", "values": {"u": 50.0, "v": 0.0, "w": 0.0}}, 
                {"location": "x_max", "type": "outflow", "values": {"p": 0.0}}
            ],
            "mask": [0] * (nx * ny * nz),
            "external_forces": {"force_vector": [0.0, -9.81, 0.0]}
        }

        try:
            # Setup environment
            input_path.write_text(json.dumps(input_data))
            config_path.write_text(json.dumps(config_data))

            with caplog.at_level(logging.WARNING):
                # We expect the solver to eventually give up after 10 attempts
                with pytest.raises(RuntimeError) as excinfo:
                    run_solver(input_filename)

                # --- FORENSIC AUDIT ---
                error_msg = str(excinfo.value)
                print(f"\n[Forensics] Error caught: {error_msg}")

                # Verify the error comes from our ElasticManager stabilization logic
                assert "stable" in error_msg.lower()
                assert "dt_floor" in error_msg

                # Verify that stabilization attempts were logged
                stabilization_logs = [rec for rec in caplog.records if "Instability detected" in rec.message]
                print(f"[Forensics] Captured {len(stabilization_logs)} stabilization retry events.")
                
                # It should have tried multiple times before hitting the RuntimeError
                assert len(stabilization_logs) > 0, "ELASTICITY FAIL: Stabilization retries never triggered."
                assert len(stabilization_logs) <= 10, "ELASTICITY FAIL: Exceeded maximum allowed retries."

        finally:
            # # 7. PURGE: Universal Cleanup (Rule 2: Zero Debt)

            # if input_path.exists():
            #     input_path.unlink()

            # if config_path.exists():
            #     config_path.unlink()

            # if output_dir.exists():
            #     for item in output_dir.iterdir():
            #         if item.name == ".gitkeep":
            #             continue
            #         if item.is_dir():
            #             shutil.rmtree(item)
            #         else:
            #             item.unlink()

            #     print(f"\n[Sanitization] Purged directory: {output_dir.name}")

            # if temporary_output_dir.exists():
            #     shutil.rmtree(temporary_output_dir)
            #     print(f"\n[Sanitization] Removed directory and all contents: {temporary_output_dir.name}")

            # # 8. FINAL AUDIT

            # if temporary_output_dir.exists():
            #     remaining = [p for p in temporary_output_dir.iterdir() if p.name != ".gitkeep"]
            # else:
            #     remaining = []

            # assert not remaining, (
            #     f"CLEANUP FAILURE: unexpected items in {temporary_output_dir}: "
            #     f"{[p.name for p in remaining]}"
            # )

            # assert not temporary_output_dir.exists(), (
            #     f"CLEANUP FAILURE: directory was not deleted: {temporary_output_dir}"
            # )
            pass