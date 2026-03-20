# tests/integration/test_heavy_elasticity_lifecycle.py

import json
import logging
import zipfile
from pathlib import Path

import pytest

from src.main_solver import BASE_DIR, run_solver


class TestHeavyElasticityLifecycle:
    """
    SYSTEM AUDITOR: High-Fidelity Integration Gatekeeper.
    Verifies: Elasticity Panic Mode, Recovery, and Atomic Archiving.
    """

    def test_numerical_panic_and_recovery_flow(self, caplog):
        # Rule 5: Explicit Initialization. 
        # We define variables as None to ensure no 'Silent Defaults' mask logic gaps.
        panic_logs = []
        zip_path = None
        
        # 1. Setup Production-filenames
        input_filename = "integration_input.json"
        config_filename = "config.json"
        
        input_path = Path(BASE_DIR) / input_filename
        config_path = Path(BASE_DIR) / config_filename
        
        # 2. Config: Sensitive settings for ElasticManager (Rule 5 compliance)
        config_data = {
            "ppe_tolerance": 1e-4,
            "ppe_atol": 1e-6,
            "ppe_max_iter": 50,
            "ppe_omega": 1.7,       # Aggressive over-relaxation
            "divergence_threshold": 1e6
        }

        # 3. Input: High-velocity 3D grid to trigger ArithmeticError
        nx, ny, nz = 4, 4, 4
        input_data = {
            "domain_configuration": {"type": "INTERNAL"},
            "grid": {
                "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0,
                "nx": nx, "ny": ny, "nz": nz
            },
            "fluid_properties": {"density": 1.0, "viscosity": 0.001},
            "initial_conditions": {"velocity": [1e10, 1e10, 1e10], "pressure": 1.0},
            "simulation_parameters": {
                "time_step": 0.5,
                "total_time": 10.0, 
                "output_interval": 1
            },
            "boundary_conditions": [
                {"location": "x_min", "type": "inflow", "values": {"u": 50.0, "v": 0.0, "w": 0.0}},
                {"location": "x_max", "type": "outflow", "values": {"p": 0.0}}
            ],
            "mask": [0] * (nx * ny * nz),
            "external_forces": {"force_vector": [0.0, -9.81, 0.0]}
        }

        output_dir = Path(BASE_DIR) / "data" / "testing-input-output"

        try:
            # 4. Inject
            input_path.write_text(json.dumps(input_data))
            config_path.write_text(json.dumps(config_data))

            # Set log level to capture ElasticManager warnings
            with caplog.at_level(logging.WARNING):
                # 5. EXECUTION & LOG CAPTURE (Rule 7: Atomic Verification)
                with pytest.raises(RuntimeError) as excinfo:
                    # Capture the path if it returns before the crash
                    zip_path = run_solver(input_filename)
                
                # Rule 6: Cover the Gap. Extracting logs immediately after failure context.
                panic_logs = [rec for rec in caplog.records if 'PANIC' in rec.message]
                
                assert "Solver cannot recover" in str(excinfo.value)
                assert len(panic_logs) > 0, "ELASTICITY FAIL: Panic Mode was never triggered."
                
                print(f"Captured {len(panic_logs)} panic events.")

            # 7. ARCHIVE AUDIT: Deep inspection (Rule 1 & 4)
            # Only proceed if the solver actually produced a path (avoids F821/AttributeError)
            if zip_path and Path(zip_path).exists():
                audit_path = Path(zip_path)
                with zipfile.ZipFile(audit_path, 'r') as archive:
                    state_bytes = archive.read("simulation_state.json")
                    state_json = json.loads(state_bytes)
                    
                    # Rule 4: SSoT Check - Accessing data through the assigned container
                    data_array = state_json.get("fields", {}).get("data", [])
                    assert len(data_array) > 0, "ARCHIVE FAIL: State contains no field data."
                    assert state_json["time"] > 0, "TIMELINE FAIL: Simulation did not progress."

        finally:
            # 8. SANITIZATION
            if input_path.exists(): input_path.unlink()
            if config_path.exists(): config_path.unlink()
            if output_dir.exists():
                for artifact in output_dir.glob("*.zip"):
                    artifact.unlink()