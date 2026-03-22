# tests/property_integrity/test_heavy_elasticity_lifecycle.py

import json
import logging
from pathlib import Path

import pytest

from src.main_solver import BASE_DIR, run_solver


class TestHeavyElasticityLifecycle:

    @pytest.fixture
    def base_config(self):
        return {
            "dt_min_limit": 0.0001,
            "ppe_tolerance": 1e-4, 
            "ppe_atol": 1e-10,
            "ppe_max_iter": 20,
            "ppe_omega": 1.7,
            "ppe_max_retries": 10
        }

    @pytest.fixture
    def base_input(self):
        return {
            "domain_configuration": {"type": "INTERNAL"},
            "grid": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0, "nx": 4, "ny": 4, "nz": 4},
            "fluid_properties": {"density": 1.0, "viscosity": 0.001},
            "initial_conditions": {"velocity": [0.0, 0.0, 0.0], "pressure": 1.0},
            "simulation_parameters": {"time_step": 0.1, "total_time": 0.2, "output_interval": 1},
            "boundary_conditions": [
                {"location": "x_min", "type": "inflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0}}, 
                {"location": "x_max", "type": "outflow", "values": {"p": 0.0}}
            ],
            "mask": [0] * 64,
            "external_forces": {"force_vector": [0.0, -9.81, 0.0]}
        }

    def test_scenario_1_pure_success(self, caplog, base_config, base_input):
        """
        Scenario 1: Run completes without any instabilities.
        Aligned with Phase C: Rule 2 (Zero-Debt) and Rule 7 (Scientific Truth).
        """
        import zipfile

        # 1. SETUP (Rule 5: Explicit Initialization)
        input_filename = "test_success_input.json"
        config_path = Path(BASE_DIR) / "config.json"
        input_path = Path(BASE_DIR) / input_filename
        
        config_path.write_text(json.dumps(base_config))
        input_path.write_text(json.dumps(base_input))

        with caplog.at_level(logging.WARNING, logger=""):
            # 2. EXECUTION
            zip_path = run_solver(input_filename)

            # 3. LOG AUDIT (Rule 6: Efficiency)
            # Confirming that 100% of the successful path was exercised without retries
            instability_logs = [r for r in caplog.records if  "instability" in r.message.lower()]
            assert len(instability_logs) == 0, (
                f"Scenario 1 Error: Expected 0 retries, but found {len(instability_logs)}. "
                f"Logs: {[r.message for r in instability_logs]}"
            )
            
            # 4. PATH AUDIT
            assert Path(zip_path).exists(), "The final archive was not created."
            assert zip_path.endswith(".zip")
            assert "navier_stokes_output.zip" in zip_path

            # 5. DEEP ARCHIVE INSPECTION
            with zipfile.ZipFile(zip_path, 'r') as archive:
                namelist = archive.namelist()
                csv_files = sorted([f for f in namelist if f.endswith('.h5')])
                assert len(csv_files) >= 2 

                with archive.open(csv_files[-1]) as f:
                    header = f.read(8)
                    assert header.startswith(b'\x89HDF'), 'Foundation Error: Snapshot is not a valid HDF5 binary'
                    # Rule 9: Structural Foundation Audit
                    from io import BytesIO

                    import h5py
                    # Re-verify internal H5 integrity by attempting a structural peek
                    f.seek(0)
                    with h5py.File(BytesIO(f.read()), 'r') as h5_audit:
                        assert 'vx' in h5_audit.keys(), 'Foundation Error: Missing VX dataset'
                        # Rule 7: Verify Physics Propagation in Foundation
                        vx_data = h5_audit['vx'][:]
                        # Check for non-zero velocity and finite values
                        import numpy as np
                        assert np.all(np.isfinite(vx_data)), 'Foundation Error: Non-finite values in VX'
                        assert np.max(np.abs(vx_data)) > 0, 'Physics Error: Zero velocity propagation'
                        # Rule 7: Verify Physics Propagation in Foundation
                        vx_data = h5_audit['vx'][:]
        
    def test_scenario_2_retry_and_recover(self, caplog, base_config, base_input):
    """
    Scenario 2: Recovery Audit.
    Verifies that ArithmeticError triggers dt reduction AND results in valid HDF5 data.
    """
    import zipfile
    import h5py
    import numpy as np
    from io import BytesIO

    # 1. SETUP: Force an unstable condition (Rule 7: Scientific Truth)
    # High velocity + High dt = Guaranteed Courant Number violation/Instability
    base_input["simulation_parameters"]["time_step"] = 0.8 
    base_input["boundary_conditions"][0]["values"]["u"] = 1e15 
    
    config_path = Path(BASE_DIR) / "config.json"
    input_path = Path(BASE_DIR) / "test_recovery_input.json"
    
    config_path.write_text(json.dumps(base_config))
    input_path.write_text(json.dumps(base_input))

    with caplog.at_level(logging.WARNING, logger=""):
        # 2. EXECUTION
        zip_path = run_solver("test_recovery_input.json")
        
        # --- PHASE A: LOGICAL RECOVERY AUDIT ---
        
        # Verify that we didn't accidentally slip into a 'Pure Success' path
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        stabilization_logs = [r for r in warnings if "instability" in r.message.lower()]
        
        assert len(stabilization_logs) > 0, (
            "TEST SLIPPAGE: Scenario 2 expected an instability/retry, but none occurred. "
            "The test input is not aggressive enough to trigger the recovery path."
        )
        assert len(stabilization_logs) < base_config["ppe_max_retries"], "Solver hit max retries."

        # --- PHASE B: SCIENTIFIC INTEGRITY AUDIT (Rule 9) ---
        
        assert Path(zip_path).exists(), "Recovery succeeded but archive was not generated."
        
        with zipfile.ZipFile(zip_path, 'r') as archive:
            h5_files = sorted([f for f in archive.namelist() if f.endswith('.h5')])
            # We need at least the initial state and one successful step post-recovery
            assert len(h5_files) >= 2, f"Expected at least 2 snapshots, got {len(h5_files)}"

            # Inspect the LAST snapshot to ensure the physics 'settled' after recovery
            with archive.open(h5_files[-1]) as f:
                content = f.read()
                assert content.startswith(b'\x89HDF'), "Final snapshot is corrupted HDF5"
                
                with h5py.File(BytesIO(content), 'r') as h5_audit:
                    # Check for existence of core datasets
                    for field in ['vx', 'vy', 'vz', 'p']:
                        assert field in h5_audit.keys(), f"Physics Error: Missing {field} in recovery output"
                    
                    vx_data = h5_audit['vx'][:]
                    
                    # THE SMOKING GUN: If recovery worked, values MUST be finite.
                    # If it 'failed upward', these would be INF or NAN.
                    assert np.all(np.isfinite(vx_data)), (
                        "ROOT CAUSE: Recovery path executed, but the resulting physics "
                        "contains Non-Finite values (NaN/Inf). Stabilization failed."
                    )
                    
                    # Ensure the simulation actually 'moved'
                    assert np.max(np.abs(vx_data)) > 0, "Physics Error: Recovery resulted in a dead/zero field."

    print(f"DEBUG: Successfully recovered from {len(stabilization_logs)} instabilities.")

    def test_scenario_3_terminal_failure(self, caplog, base_config, base_input):
        """Scenario 3: 10 failed attempts lead to terminal RuntimeError."""
        base_input["boundary_conditions"][0]["values"]["u"] = 1e15 # Global explosion
        
        Path(BASE_DIR / "config.json").write_text(json.dumps(base_config))
        Path(BASE_DIR / "test_input.json").write_text(json.dumps(base_input))

        with caplog.at_level(logging.WARNING, logger=""):
            with pytest.raises(RuntimeError) as excinfo:
                run_solver("test_input.json")
            
            error_msg = str(excinfo.value)
            assert "unstable" in error_msg.lower()
            logs = [r for r in caplog.records if  "instability" in r.message.lower()]
            assert len(logs) == base_config["ppe_max_retries"]