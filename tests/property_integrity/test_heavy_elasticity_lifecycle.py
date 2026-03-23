# tests/property_integrity/test_heavy_elasticity_lifecycle.py

import json
import logging
import zipfile
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.main_solver import BASE_DIR, run_solver


class TestHeavyElasticityLifecycle:

    @pytest.fixture
    def base_config(self):
        """Standard stable config for the solver."""
        return {
            "dt": 0.01,
            "dt_min_limit": 0.001,
            "ppe_tolerance": 1e-4, 
            "ppe_atol": 1e-10,
            "ppe_max_iter": 20,
            "ppe_omega": 1.7,
            "ppe_max_retries": 5
        }

    @pytest.fixture
    def base_input(self):
        """Standard physical setup with reasonable constraints."""
        return {
            "domain_configuration": {"type": "INTERNAL"},
            "grid": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0, "nx": 4, "ny": 4, "nz": 4},
            "fluid_properties": {"density": 1.0, "viscosity": 0.01},
            "initial_conditions": {"velocity": [0.1, 0.0, 0.0], "pressure": 1.0},
            "simulation_parameters": {"time_step": 0.01, "total_time": 0.02, "output_interval": 1},
            "boundary_conditions": [
                {"location": "x_min", "type": "inflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0}}, 
                {"location": "x_max", "type": "outflow", "values": {"p": 0.0}}
            ],
            "mask": [0] * 64,
            "external_forces": {"force_vector": [0.0, 0.0, 0.0]},
            # SMALL SCALE CONSTRAINTS
            "physical_constraints": {"min_velocity": -20.0, "max_velocity": 20.0, "min_pressure": -50.0, "max_pressure": 50.0}
        }

    def test_scenario_1_pure_success(self, caplog, base_config, base_input):
        """Scenario 1: Normal run. No stability triggers should fire."""
        input_filename = "test_success_input.json"
        (Path(BASE_DIR) / "config.json").write_text(json.dumps(base_config))
        (Path(BASE_DIR) / input_filename).write_text(json.dumps(base_input))

        with caplog.at_level(logging.WARNING, logger="Solver.Main"):
            zip_path = run_solver(input_filename)
            
            triggers = [r for r in caplog.records if "STABILITY TRIGGER" in r.message]
            assert len(triggers) == 0

            with zipfile.ZipFile(zip_path, 'r') as archive:
                h5_files = [f for f in archive.namelist() if f.endswith('.h5')]
                assert len(h5_files) > 0
                
                with archive.open(h5_files[-1]) as f:
                    with h5py.File(BytesIO(f.read()), 'r') as h5_audit:
                        assert 'vx' in h5_audit.keys()
                        assert np.all(np.isfinite(h5_audit['vx'][:]))

    def test_scenario_2_retry_and_recover(self, caplog, base_config, base_input):
        """
        Scenario 2: THE ELASTICITY TEST.
        We use a velocity that is 'fast' (15.0) but allowed by Audit (20.0).
        The large dt (0.1) will force a math trigger, and the solver must downstep to 0.001 to pass.
        """
        # 1. Setup physics: 15.0 < 20.0 (Audit passes, but CFL math will likely fail)
        base_input["boundary_conditions"][0]["values"]["u"] = 15.0
        
        # 2. Config: Force a downstep by starting with a huge DT
        base_config["dt"] = 0.1 
        base_config["dt_min_limit"] = 0.0001
        base_config["ppe_max_retries"] = 10 
        
        input_filename = "test_recovery_input.json"
        config_path = Path(BASE_DIR) / "config.json"
        input_path = Path(BASE_DIR) / input_filename
        
        config_path.write_text(json.dumps(base_config))
        input_path.write_text(json.dumps(base_input))

        with caplog.at_level(logging.WARNING, logger="Solver.Main"):
            zip_path = run_solver(input_filename)
        
        assert zip_path is not None
        # Verify the logs show at least one reduction happened
        assert any("STABILITY TRIGGER" in r.message for r in caplog.records)
        
        # Cleanup
        input_path.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)

    def test_scenario_3_terminal_failure(self, caplog, base_config, base_input):
        """
        Scenario 3: Force a crash.
        Velocity (50.0) is way ABOVE the Audit Limit (20.0).
        Even if it shrinks dt, the Audit will block it every time.
        """
        base_input["boundary_conditions"][0]["values"]["u"] = 50.0
        base_config["dt_min_limit"] = 0.01 
        base_config["ppe_max_retries"] = 2 
        
        input_filename = "test_terminal_fail.json"
        (Path(BASE_DIR) / "config.json").write_text(json.dumps(base_config))
        (Path(BASE_DIR) / input_filename).write_text(json.dumps(base_input))

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError) as excinfo:
                run_solver(input_filename)
        
        # 3. Forensic Assertions
        error_msg = str(excinfo.value)
        assert "CRITICAL INSTABILITY" in error_msg
        assert "Exhausted" in error_msg
        
        # Check logs for the audit trail
        assert "AUDIT [Explosion]" in caplog.text
        assert "STABILITY TRIGGER" in caplog.text
        
        # Cleanup
        input_path.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)