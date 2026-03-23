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
        return {
            "dt_min_limit": 0.0001,
            "ppe_tolerance": 1e-4, 
            "ppe_atol": 1e-10,
            "ppe_max_iter": 20,
            "ppe_omega": 1.7,
            "ppe_max_retries": 5 # Lowered for faster test execution
        }

    @pytest.fixture
    def base_input(self, tmp_path):
        # Use a small 4x4x4 grid for speed
        return {
            "domain_configuration": {"type": "INTERNAL"},
            "grid": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0, "nx": 4, "ny": 4, "nz": 4},
            "fluid_properties": {"density": 1.0, "viscosity": 0.001},
            "initial_conditions": {"velocity": [0.1, 0.0, 0.0], "pressure": 1.0},
            "simulation_parameters": {"time_step": 0.01, "total_time": 0.03, "output_interval": 1},
            "boundary_conditions": [
                {"location": "x_min", "type": "inflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0}}, 
                {"location": "x_max", "type": "outflow", "values": {"p": 0.0}}
            ],
            "mask": [0] * 64,
            "external_forces": {"force_vector": [0.0, 0.0, 0.0]}
            "physical_constraints": {"min_velocity": -1e6, "max_velocity": 1e6, "min_pressure": -1e6, "max_pressure": 1e6},
        }

    def test_scenario_1_pure_success(self, caplog, base_config, base_input):
        """Scenario 1: Normal run. No stability triggers should fire."""
        input_filename = "test_success_input.json"
        (Path(BASE_DIR) / "config.json").write_text(json.dumps(base_config))
        (Path(BASE_DIR) / input_filename).write_text(json.dumps(base_input))

        with caplog.at_level(logging.WARNING, logger="Solver.Main"):
            zip_path = run_solver(input_filename)
            
            # Audit: No "STABILITY TRIGGER" should be in logs
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
        """Scenario 2: Force recovery via extreme velocity."""
        # High velocity guarantees a stability trigger if your audit is active
        base_input["boundary_conditions"][0]["values"]["u"] = 1e10 
        
        input_filename = "test_recovery_input.json"
        (Path(BASE_DIR) / "config.json").write_text(json.dumps(base_config))
        (Path(BASE_DIR) / input_filename).write_text(json.dumps(base_input))

        with caplog.at_level(logging.WARNING, logger="Solver.Main"):
            zip_path = run_solver(input_filename)
            
            # Check for recovery logs
            stabilization_logs = [r for r in caplog.records if "STABILITY TRIGGER" in r.message]
            assert len(stabilization_logs) > 0, "Recovery logic failed to trigger on extreme velocity."

            # Verify scientific integrity of the final file
            with zipfile.ZipFile(zip_path, 'r') as archive:
                h5_files = sorted([f for f in archive.namelist() if f.endswith('.h5')])
                with archive.open(h5_files[-1]) as f:
                    with h5py.File(BytesIO(f.read()), 'r') as h5_audit:
                        # Even after recovery, data must be finite
                        assert np.all(np.isfinite(h5_audit['vx'][:]))

    def test_scenario_3_terminal_failure(self, caplog, base_config, base_input):
        """Scenario 3: Force a crash by making stability impossible."""
        base_input["boundary_conditions"][0]["values"]["u"] = 1e20
        base_config["dt_min_limit"] = 0.1 # Floor is too high to recover
        
        input_filename = "test_fail_input.json"
        (Path(BASE_DIR) / "config.json").write_text(json.dumps(base_config))
        (Path(BASE_DIR) / input_filename).write_text(json.dumps(base_input))

        with pytest.raises(RuntimeError) as excinfo:
            run_solver(input_filename)
        
        assert "CRITICAL INSTABILITY" in str(excinfo.value)