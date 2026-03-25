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
            "ppe_max_iter": 5,
            "ppe_omega": 1.7,
            "ppe_max_retries": 5,
            "divergence_threshold": 1e12,
        }

    @pytest.fixture
    def base_input(self):
        """Standard physical setup with reasonable constraints."""
        return {
            "domain_configuration": {"type": "INTERNAL"},
            "grid": {
                "x_min": 0.0,
                "x_max": 1.0,
                "y_min": 0.0,
                "y_max": 1.0,
                "z_min": 0.0,
                "z_max": 1.0,
                "nx": 4,
                "ny": 4,
                "nz": 4,
            },
            "fluid_properties": {"density": 1.0, "viscosity": 0.01},
            "initial_conditions": {"velocity": [0.1, 0.0, 0.0], "pressure": 1.0},
            "simulation_parameters": {
                "time_step": 0.01,
                "total_time": 0.02,
                "output_interval": 1,
            },
            # --- [Rule 5 Alignment]: Strictly following your JSON Schema ---
            "physical_constraints": {
                "min_velocity": -100.0,
                "max_velocity": 40.0,
                "min_pressure": -10.0,
                "max_pressure": 100.0,
            },
            "boundary_conditions": [
                # Note: Even for slip/no-slip, the schema requires "values"
                # to avoid a ValidationError, even if the kernel ignores them.
                {"location": "y_min", "type": "free-slip", "values": {"u": 0.0}},
                {"location": "y_max", "type": "free-slip", "values": {"u": 0.0}},
                {"location": "z_min", "type": "free-slip", "values": {"u": 0.0}},
                {"location": "z_max", "type": "free-slip", "values": {"u": 0.0}},
                {
                    "location": "x_min",
                    "type": "inflow",
                    "values": {"u": 1.0, "v": 0.0, "w": 0.0},
                },
                {"location": "x_max", "type": "outflow", "values": {"p": 0.0}},
            ],
            "mask": [0] * 64,
            "external_forces": {"force_vector": [0.0, 0.0, 0.0]},
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

            with zipfile.ZipFile(zip_path, "r") as archive:
                h5_files = [f for f in archive.namelist() if f.endswith(".h5")]
                assert len(h5_files) > 0

                with archive.open(h5_files[-1]) as f:
                    with h5py.File(BytesIO(f.read()), "r") as h5_audit:
                        assert "vx" in h5_audit.keys()
                        assert np.all(np.isfinite(h5_audit["vx"][:]))

    def test_scenario_2_elasticity_terminal_failure(self, caplog, base_config, base_input):
        """
        Scenario 2: Elasticity exercised fully, but no dt can stabilize the run.
        Expect a CRITICAL INSTABILITY after exhausting retries.
        """
        # --- CFL setup: deliberately aggressive to force PPE/pressure failure ---
        u = np.float64(20.0)
        dt_initial = np.float64(0.08)

        x_max = np.float64(base_input["grid"]["x_max"])
        x_min = np.float64(base_input["grid"]["x_min"])
        nx = np.float64(base_input["grid"]["nx"])
        dx = (x_max - x_min) / nx
        courant_number = (u * dt_initial) / dx

        logging.info(
            f"DEBUG: Scenario 2 Initial Courant Number = {courant_number:.4f} (Target >> 1.0)"
        )

        # 1. Physics: strong inflow, wide pressure corridor to let PPE roam
        base_input["boundary_conditions"][0]["values"]["u"] = float(u)
        base_input["physical_constraints"]["max_velocity"] = 40.0
        base_input["physical_constraints"]["max_pressure"] = 1000.0

        # 2. Config: aggressive dt, deep retry ladder
        base_config["dt"] = float(dt_initial)
        base_config["dt_min_limit"] = 1e-6
        base_config["ppe_max_retries"] = 20
        base_config["ppe_max_iter"] = 200
        base_config["ppe_tolerance"] = 1e-6

        input_filename = "test_elastic_terminal_fail.json"
        config_path = Path(BASE_DIR) / "config.json"
        input_path = Path(BASE_DIR) / input_filename

        config_path.write_text(json.dumps(base_config))
        input_path.write_text(json.dumps(base_input))

        with caplog.at_level(logging.INFO):
            with pytest.raises(RuntimeError) as excinfo:
                run_solver(input_filename)

        error_msg = str(excinfo.value)
        assert "CRITICAL INSTABILITY" in error_msg
        assert "Exhausted" in error_msg

        # Elasticity must have actually fired
        assert "STABILITY TRIGGER" in caplog.text
        # And the physical audit must have been involved
        assert "AUDIT [Explosion]" in caplog.text or "AUDIT [Limit]" in caplog.text

        input_path.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)

    def test_scenario_3_terminal_failure_hard_config(self, caplog, base_config, base_input):
        """
        Scenario 3: Force a crash quickly via hard configuration limits.
        This is a sharper, low-retry failure mode.
        """
        base_input["boundary_conditions"][0]["values"]["u"] = 50.0
        base_config["dt_min_limit"] = 0.01
        base_config["ppe_max_retries"] = 2

        input_filename = "test_terminal_fail.json"

        config_path = Path(BASE_DIR) / "config.json"
        input_path = Path(BASE_DIR) / input_filename

        config_path.write_text(json.dumps(base_config))
        input_path.write_text(json.dumps(base_input))

        with caplog.at_level(logging.WARNING):
            with pytest.raises(RuntimeError) as excinfo:
                run_solver(input_filename)

        error_msg = str(excinfo.value)
        assert "CRITICAL INSTABILITY" in error_msg
        assert "Exhausted" in error_msg

        assert "AUDIT [Limit]" in caplog.text
        assert "STABILITY TRIGGER" in caplog.text

        input_path.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)

    def test_scenario_4_retry_and_recover(self, caplog, base_config, base_input):
        """
        Scenario 4: Elasticity recovers (mock‑based).
        First audit fails → elasticity triggers.
        Second audit succeeds → solver commits and completes.
        """

        from unittest.mock import patch

        # Normal, stable physical setup (we are not testing physics here)
        input_filename = "test_recovery_input.json"
        config_path = Path(BASE_DIR) / "config.json"
        input_path = Path(BASE_DIR) / input_filename

        config_path.write_text(json.dumps(base_config))
        input_path.write_text(json.dumps(base_input))

        # --- MOCK STRATEGY ---
        # audit_physical_bounds() will:
        #   1st call → raise ArithmeticError (forces elasticity)
        #   2nd+ calls → succeed (allows recovery)
        side_effects = [ArithmeticError("Mocked explosion"), None, None, None]

        with patch(
            "src.common.solver_state.SolverState.audit_physical_bounds",
            side_effect=side_effects
        ):
            with caplog.at_level(logging.INFO):
                zip_path = run_solver(input_filename)

        # Solver must complete successfully
        assert zip_path is not None

        # Elasticity must have fired at least once
        trigger_found = any("STABILITY TRIGGER" in r.message for r in caplog.records)
        assert trigger_found, "Elasticity never fired; mock did not trigger retry logic."

        # No terminal instability should occur
        assert "CRITICAL INSTABILITY" not in caplog.text

        # Cleanup
        input_path.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)
