# tests/property_integrity/test_solver_lifecycle.py

import json
import os
from pathlib import Path
from src.main_solver import BASE_DIR, run_solver
from tests.helpers.solver_input_schema_dummy import create_validated_input

class TestSolverLifecycle:
    """
    SYSTEM AUDITOR: Verifies the full pipeline flow 
    from Disk Input -> Numerical Loop -> Zip Archive.
    """

    def test_full_solver_pipeline_integrity(self, tmp_path, monkeypatch):
        # 1. Setup Isolated Environment (Rule 5)
        # We move the execution to a temp directory so we don't touch real files
        test_dir = tmp_path / "run_env"
        test_dir.mkdir()
        
        # Mock BASE_DIR in main_solver to point to our temp directory
        monkeypatch.setattr("src.main_solver.BASE_DIR", test_dir)

        # 2. Create Config (Deterministic numerical settings)
        config_file = test_dir / "config.json"
        config_dict = {
            "ppe_tolerance": 1e-4,
            "ppe_atol": 1e-6,
            "ppe_max_iter": 5,
            "ppe_omega": 1.2
        }
        config_file.write_text(json.dumps(config_dict))

        # 3. Create Input using SSoT Helper (Rule 4)
        # Use a tiny grid (3x3x3) to keep the system test fast
        input_obj = create_validated_input(nx=3, ny=3, nz=3)
        input_obj.simulation_parameters.total_time = 0.002
        input_obj.simulation_parameters.time_step = 0.001
        
        input_file = test_dir / "input_test.json"
        input_file.write_text(json.dumps(input_obj.to_dict()))

        # 4. Mock the Schema location
        # main_solver looks for 'schema/solver_input_schema.json' relative to BASE_DIR
        schema_dir = test_dir / "schema"
        schema_dir.mkdir()
        real_schema = Path(__file__).resolve().parent.parent.parent / "schema" / "solver_input_schema.json"
        (schema_dir / "solver_input_schema.json").write_text(real_schema.read_text())

        # 5. EXECUTION
        # We pass the filename string as run_solver expects
        os.chdir(test_dir)
        zip_path = run_solver("input_test.json")

        # 6. ASSERTIONS
        archive = Path(zip_path)
        assert archive.exists(), "CRITICAL: Pipeline finished but no zip archive found."
        assert archive.stat().st_size > 0, "CRITICAL: Archive is empty."
        
        # Log success for the audit trail
        print(f"\nDEBUG [Audit]: Successfully archived {archive.name}")