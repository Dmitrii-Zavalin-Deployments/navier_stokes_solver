"""
Literate Integration Test: Scenario 2 - Accelerated Flow Field Integration.
Validates the Navier-Stokes execution engine under constant body force acceleration
vector F = [1.0, 1.0, 1.0] across all main pipeline execution stages[cite: 1].

The momentum conservation equation governing the velocity field u under external body forces F is:
    du/dt + (u \cdot \nabla)u = -\nabla p + \nu \nabla^2 u + F
For a constant acceleration vector F = [1.0, 1.0, 1.0] N/kg applied to a 4x4x4 domain,
velocity components must monotonically increase beyond initial conditions u_0 = 0.1 m/s.
"""

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

# Module under test
from src.main import main


def test_integration_accelerated_flow_pipeline(workspace_folder, monkeypatch):
    """
    Executes end-to-end integration for accelerated flow on a 4x4x4 cubic grid,
    validating CLI ingestion, C++ solver execution, and archive artifact packaging.
    """
    # -------------------------------------------------------------------------
    # NARRATIVE SECTION 1: Input Payload Configuration & Spatial Setup
    # -------------------------------------------------------------------------
    # The computational domain is defined as a uniform cubic grid:
    #     V = nx * ny * nz = 4 * 4 * 4 = 64 cells
    # Initial velocities are initialized to u_0 = 0.1, v_0 = 0.1, w_0 = 0.1 m/s.
    # Constant body forces are injected as fx = 1.0, fy = 1.0, fz = 1.0 N/kg.
    # -------------------------------------------------------------------------
    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file
    output_manifest_name = "accelerated_flow_manifest.json"

    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    input_data["grid"].update({"nx": 4, "ny": 4, "nz": 4})
    input_data["mask"] = [0] * 64
    input_data["initial_conditions"]["velocity"] = [0.1, 0.1, 0.1]
    input_data["external_forces"]["force_vector"] = [1.0, 1.0, 1.0]
    input_data["external_forces"]["gravity_vector"] = [0.0, 0.0, 0.0]

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(input_data, f, indent=2)

    # -------------------------------------------------------------------------
    # NARRATIVE SECTION 2: CLI Invocation and Pipeline Execution
    # -------------------------------------------------------------------------
    # Configure command-line arguments to drive the unmocked python entry point.
    # -------------------------------------------------------------------------
    cli_args = [
        "main.py",
        "--input_output_folder", folder,
        "--input_file_name", input_file,
        "--output_file_name", output_manifest_name,
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    main()

    # -------------------------------------------------------------------------
    # NARRATIVE SECTION 3: Manifest Validation & Ingestion Compliance
    # -------------------------------------------------------------------------
    # Verify that the output manifest is generated and correctly reflects 
    # the requested [1.0, 1.0, 1.0] force vector configuration.
    # -------------------------------------------------------------------------
    manifest_path = Path(folder) / output_manifest_name
    assert manifest_path.is_file(), f"Output JSON manifest not created at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["inputs"]["external_forces"]["force_vector"] == [1.0, 1.0, 1.0]
    assert manifest["results"]["status"] == "SUCCESS"

    # -------------------------------------------------------------------------
    # NARRATIVE SECTION 4: ZIP Container Integrity & Field Acceleration Checks
    # -------------------------------------------------------------------------
    # Inspect archived NumPy binary snapshots for shape consistency (4x4x4),
    # absence of numerical explosions (NaN/Inf), and positive velocity acceleration 
    # exceeding the initial baseline of 0.1 m/s.
    # -------------------------------------------------------------------------
    zip_filename = manifest["results"]["zip_filename"]
    zip_path = Path(folder) / zip_filename
    assert zip_path.is_file(), f"ZIP archive missing at {zip_path}"

    final_step = 3
    field_names = ["field_u", "field_v", "field_w", "field_p"]

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in field_names:
            snapshot_filename = f"{name}_step_{final_step:06d}.npy"
            assert snapshot_filename in zf.namelist(), f"Snapshot {snapshot_filename} missing from archive."

            raw_bytes = zf.read(snapshot_filename)
            field_data = np.load(io.BytesIO(raw_bytes))

            assert field_data.shape == (4, 4, 4)
            assert not np.isnan(field_data).any(), f"NaN detected in {snapshot_filename}"
            assert not np.isinf(field_data).any(), f"Inf detected in {snapshot_filename}"

            if name in ["field_u", "field_v", "field_w"]:
                assert np.max(np.abs(field_data)) > 0.1, (
                    f"Velocity field {name} failed to accelerate under constant force."
                )