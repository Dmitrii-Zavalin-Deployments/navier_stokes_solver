# ==============================================================================
# Test Name: test_free_slip_symmetry.py
# Description: End-to-end integration test verifying Free-Slip Symmetry Plane 
#              boundary enforcement (Scenario 3.4) using the unmocked Python 
#              application wrapper.
# ==============================================================================

# LITERATE TESTING NARRATIVE & MATHEMATICAL GOVERNING EQUATIONS:
# -------------------------------------------------------------
# Free-slip symmetry boundaries enforce zero normal velocity and zero shear stress
# gradients across the boundary plane. For a boundary aligned with the y-axis (y_max):
#
#     v_normal = v = 0.0
#     du/dy = 0, dw/dy = 0
#
# TEST SCENARIO:
# * Scenario 3.4 (Free-Slip Symmetry Plane Verification):
#   - Grid dimensions: nx = 6, ny = 8, nz = 6, dx = dy = dz = 0.1 m.
#   - Initializes non-zero normal velocity (v = 1.0 m/s) to verify that pre-step/solver
#     boundary enforcement correctly zeroes out the normal velocity component 
#     at the y_max symmetry plane.
#   - Expectation: Execution completes successfully (SUCCESS status) and archived 
#     snapshots confirm strict zero normal velocity (\(v = 0.0\)) along the \(y_{max}\) boundary.

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

# We import the unmocked main entry point of the Python application wrapper
from src.main import main


def test_free_slip_symmetry_plane(workspace_folder, monkeypatch):
    """
    Scenario 3.4: Free-Slip Symmetry Plane Boundary Verification
    - Configures a uniform grid with nx = 6, ny = 8, nz = 6 and dx = dy = dz = 0.1 m.
    - Configures a free-slip boundary condition at the y_max boundary.
    - Executes the unmocked main pipeline via Python wrapper.
    - Verifies successful completion (SUCCESS status), valid ZIP archive generation,
      and that normal velocity v is strictly zero across all cells at the y_max boundary plane.
    """
    print("\n================================================================================")
    print("DIAGNOSTIC START: test_free_slip_symmetry_plane (Scenario 3.4)")
    print("================================================================================")

    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file
    output_manifest_name = "free_slip_symmetry_manifest.json"

    print(f"[1/4] Loading input configuration from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # We configure the grid dimensions matching Scenario 3.4:
    #     nx = 6, ny = 8, nz = 6, dx = dy = dz = 0.1 m
    input_data["grid"] = {
        "x_min": 0.0,
        "x_max": 0.6,
        "y_min": 0.0,
        "y_max": 0.8,
        "z_min": 0.0,
        "z_max": 0.6,
        "nx": 6,
        "ny": 8,
        "nz": 6,
        "dx": 0.1,
        "dy": 0.1,
        "dz": 0.1
    }

    # We set simulation time parameters:
    #     dt = 0.001 s, total_time = 0.003 s (3 steps)
    input_data["simulation_parameters"] = {
        "time_step": 0.001,
        "total_time": 0.003,
        "output_interval": 1
    }

    # Active fluid domain mask
    total_cells = 6 * 8 * 6
    input_data["mask"] = [1] * total_cells

    # Zero external forces and gravity
    input_data["external_forces"] = {
        "force_vector": [0.0, 0.0, 0.0],
        "gravity_vector": [0.0, 0.0, 0.0]
    }

    # We configure boundary conditions including free-slip symmetry at y_max:
    input_data["boundary_conditions"] = [
        {"location": "y_max", "type": "free-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}},
        {"location": "y_min", "type": "wall", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}},
        {"location": "x_min", "type": "outflow", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}},
        {"location": "x_max", "type": "outflow", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}},
        {"location": "z_min", "type": "outflow", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}},
        {"location": "z_max", "type": "outflow", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}}
    ]

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(input_data, f, indent=2)

    cli_args = [
        "main.py",
        "--input_output_folder", folder,
        "--input_file_name", input_file,
        "--output_file_name", output_manifest_name,
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    print("[2/4] Executing unmocked python wrapper main() for free-slip symmetry verification...")
    try:
        main()
    except (RuntimeError, ValueError, OSError, ArithmeticError) as e:
        pytest.fail(f"Scenario 3.4 Failed: Pipeline threw unexpected exception: {e}")

    print("[3/4] Validating manifest status and ZIP container output...")
    manifest_path = Path(folder) / output_manifest_name
    assert manifest_path.is_file(), f"Manifest file missing at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["results"]["status"] == "SUCCESS", "Scenario 3.4 Failed: Pipeline did not complete with SUCCESS status."

    zip_filename = manifest["results"]["zip_filename"]
    zip_path = Path(folder) / zip_filename
    assert zip_path.is_file(), f"ZIP archive missing at {zip_path}"

    print("[4/4] Inspecting archived field snapshots for zero normal velocity at y_max symmetry boundary...")
    nx, ny, nz = 6, 8, 6
    top_j = ny - 1

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        for step in [1, 2, 3]:
            step_str = f"{step:06d}"
            snapshot_name = f"field_v_step_{step_str}.npy"
            assert snapshot_name in namelist, f"Snapshot {snapshot_name} missing from ZIP archive"
            
            v_data = np.load(io.BytesIO(zf.read(snapshot_name))).flatten()
            
            # Verify normal velocity v is exactly zero across all cells at the y_max boundary (top row)
            for k in range(nz):
                for i in range(nx):
                    boundary_idx = i + nx * (top_j + ny * k)
                    val = v_data[boundary_idx]
                    assert abs(val) < 1e-7, (
                        f"Free-slip symmetry failure at step {step} (i={i}, k={k}): "
                        f"Normal velocity v = {val} is non-zero at y_max boundary."
                    )

    print("DIAGNOSTIC SUCCESS: Scenario 3.4 (Free-Slip Symmetry Plane) validated successfully.")
