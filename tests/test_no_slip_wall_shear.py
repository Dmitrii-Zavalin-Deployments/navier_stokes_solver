# ==============================================================================
# Test Name: test_no_slip_wall_shear.py
# Description: End-to-end integration test verifying no-slip wall shear boundary conditions 
#              and viscous boundary layer deceleration in 3D incompressible Navier-Stokes flow
#              using the unmocked Python application wrapper and archived numpy snapshots.
# ==============================================================================

# LITERATE TESTING NARRATIVE & MATHEMATICAL GOVERNING EQUATIONS:
# -------------------------------------------------------------
# No-slip wall boundaries enforce absolute velocity stagnation at solid boundaries:
#
#     u_wall = 0.0, v_wall = 0.0, w_wall = 0.0
#
# Furthermore, viscous shear forces create a boundary layer velocity profile where 
# fluid velocity monotonically decreases from the core flow towards the stationary wall:
#
#     u_near_wall < u_core
#
# TEST SCENARIO:
# - Configures a Cartesian grid domain (nx = 6, ny = 8, nz = 6) with a solid wall mask at y_min (j = 0).
# - Sets up simulation parameters and boundary conditions via the configuration dictionary.
# - Executes the simulation via the unmocked Python application wrapper main().
# - Extracts velocity field snapshots from the output ZIP container.
# - Asserts that near-wall velocity decelerates relative to the core flow due to viscous shear.

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

# We import the unmocked main entry point of the Python application wrapper
from src.main import main


def test_no_slip_wall_shear(workspace_folder, monkeypatch):
    """
    Verifies that no-slip wall boundaries and viscous shear create the expected 
    boundary layer velocity profile during the simulation pipeline.
    """
    print("\n================================================================================")
    print("DIAGNOSTIC START: test_no_slip_wall_shear")
    print("================================================================================")

    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file
    output_manifest_name = "no_slip_wall_shear_manifest.json"

    print(f"[1/4] Loading input configuration from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # Configure a Cartesian grid domain: nx = 6, ny = 8, nz = 6 with dx = dy = dz = 0.1 m
    nx, ny, nz = 6, 8, 6
    input_data["grid"] = {
        "x_min": 0.0,
        "x_max": 0.6,
        "y_min": 0.0,
        "y_max": 0.8,
        "z_min": 0.0,
        "z_max": 0.6,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx": 0.1,
        "dy": 0.1,
        "dz": 0.1
    }

    input_data["simulation_parameters"] = {
        "time_step": 0.001,
        "total_time": 0.002,
        "output_interval": 1
    }

    # Setup domain cell mask: active fluid cells (1), with solid walls (mask = -1) 
    # at the y_min boundary (j = 0) and enclosing outer boundaries.
    total_cells = nx * ny * nz
    mask = [1] * total_cells
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
                    idx = k * (nx * ny) + j * nx + i
                    mask[idx] = -1

    input_data["mask"] = mask

    input_data["external_forces"] = {
        "force_vector": [1.0, 0.0, 0.0],  # Driving force in x-direction to establish shear flow
        "gravity_vector": [0.0, 0.0, 0.0]
    }

    input_data["boundary_conditions"] = [
        {"location": "y_min", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}},
        {"location": "wall", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}}
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

    print("[2/4] Executing unmocked python wrapper main() for no-slip wall shear validation...")
    try:
        main()
    except (RuntimeError, ValueError, OSError, ArithmeticError, SystemExit) as e:
        pytest.fail(f"No-Slip Wall Shear Test Failed: Pipeline threw unexpected exception: {e}")

    print("[3/4] Validating manifest status and extracting ZIP container output...")
    manifest_path = Path(folder) / output_manifest_name
    assert manifest_path.is_file(), f"Manifest file missing at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["results"]["status"] == "SUCCESS", "No-Slip Wall Shear Test Failed: Pipeline did not complete with SUCCESS status."

    zip_filename = manifest["results"]["zip_filename"]
    zip_path = Path(folder) / zip_filename
    assert zip_path.is_file(), f"ZIP archive missing at {zip_path}"

    print("[4/4] Inspecting velocity field snapshots and verifying boundary layer shear profile...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        step_str = "000002"
        u_name = f"field_u_step_{step_str}.npy"
        
        assert u_name in namelist, f"Missing snapshot {u_name} in archive."

        u = np.load(io.BytesIO(zf.read(u_name))).reshape((nz, ny, nx))

        # We evaluate near-wall deceleration relative to the core flow due to viscous shear:
        k_idx = nz // 2
        i_idx = nx // 2
        near_wall_u = u[k_idx, 1, i_idx]
        core_u = u[k_idx, ny - 2, i_idx]

        assert near_wall_u < core_u, "Viscous boundary layer failure: Near-wall velocity did not decelerate relative to core flow."

    print("DIAGNOSTIC SUCCESS: No-Slip Wall Shear Integration Test validated successfully.")
