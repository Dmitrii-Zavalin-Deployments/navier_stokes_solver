# ==============================================================================
# Test Name: test_no_penetration_clamping.py
# Description: End-to-end integration test verifying pressure-based velocity projection
#              (corrector step) at fluid-solid interfaces to prevent artificial wall
#              penetration in 3D incompressible Navier-Stokes flow.
# ==============================================================================

# LITERATE TESTING NARRATIVE & MATHEMATICAL GOVERNING EQUATIONS:
# -------------------------------------------------------------
# The Navier-Stokes corrector step projects trial velocities (u_star) using the pressure 
# gradient field to enforce the divergence-free continuity constraint:
#
#     u_new = u_star - (dt / rho) * grad(p)
#
# At a fluid-solid interface where a solid boundary blocks motion, the pressure gradient
# must analytically counteract any trial velocity directed toward the wall:
#
#     u_new = u_star - (dt / rho) * (dp / dx) = 0.0
#
# TEST SCENARIO:
# - Configures a small Cartesian grid domain with a designated solid wall boundary mask (-1).
# - Sets up simulation parameters and boundary conditions via the configuration dictionary.
# - Executes the simulation via the unmocked Python application wrapper main().
# - Extracts velocity field snapshots from the output ZIP container.
# - Asserts that the velocity normal to solid boundaries satisfies no-penetration clamping.

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

# We import the unmocked main entry point of the Python application wrapper
from src.main import main


def test_no_penetration_clamping(workspace_folder, monkeypatch):
    """
    Verifies that the pressure-velocity projection correctly enforces the no-penetration 
    condition at fluid-solid interfaces during the simulation pipeline.
    """
    print("\n================================================================================")
    print("DIAGNOSTIC START: test_no_penetration_clamping")
    print("================================================================================")

    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file
    output_manifest_name = "no_penetration_manifest.json"

    print(f"[1/4] Loading input configuration from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # Configure a small 5x5x5 Cartesian grid domain with equal spacing dx = dy = dz = 0.1 m
    nx, ny, nz = 5, 5, 5
    input_data["grid"] = {
        "x_min": 0.0,
        "x_max": 0.5,
        "y_min": 0.0,
        "y_max": 0.5,
        "z_min": 0.0,
        "z_max": 0.5,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx": 0.1,
        "dy": 0.1,
        "dz": 0.1
    }

    input_data["simulation_parameters"] = {
        "time_step": 0.01,
        "total_time": 0.01,
        "output_interval": 1
    }

    # Setup domain cell mask: active fluid cells (1), with solid walls (mask = -1) 
    # enclosing the outer boundaries to test interface clamping.
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
        "force_vector": [0.0, 0.0, 0.0],
        "gravity_vector": [0.0, 0.0, 0.0]
    }

    input_data["boundary_conditions"] = [
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

    print("[2/4] Executing unmocked python wrapper main() for no-penetration validation...")
    try:
        main()
    except (RuntimeError, ValueError, OSError, ArithmeticError, SystemExit) as e:
        pytest.fail(f"No-Penetration Clamping Test Failed: Pipeline threw unexpected exception: {e}")

    print("[3/4] Validating manifest status and extracting ZIP container output...")
    manifest_path = Path(folder) / output_manifest_name
    assert manifest_path.is_file(), f"Manifest file missing at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["results"]["status"] == "SUCCESS", "No-Penetration Clamping Test Failed: Pipeline did not complete with SUCCESS status."

    zip_filename = manifest["results"]["zip_filename"]
    zip_path = Path(folder) / zip_filename
    assert zip_path.is_file(), f"ZIP archive missing at {zip_path}"

    print("[4/4] Inspecting velocity field snapshots and verifying boundary clamping...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        step_str = "000001"
        u_name = f"field_u_step_{step_str}.npy"
        
        assert u_name in namelist, f"Missing snapshot {u_name} in archive."

        u = np.load(io.BytesIO(zf.read(u_name))).reshape((nz, ny, nx))
        mask_arr = np.array(mask).reshape((nz, ny, nx))

        # We verify that velocity components adjacent to solid walls satisfy 
        # the no-penetration clamping boundary condition (u -> 0.0 at walls).
        for k in range(1, nz - 1):
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    if mask_arr[k, j, i + 1] == -1 or mask_arr[k, j, i - 1] == -1:
                        local_u = abs(u[k, j, i])
                        assert local_u < 1e-1, "No-penetration violation: velocity at solid interface exceeds limit."

    print("DIAGNOSTIC SUCCESS: No-Penetration Clamping Integration Test validated successfully.")
