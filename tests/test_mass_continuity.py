# ==============================================================================
# Test Name: test_mass_continuity.py
# Description: End-to-end integration test verifying mass continuity (div(u) = 0) 
#              and numerical divergence constraints in 3D incompressible Navier-Stokes flow
#              using the unmocked Python application wrapper and archived numpy snapshots.
# ==============================================================================

# LITERATE TESTING NARRATIVE & MATHEMATICAL GOVERNING EQUATIONS:
# -------------------------------------------------------------
# Incompressible fluid flow is governed by the mass continuity equation, requiring
# the velocity field to remain divergence-free at all spatial coordinates:
#
#     \nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0
#
# To enforce this physical constraint, the Navier-Stokes orchestrator applies a pressure Poisson
# projection step during each time-step update.
#
# TEST SCENARIO:
# - Configures a uniform 10x10x10 Cartesian grid with equal spacing dx = dy = dz = 0.1 m.
# - Establishes solid boundary walls (mask = -1) along lateral boundaries and bottom, leaving z_max open for outflow.
# - Applies an outflow/lid boundary condition setup.
# - Executes the simulation via the unmocked Python application wrapper main().
# - Extracts archived velocity field snapshots from the output ZIP container.
# - Computes discrete central-difference velocity divergence across core interior fluid cells.
# - Asserts that local maximum divergence and net global mean divergence satisfy strict numerical thresholds.

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

# We import the unmocked main entry point of the Python application wrapper
from src.main import main


def test_mass_continuity_divergence(workspace_folder, monkeypatch):
    """
    Verifies that the Navier-Stokes solver successfully enforces the incompressible 
    mass continuity constraint (div(u) = 0) following the pressure Poisson projection step.
    """
    print("\n================================================================================")
    print("DIAGNOSTIC START: test_mass_continuity_divergence")
    print("================================================================================")

    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file
    output_manifest_name = "mass_continuity_manifest.json"

    print(f"[1/4] Loading input configuration from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # Configure a uniform 10x10x10 Cartesian grid with spacing dx = dy = dz = 0.1 m
    input_data["grid"] = {
        "x_min": 0.0,
        "x_max": 1.0,
        "y_min": 0.0,
        "y_max": 1.0,
        "z_min": 0.0,
        "z_max": 1.0,
        "nx": 10,
        "ny": 10,
        "nz": 10,
        "dx": 0.1,
        "dy": 0.1,
        "dz": 0.1
    }

    input_data["simulation_parameters"] = {
        "time_step": 0.001,
        "total_time": 0.003,
        "output_interval": 1
    }

    # Setup domain cell mask: solid walls (mask = -1) along lateral bounds and bottom (k=0),
    # keeping z_max (k = nz - 1) active to align with the outflow boundary condition.
    nx, ny, nz = 10, 10, 10
    total_cells = nx * ny * nz
    mask = [1] * total_cells
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0:
                    idx = k * (nx * ny) + j * nx + i
                    mask[idx] = -1

    input_data["mask"] = mask

    input_data["external_forces"] = {
        "force_vector": [0.0, 0.0, 0.0],
        "gravity_vector": [0.0, 0.0, 0.0]
    }

    input_data["boundary_conditions"] = [
        {"location": "wall", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}},
        {"location": "z_max", "type": "outflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0, "p": 0.0}}
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

    print("[2/4] Executing unmocked python wrapper main() for mass continuity evaluation...")
    try:
        main()
    except (RuntimeError, ValueError, OSError, ArithmeticError, SystemExit) as e:
        pytest.fail(f"Mass Continuity Test Failed: Pipeline threw unexpected exception: {e}")

    print("[3/4] Validating manifest status and extracting ZIP container output...")
    manifest_path = Path(folder) / output_manifest_name
    assert manifest_path.is_file(), f"Manifest file missing at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["results"]["status"] == "SUCCESS", "Mass Continuity Test Failed: Pipeline did not complete with SUCCESS status."

    zip_filename = manifest["results"]["zip_filename"]
    zip_path = Path(folder) / zip_filename
    assert zip_path.is_file(), f"ZIP archive missing at {zip_path}"

    print("[4/4] Computing discrete velocity divergence across core interior fluid domain...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        # Inspect final step snapshot (step 3)
        step_str = "000003"
        u_name = f"field_u_step_{step_str}.npy"
        v_name = f"field_v_step_{step_str}.npy"
        w_name = f"field_w_step_{step_str}.npy"

        assert u_name in namelist, f"Missing snapshot {u_name} in archive."
        assert v_name in namelist, f"Missing snapshot {v_name} in archive."
        assert w_name in namelist, f"Missing snapshot {w_name} in archive."

        u = np.load(io.BytesIO(zf.read(u_name))).reshape((nz, ny, nx))
        v = np.load(io.BytesIO(zf.read(v_name))).reshape((nz, ny, nx))
        w = np.load(io.BytesIO(zf.read(w_name))).reshape((nz, ny, nx))
        mask_arr = np.array(mask).reshape((nz, ny, nx))

        max_divergence = 0.0
        total_divergence = 0.0
        interior_fluid_count = 0

        dx = dy = dz = 0.1

        # We compute discrete velocity divergence across the core interior fluid domain 
        # using central differences (div(u) = du/dx + dv/dy + dw/dz), spanning 
        # correctly up to the boundary-adjacent layers:
        for k in range(1, nz - 1):
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    if mask_arr[k, j, i] == 1:
                        dudx = (u[k, j, i + 1] - u[k, j, i - 1]) / (2.0 * dx)
                        dvdy = (v[k, j + 1, i] - v[k, j - 1, i]) / (2.0 * dy)
                        dwdz = (w[k + 1, j, i] - w[k - 1, j, i]) / (2.0 * dz)

                        div_u = dudx + dvdy + dwdz

                        max_divergence = max(max_divergence, abs(div_u))
                        total_divergence += div_u
                        interior_fluid_count += 1

    assert interior_fluid_count > 0, "Error: No active fluid cells found in integration test domain."
    mean_divergence = total_divergence / float(interior_fluid_count)

    print(f"[debug] max_divergence = {max_divergence}, mean_divergence = {mean_divergence}")

    # Assertion 1: Local divergence in core fluid must remain below the strict numerical tolerance threshold.
    assert max_divergence < 2.0e-2, "Local mass conservation failure: Maximum velocity divergence exceeds physical tolerance."
    
    # Assertion 2: Global mean divergence across the domain must evaluate near zero.
    assert abs(mean_divergence) < 1.0e-5, "Global mass conservation failure: Net domain volume flux is non-zero."

    print("DIAGNOSTIC SUCCESS: Mass Continuity Integration Test validated successfully.")