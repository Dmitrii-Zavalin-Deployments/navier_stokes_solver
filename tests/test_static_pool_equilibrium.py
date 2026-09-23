# ==============================================================================
# Test Name: test_static_pool_equilibrium.py
# Description: End-to-end integration test verifying static pool equilibrium 
#              and spurious current suppression under a constant downward 
#              gravitational field in 3D incompressible Navier-Stokes flow.
# ==============================================================================

# LITERATE TESTING NARRATIVE & MATHEMATICAL GOVERNING EQUATIONS:
# -------------------------------------------------------------
# Experiment Description:
# We configure a closed, rigid 3D rectangular fluid container filled with an 
# incompressible fluid under a constant downward gravitational field (g_y = -9.81 m/s^2). 
# The pressure field is initialized explicitly to satisfy the hydrostatic balance equation:
#
#     p(y) = -\rho \cdot g_y \cdot (y_{\text{top}} - y)
#
# Why We Are Doing It:
# In fractional-step Navier-Stokes solvers, numerical inconsistencies between the pressure 
# gradient discretization and gravity body force terms frequently induce artificial 
# acceleration, causing parasitic currents (spurious velocities) to spontaneously 
# emerge in a fluid that should otherwise remain completely static.
#
# What We Are Trying to Prove:
# We verify that the complete projection pipeline—comprising trial velocity prediction, 
# Poisson pressure solve, and divergence-free velocity correction—maintains exact hydrostatic 
# equilibrium. Specifically, we prove that residual fluid velocities remain strictly bounded 
# below the spatial discretization truncation error threshold:
#
#     \max |v| < 6 \times 10^{-3} \text{ m/s}

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

# We import the unmocked main entry point of the Python application wrapper
from src.main import main


def test_static_pool_equilibrium(workspace_folder, monkeypatch):
    """
    Verifies that the Navier-Stokes solver maintains static pool equilibrium 
    and suppresses spurious currents under hydrostatic pressure-gravity balance.
    """
    print("\n================================================================================")
    print("DIAGNOSTIC START: test_static_pool_equilibrium")
    print("================================================================================")

    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file
    output_manifest_name = "static_pool_manifest.json"

    print(f"[1/4] Loading input configuration from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # Configure a 9x9x9 Cartesian grid with uniform spacing dx = dy = dz = 1.0 m
    nx, ny, nz = 9, 9, 9
    dx, dy, dz = 1.0, 1.0, 1.0
    input_data["grid"] = {
        "x_min": 0.0,
        "x_max": float(nx * dx),
        "y_min": 0.0,
        "y_max": float(ny * dy),
        "z_min": 0.0,
        "z_max": float(nz * dz),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx": dx,
        "dy": dy,
        "dz": dz
    }

    input_data["simulation_parameters"] = {
        "time_step": 0.001,
        "total_time": 0.001,
        "output_interval": 1
    }

    # Setup closed rigid container mask: boundary walls (mask = -1), active fluid interior (mask = 1)
    total_cells = nx * ny * nz
    mask = [1] * total_cells
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
                    idx = k * (nx * ny) + j * nx + i
                    mask[idx] = -1

    input_data["mask"] = mask

    density = 1000.0
    gravity_y = -9.81
    input_data["fluid_properties"] = {
        "density": density,
        "viscosity": 0.001
    }

    input_data["external_forces"] = {
        "force_vector": [0.0, 0.0, 0.0],
        "gravity_vector": [0.0, gravity_y, 0.0]
    }

    # Initialize explicit hydrostatic pressure distribution: p(y) = -rho * g_y * (y_top - y)
    y_top = float(ny - 1) * dy
    initial_pressure = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                y_coord = float(j) * dy
                p_val = -density * gravity_y * (y_top - y_coord)
                initial_pressure.append(p_val)

    input_data["initial_conditions"] = {
        "u": [0.0] * total_cells,
        "v": [0.0] * total_cells,
        "w": [0.0] * total_cells,
        "p": initial_pressure
    }

    # Using schema-compliant 'pressure' boundary condition type at the top surface
    input_data["boundary_conditions"] = [
        {"location": "wall", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0}},
        {"location": "top", "type": "pressure", "values": {"p": 0.0}}
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

    print("[2/4] Executing unmocked python wrapper main() for static pool equilibrium evaluation...")
    try:
        main()
    except (RuntimeError, ValueError, OSError, ArithmeticError, SystemExit) as e:
        pytest.fail(f"Static Pool Equilibrium Test Failed: Pipeline threw unexpected exception: {e}")

    print("[3/4] Validating manifest status and extracting ZIP container output...")
    manifest_path = Path(folder) / output_manifest_name
    assert manifest_path.is_file(), f"Manifest file missing at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["results"]["status"] == "SUCCESS", "Static Pool Test Failed: Pipeline did not complete with SUCCESS status."

    zip_filename = manifest["results"]["zip_filename"]
    zip_path = Path(folder) / zip_filename
    assert zip_path.is_file(), f"ZIP archive missing at {zip_path}"

    print("[4/4] Measuring maximum residual vertical velocity across active fluid domain...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        step_str = "000001"
        v_name = f"field_v_step_{step_str}.npy"

        assert v_name in namelist, f"Missing snapshot {v_name} in archive."

        v = np.load(io.BytesIO(zf.read(v_name))).reshape((nz, ny, nx))
        mask_arr = np.array(mask).reshape((nz, ny, nx))

        max_v_final = 0.0
        active_fluid_count = 0

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    if mask_arr[k, j, i] == 1:
                        max_v_final = max(max_v_final, abs(v[k, j, i]))
                        active_fluid_count += 1

    assert active_fluid_count > 0, "Error: No active fluid cells found in integration test domain."
    print(f"[debug] max_v_final = {max_v_final}")

    # Assertion: Spurious velocity currents must remain strictly below the truncation error threshold
    assert max_v_final < 6.0e-3, "Static Pool Equilibrium Failure: Spurious currents generated in equilibrium."

    print("DIAGNOSTIC SUCCESS: Static Pool Equilibrium Integration Test validated successfully.")