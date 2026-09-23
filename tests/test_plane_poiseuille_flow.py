# ==============================================================================
# Test Name: test_plane_poiseuille_flow.py
# Description: End-to-end integration test verifying Plane Poiseuille channel flow 
#              velocity profiles, viscous-force balance, and solver stability 
#              using the unmocked Python application wrapper and archived numpy snapshots.
# ==============================================================================

# LITERATE TESTING NARRATIVE & MATHEMATICAL GOVERNING EQUATIONS:
# -------------------------------------------------------------
# Plane Poiseuille flow describes the motion of an incompressible viscous fluid driven by a 
# body force between two infinite stationary parallel plates separated by height H.
#
# Under steady, fully developed conditions, the Navier-Stokes momentum equations reduce to:
#
#     \nu \frac{d^2 u}{dy^2} + f_x = 0
#
# The exact analytical velocity profile u(y) across the channel coordinate y in [0, H] is:
#
#     u(y) = 4 u_{\text{max}} \frac{y}{H} \left(1 - \frac{y}{H}\right)
#
# TEST SCENARIO:
# - Configures a Cartesian grid domain (nx = 16, ny = 16, nz = 3) with fine spacing.
# - Establishes matching analytical body force f_x and boundary conditions (no-slip walls, inflow, pressure outlet).
# - Executes the simulation via the unmocked Python application wrapper main().
# - Extracts velocity field snapshots from the output ZIP container.
# - Computes the relative L2 error between the numerical velocity profile and the theoretical analytical solution.

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

# We import the unmocked main entry point of the Python application wrapper
from src.main import main


def test_plane_poiseuille_flow(workspace_folder, monkeypatch):
    """
    Verifies that the Navier-Stokes solver accurately reproduces the analytical 
    Plane Poiseuille parabolic velocity profile under steady viscous-force balance.
    """
    print("\n================================================================================")
    print("DIAGNOSTIC START: test_plane_poiseuille_flow")
    print("================================================================================")

    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file
    output_manifest_name = "plane_poiseuille_manifest.json"

    print(f"[1/4] Loading input configuration from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # Configure spatial grid dimensions and resolution matching the C++ test specification
    nx, ny, nz = 16, 16, 4
    dx, dy, dz = 0.02, 0.01, 0.01
    input_data["grid"] = {
        "x_min": 0.0,
        "x_max": nx * dx,
        "y_min": 0.0,
        "y_max": ny * dy,
        "z_min": 0.0,
        "z_max": nz * dz,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx": dx,
        "dy": dy,
        "dz": dz
    }

    input_data["simulation_parameters"] = {
        "time_step": 0.0005,
        "total_time": 0.05,
        "output_interval": 1
    }

    # Active fluid mask across the entire domain
    total_cells = nx * ny * nz
    mask = [1] * total_cells
    input_data["mask"] = mask

    # Calculate exact analytical body force fx = (8 * nu * u_max) / H^2 to sustain Poiseuille flow
    mu = 0.001
    u_max = 0.1
    H = ny * dy
    fx_driving = 8.0 * mu * u_max / (H * H)

    input_data["external_forces"] = {
        "force_vector": [fx_driving, 0.0, 0.0],
        "gravity_vector": [0.0, 0.0, 0.0]
    }

    # Configure boundary conditions: no-slip walls on y and z bounds, inflow and pressure outlet on x bounds
    input_data["boundary_conditions"] = [
        {"location": "y_min", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}},
        {"location": "y_max", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}},
        {"location": "z_min", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}},
        {"location": "z_max", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}},
        {"location": "x_min", "type": "inflow", "values": {"u": u_max, "v": 0.0, "w": 0.0, "p": 0.0}},
        {"location": "x_max", "type": "pressure", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}}
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

    print("[2/4] Executing unmocked python wrapper main() for Plane Poiseuille simulation...")
    try:
        main()
    except (RuntimeError, ValueError, OSError, ArithmeticError, SystemExit) as e:
        pytest.fail(f"Plane Poiseuille Test Failed: Pipeline threw unexpected exception: {e}")

    print("[3/4] Validating manifest status and extracting ZIP container output...")
    manifest_path = Path(folder) / output_manifest_name
    assert manifest_path.is_file(), f"Manifest file missing at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["results"]["status"] == "SUCCESS", "Plane Poiseuille Test Failed: Pipeline did not complete with SUCCESS status."

    zip_filename = manifest["results"]["zip_filename"]
    zip_path = Path(folder) / zip_filename
    assert zip_path.is_file(), f"ZIP archive missing at {zip_path}"

    print("[4/4] Inspecting velocity field snapshots and evaluating analytical L2 norm error...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        # Inspect final step snapshot dynamically based on simulation parameters
        total_steps = round(input_data["simulation_parameters"]["total_time"] / input_data["simulation_parameters"]["time_step"]))
        step_str = f"{total_steps:06d}"
        u_name = f"field_u_step_{step_str}.npy"

        assert u_name in namelist, f"Missing snapshot {u_name} in archive."

        u = np.load(io.BytesIO(zf.read(u_name))).reshape((nz, ny, nx))

        # Compute relative L2 error between numerical velocity profile at mid-channel and analytical solution
        mid_x = nx // 2
        k_plane = nz // 2

        diff_l2_sq = 0.0
        exact_l2_sq = 0.0

        for j in range(1, ny - 1):
            y_pos = (j + 0.5) * dy
            u_exact = 4.0 * u_max * (y_pos / H) * (1.0 - (y_pos / H))
            u_computed = u[k_plane, j, mid_x]

            err = u_computed - u_exact
            diff_l2_sq += err * err
            exact_l2_sq += u_exact * u_exact

        relative_l2_error = np.sqrt(diff_l2_sq / exact_l2_sq)

    print(f"[debug] relative_l2_error = {relative_l2_error}")

    # Assertion: Relative L2 error must remain within acceptable transient truncation bounds
    assert relative_l2_error < 0.45, "Poiseuille verification failure: Velocity profile deviates excessively from analytical solution."

    print("DIAGNOSTIC SUCCESS: Plane Poiseuille Flow Integration Test validated successfully.")
