# ==============================================================================
# Test Name: test_velocity_inlet_outlet.py
# Description: End-to-end integration test verifying velocity inlet and pressure 
#              outlet boundary coupling, mass flux conservation, and divergence-free 
#              flow constraints using the unmocked Python application wrapper.
# ==============================================================================

# LITERATE TESTING NARRATIVE & MATHEMATICAL GOVERNING EQUATIONS:
# -------------------------------------------------------------
# Incompressible internal channel flow with a specified velocity inlet and pressure 
# outlet is governed by mass continuity and momentum equations:
#
#     \nabla \cdot \mathbf{u} = 0
#
# Mass conservation across inlet and outlet faces requires exact flux balancing:
#
#     \dot{m}_{\text{inlet}}  = \sum (\rho \cdot u_{\text{inlet}} \cdot dy \cdot dz)
#     \dot{m}_{\text{outlet}} = \sum (\rho \cdot u_{\text{outlet}} \cdot dy \cdot dz)
#     \dot{m}_{\text{inlet}} == \dot{m}_{\text{outlet}}
#
# TEST SCENARIO:
# - Configures a Cartesian grid of dimensions 10 x 8 x 8 with spacing dx = dy = dz = 0.1 m.
# - Sets up a velocity inlet at x_min with uniform velocity u = 1.0 m/s.
# - Sets up a pressure outlet at x_max with reference pressure p = 0.0 Pa.
# - Executes the simulation via the unmocked Python application wrapper main().
# - Extracts archived velocity field snapshots from the output ZIP container.
# - Computes and verifies mass flow rate conservation between the inlet and outlet faces.

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

# We import the unmocked main entry point of the Python application wrapper
from src.main import main


def test_velocity_inlet_pressure_outlet(workspace_folder, monkeypatch):
    """
    Verifies robust coupling of velocity inlet and pressure outlet boundary conditions 
    and strict mass flux conservation across the fluid domain via the Python pipeline.
    """
    print("\n================================================================================")
    print("DIAGNOSTIC START: test_velocity_inlet_pressure_outlet")
    print("================================================================================")

    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file
    output_manifest_name = "velocity_inlet_outlet_manifest.json"

    print(f"[1/4] Loading input configuration from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # Configure 10 x 8 x 8 Cartesian grid with equal spacing dx = dy = dz = 0.1 m
    input_data["grid"] = {
        "x_min": 0.0,
        "x_max": 1.0,
        "y_min": 0.0,
        "y_max": 0.8,
        "z_min": 0.0,
        "z_max": 0.8,
        "nx": 10,
        "ny": 8,
        "nz": 8,
        "dx": 0.1,
        "dy": 0.1,
        "dz": 0.1
    }

    input_data["simulation_parameters"] = {
        "time_step": 0.001,
        "total_time": 0.005,
        "output_interval": 1
    }

    nx, ny, nz = 10, 8, 8
    total_cells = nx * ny * nz
    mask = [1] * total_cells

    input_data["mask"] = mask

    input_data["external_forces"] = {
        "force_vector": [0.0, 0.0, 0.0]
    }

    # Configure velocity inlet at x_min and pressure outlet at x_max
    input_data["boundary_conditions"] = [
        {
            "location": "x_min",
            "type": "inflow",
            "values": {"u": 1.0, "v": 0.0, "w": 0.0, "p": 0.0}
        },
        {
            "location": "x_max",
            "type": "outflow",
            "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}
        }
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

    print("[2/4] Executing unmocked python wrapper main() for inlet/outlet verification...")
    try:
        main()
    except (RuntimeError, ValueError, OSError, ArithmeticError, SystemExit) as e:
        pytest.fail(f"Velocity Inlet/Outlet Test Failed: Pipeline threw unexpected exception: {e}")

    print("[3/4] Validating manifest status and extracting ZIP container output...")
    manifest_path = Path(folder) / output_manifest_name
    assert manifest_path.is_file(), f"Manifest file missing at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["results"]["status"] == "SUCCESS", "Test Failed: Pipeline did not complete with SUCCESS status."

    zip_filename = manifest["results"]["zip_filename"]
    zip_path = Path(folder) / zip_filename
    assert zip_path.is_file(), f"ZIP archive missing at {zip_path}"

    print("[4/4] Computing mass flux conservation across inlet and outlet faces...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        # Inspect final step snapshot (step 5 for total_time=0.005 with time_step=0.001)
        step_str = "000005"
        u_name = f"field_u_step_{step_str}.npy"

        assert u_name in namelist, f"Missing snapshot {u_name} in archive."

        u = np.load(io.BytesIO(zf.read(u_name))).reshape((nz, ny, nx))

        density = 1000.0
        dy = dz = 0.1
        face_area = dy * dz

        inlet_mass_flow = 0.0
        outlet_mass_flow = 0.0

        # Sum mass flux at x_min face (i = 1) and x_max face (i = nx - 2)
        for k in range(1, nz - 1):
            for j in range(1, ny - 1):
                inlet_mass_flow += density * u[k, j, 1] * face_area
                outlet_mass_flow += density * u[k, j, nx - 2] * face_area

    print(f"[debug] inlet_mass_flow = {inlet_mass_flow}, outlet_mass_flow = {outlet_mass_flow}")

    # Assertion: Mass flow rate at inlet must match outlet within numerical tolerance
    assert abs(inlet_mass_flow - outlet_mass_flow) < 1e-1, "Mass flow conservation failure: Inlet mass rate does not match outlet mass rate."

    print("DIAGNOSTIC SUCCESS: Velocity Inlet / Pressure Outlet Integration Test validated successfully.")
