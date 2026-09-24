"""
Literate Testing Standard — recommended for repositories with complex scientific logic.
Each test file is written as a narrative: explanatory text appears as commented prose,
while formulas, numerical computations, and assertions appear as executable code.

Test Name: test_full_pipeline_constant_flow.py
Description: Validates the full Navier-Stokes solver pipeline under constant flow conditions 
using the end-to-end Python application interface (CLI / workspace orchestration).
"""

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

# Module under test
from src.main import main


def test_full_pipeline_constant_flow(workspace_folder, monkeypatch):
    """
    Executes end-to-end integration verifying constant flow conditions:
    - Configures an 8x8x4 simulation grid with a structured wall/fluid mask.
    - Sets inflow velocity profile (w = 1.0) and outflow/no-slip boundary conditions.
    - Executes the unmocked main pipeline via CLI arguments.
    - Validates output manifest status, ZIP container integrity, and field snapshot metrics.
    """
    print("\n================================================================================")
    print("DIAGNOSTIC START: test_full_pipeline_constant_flow")
    print("================================================================================")

    # We retrieve the workspace directory path and target input configuration file.
    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file
    output_manifest_name = "constant_flow_manifest.json"

    print(f"[1/5] Loading input configuration from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # We set simulation parameters across 3 time steps (dt = 0.1).
    input_data["simulation_parameters"] = {
        "time_step": 0.1,
        "total_time": 0.3,
        "output_interval": 1
    }

    # We configure the grid dimensions to an 8x8x4 domain.
    nx, ny, nz = 8, 8, 4
    input_data["grid"].update({"nx": nx, "ny": ny, "nz": nz})

    # We define the domain geometry mask across all 4 Z-layers, designating 
    # active fluid cells (1), solid/wall boundaries (-1), and external ghost cells (0).
    single_layer_mask = [
        0,  0,  0,  0,  0,  0,  0,  0,
        0, -1, -1, -1, -1, -1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1, -1, -1, -1, -1, -1,  0,
        0,  0,  0,  0,  0,  0,  0,  0
    ]
    input_data["mask"] = single_layer_mask * nz

    # We configure external forces to zero for pure constant flow.
    input_data["external_forces"]["force_vector"] = [0.0, 0.0, 0.0]

    # We set boundary conditions with inflow at z_min (w = 1.0), outflow at z_max, and no-slip walls.
    input_data["boundary_conditions"] = [
        {"location": "z_min", "type": "inflow", "values": {"u": 0.0, "v": 0.0, "w": 1.0, "p": 0.0}},
        {"location": "z_max", "type": "outflow", "values": {"u": 0.0, "v": 0.0, "w": 1.0, "p": 0.0}},
        {"location": "wall", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}}
    ]

    # We save the updated configuration parameters back to the input file.
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(input_data, f, indent=2)
    print("[1/5] Input configuration successfully updated with constant flow parameters.")

    print("[2/5] Configuring CLI arguments and executing unmocked main()...")
    cli_args = [
        "main.py",
        "--input_output_folder", folder,
        "--input_file_name", input_file,
        "--output_file_name", output_manifest_name,
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    try:
        main()
        print("[2/5] Pipeline execution completed successfully without unhandled exceptions.")
    except Exception as e:
        print(f"[CRITICAL ERROR] Pipeline execution failed with exception: {e}", file=sys.stderr)
        raise

    print("[3/5] Validating output JSON manifest structure and contents...")
    manifest_path = Path(folder) / output_manifest_name
    assert manifest_path.is_file(), f"Output JSON manifest not created at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"Manifest Status: {manifest.get('results', {}).get('status')}")
    print(f"Manifest ZIP Filename: {manifest.get('results', {}).get('zip_filename')}")

    assert manifest["results"]["status"] == "SUCCESS"
    print("[3/5] Manifest validation passed.")

    print("[4/5] Inspecting ZIP container contents and snapshot binaries...")
    zip_filename = manifest["results"]["zip_filename"]
    zip_path = Path(folder) / zip_filename
    assert zip_path.is_file(), f"ZIP archive missing at {zip_path}"

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        print(f"Found {len(namelist)} files inside ZIP archive:")
        for name in sorted(namelist):
            print(f"  - {name}")

        field_names = ["field_u", "field_v", "field_w", "field_p"]
        expected_steps = [1, 2, 3]

        print("[5/5] Performing deep metric extraction and verifying bounded stability across iterations...")
        for step in expected_steps:
            step_str = f"{step:06d}"
            print(f"\n--- Diagnostic Inspection: Step {step} (Tag: {step_str}) ---")

            for fname in field_names:
                snapshot_filename = f"{fname}_step_{step_str}.npy"
                assert snapshot_filename in namelist, f"Snapshot {snapshot_filename} missing from archive."

                # We load the binary array representation from the archived snapshot stream.
                raw_bytes = zf.read(snapshot_filename)
                field_data = np.load(io.BytesIO(raw_bytes))

                shape = field_data.shape
                min_val = float(np.min(field_data))
                max_val = float(np.max(field_data))
                mean_val = float(np.mean(field_data))
                abs_max = float(np.max(np.abs(field_data)))

                print(
                    f"  [{fname}] shape={shape} | min={min_val:.6f} | "
                    f"max={max_val:.6f} | mean={mean_val:.6f} | abs_max={abs_max:.6f}"
                )

                # 1. Check for structural corruption (NaNs or infinities)
                assert not np.isnan(field_data).any(), f"FATAL: NaN detected in {snapshot_filename}"
                assert not np.isinf(field_data).any(), f"FATAL: Inf detected in {snapshot_filename}"

                # 2. Assert anti-blow-up thresholds to guarantee fields remain physically bounded
                if fname == "field_p":
                    pressure_blowup_limit = 15.0
                    assert abs_max < pressure_blowup_limit, (
                        f"FATAL: Pressure field {snapshot_filename} is blowing up! "
                        f"abs_max={abs_max:.4f} exceeded limit {pressure_blowup_limit}"
                    )
                else:
                    velocity_blowup_limit = 5.0
                    assert abs_max < velocity_blowup_limit, (
                        f"FATAL: Velocity field {snapshot_filename} is blowing up! "
                        f"abs_max={abs_max:.4f} exceeded limit {velocity_blowup_limit}"
                    )

    print("\n================================================================================")
    print("DIAGNOSTIC SUCCESS: Constant flow pipeline validated successfully via Python wrapper!")
    print("================================================================================")