"""
Literate Integration Test: Scenario 2 - Accelerated Flow Field Integration.
Instrumented with deep diagnostics, step-by-step console printings, 
and granular assertions across every execution stage.
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
    validating CLI ingestion, C++ solver execution, archive artifact packaging,
    and step-by-step field statistics.
    """
    print("\n================================================================================")
    print("DIAGNOSTIC START: test_integration_accelerated_flow_pipeline")
    print("================================================================================")

    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file
    output_manifest_name = "accelerated_flow_manifest.json"

    print(f"[1/5] Loading input configuration from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # Configure simulation parameters and grid
    input_data["simulation_parameters"] = {"time_step": 0.001, "total_time": 0.003, "output_interval": 1}
    input_data["grid"].update({"nx": 4, "ny": 4, "nz": 4})
    
    layer_mask = [
        0, 0, 0, 0,
        0, 1, 1, 0,
        0, 1, 1, 0,
        0, 0, 0, 0
    ]
    input_data["mask"] = layer_mask * 4
    input_data["initial_conditions"]["velocity"] = [0.1, 0.1, 0.1]
    input_data["external_forces"]["force_vector"] = [1.0, 1.0, 1.0]
    input_data["external_forces"]["gravity_vector"] = [0.0, 0.0, 0.0]
    input_data["boundary_conditions"] = [
        {"location": "z_min", "type": "inflow", "values": {"u": 0.1, "v": 0.1, "w": 0.1, "p": 0.0}},
        {"location": "z_max", "type": "outflow", "values": {"u": 0.1, "v": 0.1, "w": 0.1, "p": 0.0}},
        {"location": "wall", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}}
    ]

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(input_data, f, indent=2)
    print("[1/5] Input configuration successfully updated with test parameters.")

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
    
    assert manifest["inputs"]["external_forces"]["force_vector"] == [1.0, 1.0, 1.0]
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

        # Since output_interval=1 and total_time=0.003 (dt=0.001), steps 1, 2, and 3 should exist.
        field_names = ["field_u", "field_v", "field_w", "field_p"]
        expected_steps = [1, 2, 3]

        print("[5/5] Performing deep metric extraction and verification across all steps...")
        for step in expected_steps:
            step_str = f"{step:06d}"
            print(f"\n--- Diagnostic Inspection: Step {step} (Tag: {step_str}) ---")
            
            for fname in field_names:
                snapshot_filename = f"{fname}_step_{step_str}.npy"
                assert snapshot_filename in namelist, f"Snapshot {snapshot_filename} missing from archive."

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

                # Structural & Numerical Stability Checks
                assert not np.isnan(field_data).any(), f"FATAL: NaN detected in {snapshot_filename}"
                assert not np.isinf(field_data).any(), f"FATAL: Inf detected in {snapshot_filename}"

                # Velocity Acceleration Check
                if fname in ["field_u", "field_v", "field_w"]:
                    print(f"    -> Evaluating acceleration rule for {fname}: abs_max ({abs_max:.6f}) > 0.1")
                    assert abs_max > 0.1, (
                        f"DIAGNOSTIC FAILURE: Velocity field '{fname}' at step {step} has max absolute value {abs_max}, "
                        f"which failed to exceed the initial baseline of 0.1. "
                        f"Root cause confirmation: The C++ solver core is either overwriting initial conditions with zeros "
                        f"or failing to accumulate external force vector [1.0, 1.0, 1.0]."
                    )

    print("\n================================================================================")
    print("DIAGNOSTIC SUCCESS: All integration tests and field assertions passed successfully!")
    print("================================================================================")
