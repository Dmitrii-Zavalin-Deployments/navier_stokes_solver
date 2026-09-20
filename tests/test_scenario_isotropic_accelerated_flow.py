"""
Literate Integration Test: Scenario - Isotropic Multi-Axis Flow Acceleration.

This test validates unconstrained multi-axis acceleration within an expanded 
3D computational domain. When the fluid core volume is increased (e.g., on a 
6x6x6 grid with a spacious interior fluid region), the applied body force 
vector [1.0, 1.0, 1.0] drives physical acceleration across all spatial 
dimensions simultaneously.

Physical Principle:
    Given an external body force vector:
        F_ext = [f_x, f_y, f_z] = [1.0, 1.0, 1.0]
    
    In a sufficiently resolved interior fluid domain where boundary damping 
    does not immediately dominate transverse cells, Newton's second law and 
    the momentum equations dictate that velocity components in all directions 
    will accumulate momentum over successive time steps (dt):
        u(t+dt) > u(t)
        v(t+dt) > v(t)
        w(t+dt) > w(t)
"""

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

# Module under test
from src.main import main


def test_integration_isotropic_accelerated_flow(workspace_folder, monkeypatch):
    """
    Executes end-to-end integration verifying isotropic multi-axis flow acceleration:
    - Configures an expanded 6x6x6 grid with a widened interior fluid core.
    - Applies a uniform 3D body force vector [1.0, 1.0, 1.0].
    - Asserts that all velocity fields (field_u, field_v, field_w) successfully 
      accelerate beyond the initial baseline of 0.1.
    """
    print("\n================================================================================")
    print("DIAGNOSTIC START: test_integration_isotropic_accelerated_flow")
    print("================================================================================")

    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file
    output_manifest_name = "isotropic_acceleration_manifest.json"

    print(f"[1/5] Loading input configuration from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # We set the simulation time step (dt = 0.001) and total duration across 3 iterations.
    input_data["simulation_parameters"] = {"time_step": 0.001, "total_time": 0.003, "output_interval": 1}
    
    # We expand the grid dimensions to a 6x6x6 cubic domain to provide a larger fluid core.
    nx, ny, nz = 6, 6, 6
    input_data["grid"].update({"nx": nx, "ny": ny, "nz": nz})
    
    # We construct a 3D volumetric mask for the 6x6x6 grid where the z-axis 
    # remains open for inflow/outflow (mask = 1 across z) and transverse 
    # boundaries form enclosing walls (mask = 0).
    mask_grid = np.zeros((nz, ny, nx), dtype=int)
    mask_grid[:, 1:5, 1:5] = 1
    input_data["mask"] = mask_grid.flatten().tolist()
    
    # We apply an isotropic external body force vector across all three axes:
    #     F_ext = [1.0, 1.0, 1.0]
    input_data["external_forces"]["force_vector"] = [1.0, 1.0, 1.0]
    input_data["external_forces"]["gravity_vector"] = [0.0, 0.0, 0.0]
    
    # We set uniform initial inflow/outflow velocity seeds of 0.1 across all components.
    input_data["boundary_conditions"] = [
        {"location": "z_min", "type": "inflow", "values": {"u": 0.1, "v": 0.1, "w": 0.1, "p": 0.0}},
        {"location": "z_max", "type": "outflow", "values": {"u": 0.1, "v": 0.1, "w": 0.1, "p": 0.0}},
        {"location": "wall", "type": "no-slip", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}}
    ]

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(input_data, f, indent=2)
    print("[1/5] Input configuration successfully updated with isotropic 6x6x6 test parameters.")

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

        field_names = ["field_u", "field_v", "field_w", "field_p"]
        expected_steps = [1, 2, 3]

        print("[5/5] Performing deep metric extraction and verifying multi-axis acceleration...")
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

                # We assert structural and numerical stability (no NaNs or infinite values).
                assert not np.isnan(field_data).any(), f"FATAL: NaN detected in {snapshot_filename}"
                assert not np.isinf(field_data).any(), f"FATAL: Inf detected in {snapshot_filename}"

                # Isotropic Acceleration Verification:
                # All velocity components (u, v, w) must overcome the initial baseline 
                # of 0.1 under the influence of the 3D body force vector [1.0, 1.0, 1.0]:
                #     max(|vel|) > 0.1
                if fname in ["field_u", "field_v", "field_w"]:
                    print(f"    -> Evaluating isotropic acceleration rule for {fname}: abs_max ({abs_max:.6f}) > 0.1")
                    assert abs_max > 0.1, (
                        f"ASSERTION FAILURE: Velocity field '{fname}' at step {step} failed to accelerate "
                        f"across all axes (abs_max = {abs_max})."
                    )

    print("\n================================================================================")
    print("DIAGNOSTIC SUCCESS: Isotropic multi-axis flow acceleration validated successfully!")
    print("================================================================================")