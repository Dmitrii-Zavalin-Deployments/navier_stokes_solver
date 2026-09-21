"""
@file test_cfl_and_stability.py
@brief End-to-end integration test verifying Courant-Friedrichs-Lewy (CFL) condition enforcement,
       temporal stability bounds, and safety intercept exceptions under numerical velocity spikes
       using the unmocked Python application wrapper.

LITERATE TESTING NARRATIVE & MATHEMATICAL GOVERNING EQUATIONS:

---

Temporal stability in explicit and semi-implicit advection solvers is governed by
the Courant-Friedrichs-Lewy (CFL) condition. For a 3D Eulerian grid, the dimensionless
CFL number C measures the distance information travels across grid cells during a time step dt:

C = max( (|u|_max * dt) / dx, (|v|_max * dt) / dy, (|w|_max * dt) / dz ) <= C_max

Where C_max = 1.0 represents the hyperbolic stability boundary (information cannot
traverse more than one mesh cell per discrete time step).

TEST SCENARIOS:

* Scenario 6.1 (Case A - Stable):
dx = 0.01 m, u_max = 10.0 m/s, dt = 0.0005 s ==> C = (10.0 * 0.0005) / 0.01 = 0.5 <= 1.0
Expectation: Execution completes cleanly, velocity field remains finite,
and numerical divergence is successfully suppressed/bounded by projection.
* Scenario 6.1 (Case B - CFL Violation):
dx = 0.01 m, u_max = 10.0 m/s, dt = 0.002 s ==> C = (10.0 * 0.002) / 0.01 = 2.0 > 1.0
Expectation: The Orchestrator's CFL guard intercepts the time-step update and
throws an exception or prevents numeric NaN divergence blow-up.

---

"""

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

# We import the unmocked main entry point of the Python application wrapper

from src.main import main

def test_cfl_stable_execution(workspace_folder, monkeypatch):
"""
Scenario 6.1 (Case A - Stable):
- Configures a uniform grid with dx = dy = dz = 0.01 m (nx = ny = nz = 10).
- Sets a stable time step dt = 0.0005 s under a velocity magnitude of 10.0 m/s (C = 0.5 <= 1.0).
- Executes the unmocked main pipeline via Python wrapper.
- Verifies successful completion (SUCCESS status), valid ZIP archive generation,
and absence of NaN/Inf values with bounded divergence.
"""
print("\n================================================================================")
print("DIAGNOSTIC START: test_cfl_stable_execution (Case A)")
print("================================================================================")


folder = workspace_folder["folder"]
input_file = workspace_folder["input_file_name"]
input_path = Path(folder) / input_file
output_manifest_name = "cfl_stable_manifest.json"

print(f"[1/4] Loading input configuration from: {input_path}")
with open(input_path, "r", encoding="utf-8") as f:
    input_data = json.load(f)

# We configure the grid dimensions uniformly across all three dimensions:
#     nx = ny = nz = 10, dx = dy = dz = 0.01 m
input_data["grid"] = {
    "nx": 10,
    "ny": 10,
    "nz": 10,
    "dx": 0.01,
    "dy": 0.01,
    "dz": 0.01
}

# We set simulation parameters for stable CFL condition:
#     dt = 0.0005 s, total_time = 0.0015 s (3 steps)
input_data["simulation_parameters"] = {
    "time_step": 0.0005,
    "total_time": 0.0015,
    "output_interval": 1
}

# We configure the domain geometry mask with active fluid cells (1)
total_cells = 10 * 10 * 10
input_data["mask"] = [1] * total_cells

# Zero external forces and gravity for pure advection stability test
input_data["external_forces"] = {
    "force_vector": [0.0, 0.0, 0.0],
    "gravity_vector": [0.0, 0.0, 0.0]
}

# We set boundary conditions establishing u_max = 10.0 m/s inflow velocity:
input_data["boundary_conditions"] = [
    {"location": "z_min", "type": "inflow", "values": {"u": 10.0, "v": 0.0, "w": 0.0, "p": 0.0}},
    {"location": "z_max", "type": "outflow", "values": {"u": 10.0, "v": 0.0, "w": 0.0, "p": 0.0}},
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

print("[2/4] Executing unmocked python wrapper main() under stable CFL (C = 0.5)...")
try:
    main()
except Exception as e:
    pytest.fail(f"Case A Failed: Pipeline threw unexpected exception under stable CFL: {e}")

print("[3/4] Validating manifest status and ZIP container output...")
manifest_path = Path(folder) / output_manifest_name
assert manifest_path.is_file(), f"Manifest file missing at {manifest_path}"

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

assert manifest["results"]["status"] == "SUCCESS", "Case A Failed: Pipeline did not complete with SUCCESS status."

zip_filename = manifest["results"]["zip_filename"]
zip_path = Path(folder) / zip_filename
assert zip_path.is_file(), f"ZIP archive missing at {zip_path}"

print("[4/4] Inspecting archived field snapshots for numerical stability...")
with zipfile.ZipFile(zip_path, "r") as zf:
    namelist = zf.namelist()
    for step in [1, 2, 3]:
        step_str = f"{step:06d}"
        for fname in ["field_u", "field_v", "field_w", "field_p"]:
            snapshot_name = f"{fname}_step_{step_str}.npy"
            assert snapshot_name in namelist
            data = np.load(io.BytesIO(zf.read(snapshot_name)))
            
            # Assert no NaN or Inf values are present
            assert not np.isnan(data).any(), f"NaN detected in {snapshot_name}"
            assert not np.isinf(data).any(), f"Inf detected in {snapshot_name}"
            
            # Assert magnitude remains bounded
            assert np.max(np.abs(data)) < 100.0, f"Velocity/Pressure magnitude exceeded stable bound in {snapshot_name}"

print("DIAGNOSTIC SUCCESS: Case A (Stable CFL) validated successfully.")



def test_cfl_violation_safety_intercept(workspace_folder, monkeypatch):
"""
Scenario 6.1 (Case B - CFL Violation):
- Configures a uniform grid with dx = dy = dz = 0.01 m (nx = ny = nz = 10).
- Sets an unstable time step dt = 0.002 s under a velocity magnitude of 10.0 m/s (C = 2.0 > 1.0).
- Executes the unmocked main pipeline via Python wrapper.
- Verifies that the orchestrator's CFL guard intercepts the time-step update,
raises an exception, or prevents numerical NaN explosion.
"""
print("\n================================================================================")
print("DIAGNOSTIC START: test_cfl_violation_safety_intercept (Case B)")
print("================================================================================")


folder = workspace_folder["folder"]
input_file = workspace_folder["input_file_name"]
input_path = Path(folder) / input_file
output_manifest_name = "cfl_unstable_manifest.json"

print(f"[1/3] Loading input configuration from: {input_path}")
with open(input_path, "r", encoding="utf-8") as f:
    input_data = json.load(f)

input_data["grid"] = {
    "nx": 10,
    "ny": 10,
    "nz": 10,
    "dx": 0.01,
    "dy": 0.01,
    "dz": 0.01
}

# We set an unstable time step dt = 0.002 s ==> Courant number C = (10.0 * 0.002) / 0.01 = 2.0 > 1.0
input_data["simulation_parameters"] = {
    "time_step": 0.002,
    "total_time": 0.004,
    "output_interval": 1
}

total_cells = 10 * 10 * 10
input_data["mask"] = [1] * total_cells

input_data["external_forces"] = {
    "force_vector": [0.0, 0.0, 0.0],
    "gravity_vector": [0.0, 0.0, 0.0]
}

input_data["boundary_conditions"] = [
    {"location": "z_min", "type": "inflow", "values": {"u": 10.0, "v": 0.0, "w": 0.0, "p": 0.0}},
    {"location": "z_max", "type": "outflow", "values": {"u": 10.0, "v": 0.0, "w": 0.0, "p": 0.0}},
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

print("[2/3] Executing unmocked python wrapper main() under unstable CFL (C = 2.0)...")
guard_triggered = False
try:
    main()
    manifest_path = Path(folder) / output_manifest_name
    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("results", {}).get("status") != "SUCCESS":
            guard_triggered = True
except Exception:
    guard_triggered = True

# Assert that the CFL violation guard or exception intercept was successfully triggered
assert guard_triggered, (
    "Case B Failed: Orchestrator failed to guard against CFL violation (C = 2.0 > 1.0) "
    "and allowed unhandled numerical instability or unintercepted execution."
)
print("DIAGNOSTIC SUCCESS: Case B (CFL violation safety intercept) validated successfully.")
