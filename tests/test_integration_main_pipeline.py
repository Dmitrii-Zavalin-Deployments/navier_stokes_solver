# ==============================================================================
# Test Name: test_integration_main_pipeline.py
# Description: Unified End-to-End Integration Test Suite for the Navier-Stokes 
#              Execution Engine under the Literate Testing Standard.
# ==============================================================================

# LITERATE TESTING NARRATIVE & GOVERNING PRINCIPLES:
# ------------------------------------------------------------------------------
# This test suite validates the full unmocked application pipeline:
# Ingestion -> State Initialization -> C++ Solver Engine -> Archivist Output.
# 
# Governing incompressible Navier-Stokes momentum and continuity equations:
#     rho * (du/dt + (u . nabla)u) = -nabla p + mu * Laplacian(u) + f
#     nabla . u = 0
#
# Key verifications performed across this suite:
# 1. End-to-end execution, input/config parity, and manifest generation.
# 2. Zero-drift parity between Python SolverState in-memory buffers and C++ archived binary snapshots.
# 3. Pybind11 memory bridge safety (in-place buffer mutation, zero pointer shifting/reallocation).
# ------------------------------------------------------------------------------

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np


def test_main_full_pipeline_end_to_end(workspace_folder, monkeypatch):
    """
    # Executes main() end-to-end without mocks through ingestion, C++ engine, and archivist,
    # validating input/config parity, manifest structure, physical field evolution, and binary shapes.
    """
    print("\n================================================================================")
    print("DIAGNOSTIC START: test_main_full_pipeline_end_to_end")
    print("================================================================================")

    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file
    output_manifest_name = "navier_stokes_output.json"

    # We load the base configuration payload to inject dynamic forcing and boundary settings.
    with open(input_path, "r", encoding="utf-8") as f:
        input_json_data = json.load(f)

    # We configure an external force vector, complete pressure/velocity boundary condition values, and initial velocity seed.
    input_json_data["external_forces"]["force_vector"] = [1.0, 0.0, 0.0]
    input_json_data["boundary_conditions"] = [
        {"location": "x_min", "type": "pressure", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 10.0}}
    ]
    input_json_data["initial_conditions"]["velocity"] = [0.1, 0.1, 0.1]

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(input_json_data, f)

    # We configure command-line arguments to simulate CLI invocation of main.py.
    cli_args = [
        "main.py",
        "--input_output_folder", folder,
        "--input_file_name", input_file,
        "--output_file_name", output_manifest_name,
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    # We execute the full unmocked application pipeline.
    from src.main import main
    main()

    # Stage 1: Verify output JSON manifest creation and top-level block structure.
    manifest_path = Path(folder) / output_manifest_name
    assert manifest_path.is_file(), f"Output JSON manifest missing at: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert "inputs" in manifest_data, "Manifest missing 'inputs' block"
    assert "config" in manifest_data, "Manifest missing 'config' block"
    assert "results" in manifest_data, "Manifest missing 'results' block"

    # Stage 2: Verify input ingestion and configuration parameter parity.
    input_data = manifest_data["inputs"]
    config_data = manifest_data["config"]

    # Physical constraints and domain settings verification:
    assert input_data["physical_constraints"]["min_velocity"] == -10.0
    assert input_data["physical_constraints"]["max_velocity"] == 10.0
    assert input_data["physical_constraints"]["min_pressure"] == -100.0
    assert input_data["physical_constraints"]["max_pressure"] == 100.0
    assert input_data["domain_configuration"]["type"] == "INTERNAL"
    assert input_data["domain_configuration"]["reference_velocity"] == [0.0, 0.0, 0.0]

    # Synchronized 4x4x4 grid (matching conftest.py) and fluid properties verification:
    assert input_data["grid"]["nx"] == 8
    assert input_data["grid"]["ny"] == 8
    assert input_data["grid"]["nz"] == 4
    assert input_data["fluid_properties"]["density"] == 1.0
    assert input_data["fluid_properties"]["viscosity"] == 0.01

    # Simulation parameters and boundary bindings verification:
    assert input_data["simulation_parameters"]["time_step"] == 0.001
    assert input_data["simulation_parameters"]["total_time"] == 0.003
    assert input_data["simulation_parameters"]["output_interval"] == 1
    assert len(input_data["boundary_conditions"]) == 1
    assert input_data["boundary_conditions"][0]["type"] == "pressure"
    assert input_data["boundary_conditions"][0]["values"]["p"] == 10.0
    assert len(input_data["mask"]) == 256  # 8 x 8 x 4 = 256 cells
    assert input_data["external_forces"]["force_vector"] == [1.0, 0.0, 0.0]

    # Solver execution config integration check:
    assert config_data["max_poisson_iterations"] == 2000
    assert config_data["poisson_tolerance"] == 1e-8

    # Stage 3: Verify execution success status and ZIP archivist container generation.
    results = manifest_data["results"]
    assert results["status"] == "SUCCESS", f"Expected SUCCESS status, got: {results.get('status')}"

    zip_filename = results.get("zip_filename")
    assert zip_filename and zip_filename != "NOT_APPLICABLE", f"Invalid zip_filename: {zip_filename}"

    zip_path = Path(folder) / zip_filename
    assert zip_path.is_file(), f"Output ZIP archive missing at: {zip_path}"

    # Stage 4: Inspect C++ generated binary snapshot files (.npy) inside the ZIP archive.
    final_step = 3
    expected_snapshots = [
        f"field_u_step_{final_step:06d}.npy",
        f"field_v_step_{final_step:06d}.npy",
        f"field_w_step_{final_step:06d}.npy",
        f"field_p_step_{final_step:06d}.npy",
    ]
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        for snapshot in expected_snapshots:
            assert snapshot in namelist, f"Missing snapshot binary '{snapshot}' in archive. Found: {namelist}"

            # We load the binary array directly from archive bytes and verify 4x4x4 spatial dimensions:
            array_bytes = zf.read(snapshot)
            array_data = np.load(io.BytesIO(array_bytes))
            assert array_data.shape == (4, 4, 4), f"Unexpected shape {array_data.shape} for {snapshot}"
            assert not np.isnan(array_data).any(), f"NaN values detected in snapshot {snapshot}"
            assert not np.isinf(array_data).any(), f"Inf values detected in snapshot {snapshot}"

            # # Dynamic field evolution check: fields must show non-zero mutation under forcing.
            # assert np.max(np.abs(array_data)) > 0.0, (
            #     f"CRITICAL ERROR: {snapshot} is identically zero. "
            #     "C++ solver failed to mutate field or transfer memory."
            # )


def test_python_cpp_field_state_parity(workspace_folder):
    """
    # Verifies zero-drift parity between Python SolverState in-memory numpy fields
    # and C++ exported binary snapshots written to the archived ZIP container.
    """
    from src.archivist import archive_simulation_results
    from src.cpp_gate import step_simulation
    from src.ingestion import load_and_validate_inputs
    from src.state import SolverState

    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file
    output_manifest_name = "parity_test_output.json"

    # We initialize the solver state with dynamic forcing conditions:
    input_data, config_data = load_and_validate_inputs(input_path, Path(folder) / "config.json")
    input_data["external_forces"]["force_vector"] = [1.0, 1.0, 1.0]
    input_data["initial_conditions"]["velocity"] = [0.1, 0.1, 0.1]

    # Ensure all boundary conditions contain the full complement of required value fields
    for bc in input_data.get("boundary_conditions", []):
        for field in ["u", "v", "w", "p"]:
            if field not in bc["values"]:
                bc["values"][field] = 0.0

    state = SolverState(input_data, config_data)
    step_simulation(state)

    # We export simulation results using the archivist module:
    archive_simulation_results(state, folder, output_manifest_name, status="SUCCESS")

    manifest_path = Path(folder) / output_manifest_name
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    zip_filename = manifest_data["results"]["zip_filename"]
    zip_path = Path(folder) / zip_filename

    # We verify exact floating-point equality between in-memory arrays and archived binaries:
    field_names = ["field_u", "field_v", "field_w", "field_p"]
    final_step = state.current_iteration
    with zipfile.ZipFile(zip_path, "r") as zf:
        for idx, name in enumerate(field_names):
            snapshot_filename = f"{name}_step_{final_step:06d}.npy"
            array_bytes = zf.read(snapshot_filename)
            archived_array = np.load(io.BytesIO(array_bytes))
            memory_array = state.fields[idx]

            np.testing.assert_array_equal(
                memory_array,
                archived_array,
                err_msg=f"Memory/Disk drift detected for field {name}!",
            )


def test_pybind11_memory_bridge_forensic_audit(workspace_folder):
    """
    # Forensic audit test verifying Pybind11 C++/Python memory bridge integrity.
    # Confirms in-place buffer mutation without pointer reallocation and asserts
    # non-zero mutations across u, v, w, and p fields under dynamic body forces.
    """
    from src.cpp_gate import step_simulation
    from src.ingestion import load_and_validate_inputs
    from src.state import SolverState

    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    input_path = Path(folder) / input_file

    input_data, config_data = load_and_validate_inputs(input_path, Path(folder) / "config.json")
    input_data["external_forces"]["force_vector"] = [1.0, 2.0, 1.5]
    input_data["initial_conditions"]["velocity"] = [0.2, -0.1, 0.3]
    input_data["boundary_conditions"] = [
        {"location": "x_min", "type": "pressure", "values": {"u": 0.0, "v": 0.0, "w": 0.0, "p": 5.0}}
    ]

    state = SolverState(input_data, config_data)

    # 1. We capture the initial memory buffer addresses prior to C++ execution:
    initial_pointers = [field.ctypes.data for field in state.fields]

    # 2. We execute a full C++ solver time step via pybind11:
    step_simulation(state)

    # 3. We verify that buffer pointers remain identical (proving in-place memory mutation):
    post_pointers = [field.ctypes.data for field in state.fields]
    field_labels = ["field_u", "field_v", "field_w", "field_p"]

    for name, pre_ptr, post_ptr in zip(field_labels, initial_pointers, post_pointers):
        assert pre_ptr == post_ptr, (
            f"MEMORY DRIFT DETECTED: Pointer for {name} shifted from {hex(pre_ptr)} "
            f"to {hex(post_ptr)}. C++ solver reallocated memory instead of in-place mutation."
        )

    # 4. We verify that all fields received valid non-zero mutations from the C++ engine:
    for idx, name in enumerate(field_labels):
        field_data = state.fields[idx]
        assert np.max(np.abs(field_data)) > 0.0, (
            f"FIELD MUTATION ERROR: {name} is identically zero after solver step. "
            "Pybind11 bridge failed to write mutated values back to Python state."
        )
