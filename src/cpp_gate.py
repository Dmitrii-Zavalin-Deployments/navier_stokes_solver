"""
src/cpp_gate.py
C++ Interaction Wrapper Module with Exhaustive Forensic Tracing for u, v, w, p fields.
"""

import logging
import sys
from typing import Any

import numpy as np

from src.state import SolverState

try:
    import navier_stokes_cpp
except ImportError as e:
    raise ImportError(
        "Failed to import compiled C++ module 'navier_stokes_cpp'. "
        "Ensure the C++ library is built and Python path includes the build directory."
    ) from e

logger = logging.getLogger("Solver.CppGate")
# Ensure logs print out to stdout immediately during pytest
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _dict_to_boundary_condition(bc_dict: dict) -> Any:
    """Instantiates and populates a C++ BoundaryCondition object from a Python dict."""
    logger.info(f"[FORENSIC TRACE] _dict_to_boundary_condition called with: {bc_dict}")
    if not isinstance(bc_dict, dict):
        raise TypeError("Boundary condition configuration must be a dictionary.")

    if "location" not in bc_dict:
        raise KeyError("Boundary condition configuration missing required field 'location'.")
    if "type" not in bc_dict:
        raise KeyError("Boundary condition configuration missing required field 'type'.")

    bc_obj = navier_stokes_cpp.BoundaryCondition()

    if hasattr(bc_obj, "location"):
        try:
            bc_obj.location = str(bc_dict["location"])
        except (AttributeError, TypeError) as err:
            logger.debug(f"Could not set 'location' attribute on BoundaryCondition: {err}")

    if hasattr(bc_obj, "type"):
        try:
            bc_obj.type = str(bc_dict["type"])
        except (AttributeError, TypeError) as err:
            logger.debug(f"Could not set 'type' attribute on BoundaryCondition: {err}")

    vals = bc_dict.get("values", bc_dict)
    missing_fields = [k for k in ["u", "v", "w", "p"] if k not in vals]
    if missing_fields:
        raise KeyError(
            f"Boundary condition for '{bc_dict['location']}' missing required value field(s): {missing_fields}."
        )

    u_val = float(vals["u"])
    v_val = float(vals["v"])
    w_val = float(vals["w"])
    p_val = float(vals["p"])

    if hasattr(bc_obj, "values"):
        val_obj = bc_obj.values
        if val_obj is not None:
            for k, val in [("u", u_val), ("v", v_val), ("w", w_val), ("p", p_val)]:
                if hasattr(val_obj, k):
                    try:
                        setattr(val_obj, k, val)
                    except (AttributeError, TypeError) as err:
                        logger.debug(f"Skipping nested attribute 'values.{k}': {err}")

    field_map = {
        "u": u_val, "u_val": u_val,
        "v": v_val, "v_val": v_val,
        "w": w_val, "w_val": w_val,
        "p": p_val, "p_val": p_val, "scalar_p": p_val
    }
    for attr, val in field_map.items():
        if hasattr(bc_obj, attr):
            try:
                setattr(bc_obj, attr, val)
            except (AttributeError, TypeError) as err:
                logger.debug(f"Skipping Pybind11 attribute '{attr}' on BoundaryCondition: {err}")

    logger.info(f"[FORENSIC TRACE] Successfully created BoundaryCondition object for location={bc_dict.get('location')}")
    return bc_obj


def _log_field_max_abs(tag: str, state: SolverState) -> None:
    """Helper to log maximum absolute values across all primary fields (u, v, w, p)."""
    u_max = float(np.max(np.abs(state.u))) if hasattr(state, "u") and state.u is not None else 0.0
    v_max = float(np.max(np.abs(state.v))) if hasattr(state, "v") and state.v is not None else 0.0
    w_max = float(np.max(np.abs(state.w))) if hasattr(state, "w") and state.w is not None else 0.0
    p_max = float(np.max(np.abs(state.p))) if hasattr(state, "p") and state.p is not None else 0.0
    logger.info(
        f"[FORENSIC TRACE] {tag} -> u max abs: {u_max:.6f} | v max abs: {v_max:.6f} | "
        f"w max abs: {w_max:.6f} | p max abs: {p_max:.6f}"
    )


def _apply_initial_boundary_conditions(state: SolverState) -> None:
    """Enforces initial boundary condition values onto array boundary faces with exhaustive logging for u, v, w, p."""
    logger.info("[FORENSIC TRACE] === Entering _apply_initial_boundary_conditions ===")
    
    def process_raw_bcs(raw_bcs):
        if not raw_bcs:
            return False
        logger.info(f"[FORENSIC TRACE] Found {len(raw_bcs)} raw boundary condition definitions.")

        for idx, bc in enumerate(raw_bcs):
            if isinstance(bc, dict):
                if "location" not in bc or "type" not in bc:
                    raise KeyError("Boundary condition item missing required 'location' or 'type' key.")
                loc = str(bc["location"]).lower()
                bc_type = str(bc["type"]).lower()
                vals = bc.get("values", bc)
            else:
                loc = str(getattr(bc, "location", "")).lower()
                bc_type = str(getattr(bc, "type", "")).lower()
                if not loc or not bc_type:
                    raise KeyError("BoundaryCondition object missing required 'location' or 'type' attribute.")

                vals = {}
                for k, attr_names in [("u", ["u", "u_val"]), ("v", ["v", "v_val"]), ("w", ["w", "w_val"]), ("p", ["p", "p_val", "scalar_p"])]:
                    val = None
                    if hasattr(bc, "values") and bc.values is not None:
                        val_sub = bc.values
                        if hasattr(val_sub, k):
                            val = getattr(val_sub, k)
                    if val is None:
                        for name in attr_names:
                            if hasattr(bc, name):
                                val = getattr(bc, name)
                                break
                    if val is None:
                        raise KeyError(f"Boundary condition object for '{loc}' missing required attribute '{k}'.")
                    vals[k] = val

            logger.info(f"[FORENSIC TRACE] BC #{idx}: loc='{loc}', type='{bc_type}', vals={vals}")

            # Case-insensitive and substring match for inflow/prescribed types
            if "inflow" in bc_type or "prescribed" in bc_type or bc_type in ["inflow", "prescribed"]:
                missing_vals = [k for k in ["u", "v", "w", "p"] if k not in vals]
                if missing_vals:
                    raise KeyError(f"Inflow boundary condition '{loc}' missing required values {missing_vals}.")

                u_val = float(vals["u"])
                v_val = float(vals["v"])
                w_val = float(vals["w"])
                p_val = float(vals["p"])

                _log_field_max_abs(f"Before applying {loc}", state)

                if "x_min" in loc or "xmin" in loc:
                    state.u[0, :, :] = u_val
                    state.v[0, :, :] = v_val
                    state.w[0, :, :] = w_val
                    state.p[0, :, :] = p_val
                    if hasattr(state, "fields") and state.fields is not None:
                        state.fields[0, 0, :, :] = u_val
                        state.fields[1, 0, :, :] = v_val
                        state.fields[2, 0, :, :] = w_val
                        state.fields[3, 0, :, :] = p_val
                elif "x_max" in loc or "xmax" in loc:
                    state.u[-1, :, :] = u_val
                    state.v[-1, :, :] = v_val
                    state.w[-1, :, :] = w_val
                    state.p[-1, :, :] = p_val
                    if hasattr(state, "fields") and state.fields is not None:
                        state.fields[0, -1, :, :] = u_val
                        state.fields[1, -1, :, :] = v_val
                        state.fields[2, -1, :, :] = w_val
                        state.fields[3, -1, :, :] = p_val
                elif "y_min" in loc or "ymin" in loc:
                    state.u[:, 0, :] = u_val
                    state.v[:, 0, :] = v_val
                    state.w[:, 0, :] = w_val
                    state.p[:, 0, :] = p_val
                    if hasattr(state, "fields") and state.fields is not None:
                        state.fields[0, :, 0, :] = u_val
                        state.fields[1, :, 0, :] = v_val
                        state.fields[2, :, 0, :] = w_val
                        state.fields[3, :, 0, :] = p_val
                elif "y_max" in loc or "ymax" in loc:
                    state.u[:, -1, :] = u_val
                    state.v[:, -1, :] = v_val
                    state.w[:, -1, :] = w_val
                    state.p[:, -1, :] = p_val
                    if hasattr(state, "fields") and state.fields is not None:
                        state.fields[0, :, -1, :] = u_val
                        state.fields[1, :, -1, :] = v_val
                        state.fields[2, :, -1, :] = w_val
                        state.fields[3, :, -1, :] = p_val
                elif "z_min" in loc or "zmin" in loc:
                    state.u[:, :, 0] = u_val
                    state.v[:, :, 0] = v_val
                    state.w[:, :, 0] = w_val
                    state.p[:, :, 0] = p_val
                    if hasattr(state, "fields") and state.fields is not None:
                        state.fields[0, :, :, 0] = u_val
                        state.fields[1, :, :, 0] = v_val
                        state.fields[2, :, :, 0] = w_val
                        state.fields[3, :, :, 0] = p_val
                elif "z_max" in loc or "zmax" in loc:
                    state.u[:, :, -1] = u_val
                    state.v[:, :, -1] = v_val
                    state.w[:, :, -1] = w_val
                    state.p[:, :, -1] = p_val
                    if hasattr(state, "fields") and state.fields is not None:
                        state.fields[0, :, :, -1] = u_val
                        state.fields[1, :, :, -1] = v_val
                        state.fields[2, :, :, -1] = w_val
                        state.fields[3, :, :, -1] = p_val

                _log_field_max_abs(
                    f"AFTER applying {loc} (Target vals: u={u_val}, v={v_val}, w={w_val}, p={p_val})",
                    state,
                )
            else:
                logger.info(
                    f"[FORENSIC TRACE] BC #{idx} type '{bc_type}' skipped for direct array assignment."
                )
        return True

    executed = False
    bcs_1 = getattr(state, "boundary_conditions", None)
    if bcs_1:
        process_raw_bcs(bcs_1)
        executed = True

    input_data = getattr(state, "input_data", None)
    bcs_2 = input_data.get("boundary_conditions") if isinstance(input_data, dict) else None
    if bcs_2 and bcs_2 != bcs_1:
        process_raw_bcs(bcs_2)
        executed = True

    if not executed:
        raise KeyError("FATAL ERROR: Boundary conditions configuration missing from SolverState or input_data.")

    logger.info("[FORENSIC TRACE] === Exiting _apply_initial_boundary_conditions ===")


def _convert_boundary_conditions(state: SolverState) -> None:
    """Converts boundary conditions to C++ BoundaryCondition objects in-place."""
    logger.info("[FORENSIC TRACE] === Entering _convert_boundary_conditions ===")
    _apply_initial_boundary_conditions(state)

    bcs_1 = getattr(state, "boundary_conditions", None)
    input_data = getattr(state, "input_data", None)
    bcs_2 = input_data.get("boundary_conditions") if isinstance(input_data, dict) else None

    raw_bcs = bcs_1 if bcs_1 else bcs_2
    if not raw_bcs:
        raise KeyError("FATAL ERROR: Boundary conditions configuration missing from SolverState or input_data.")

    state.boundary_conditions = [
        _dict_to_boundary_condition(bc) if isinstance(bc, dict) else bc
        for bc in raw_bcs
    ]
    logger.info("[FORENSIC TRACE] === Exiting _convert_boundary_conditions ===")


def _get_or_create_cpp_solver(state: SolverState) -> Any:
    """
    Instance-bound initializer for the underlying C++ NavierStokesSolver engine with comprehensive u, v, w, p tracing.
    """
    logger.info("[FORENSIC TRACE] === Entering _get_or_create_cpp_solver ===")
    if state is None:
        raise ValueError("FATAL ERROR: state must be explicitly provided.")

    has_solver = hasattr(state, "_cpp_solver") and state._cpp_solver is not None
    logger.info(f"[FORENSIC TRACE] state._cpp_solver already exists? {has_solver}")

    if not has_solver:
        logger.info("[FORENSIC TRACE] Initializing C++ solver from scratch...")
        
        if hasattr(state, "input_data") and isinstance(state.input_data, dict):
            if "external_forces" in state.input_data and not hasattr(state, "external_forces"):
                state.external_forces = state.input_data["external_forces"]
            if "fluid_properties" in state.input_data and not hasattr(state, "fluid_properties"):
                state.fluid_properties = state.input_data["fluid_properties"]
            if "simulation_parameters" in state.input_data and not hasattr(state, "simulation_parameters"):
                state.simulation_parameters = state.input_data["simulation_parameters"]
                if "time_step" in state.simulation_parameters and not hasattr(state, "dt"):
                    state.dt = state.simulation_parameters["time_step"]

        _convert_boundary_conditions(state)
        
        _log_field_max_abs("Pre-constructor state", state)
        
        logger.info("Initializing instance-bound C++ NavierStokesSolver engine for SolverState...")
        state._cpp_solver = navier_stokes_cpp.NavierStokesSolver(state)
        
        _log_field_max_abs("POST-constructor state (Did C++ zero out fields?)", state)
        
        _apply_initial_boundary_conditions(state)
        
        _log_field_max_abs("POST-re-application state", state)

    logger.info("[FORENSIC TRACE] === Exiting _get_or_create_cpp_solver ===")
    return state._cpp_solver


def step_simulation(state: SolverState) -> None:
    """
    Executes a single time-integration step through the C++ bridge interface with forensic trace for u, v, w, p.
    """
    logger.info(f"[FORENSIC TRACE] === Entering step_simulation (Iteration: {getattr(state, 'current_iteration', 0)}) ===")
    if state is None:
        raise ValueError("FATAL ERROR: state must be explicitly provided.")

    # Robust fallback extraction for time step 'dt' at start of step
    try:
        dt = float(state.dt)
    except (AttributeError, TypeError):
        try:
            dt = float(state.input_data["simulation_parameters"]["time_step"])
            state.dt = dt  # Cache back onto state for consistency
        except (AttributeError, KeyError, TypeError) as inner_err:
            raise KeyError(
                "FATAL ERROR: Simulation time step 'dt' or 'simulation_parameters.time_step' must be explicitly provided."
            ) from inner_err

    grid = state.input_data["grid"]
    dx = float(grid["dx"])
    dy = float(grid["dy"])
    dz = float(grid["dz"])

    if state.u is None or state.v is None or state.w is None:
        raise ValueError("FATAL ERROR: Velocity fields (u, v, w) must be initialized prior to execution.")

    max_u = float(abs(state.u).max())
    max_v = float(abs(state.v).max())
    max_w = float(abs(state.w).max())

    cfl = dt * (max_u / dx + max_v / dy + max_w / dz)
    if cfl > 1.0:
        raise ValueError(f"CFL violation intercepted: C = {cfl:.4f} > 1.0 (dt={dt}, max_u={max_u}, dx={dx})")

    solver = _get_or_create_cpp_solver(state)

    _log_field_max_abs("Pre-solver.step() state", state)

    try:
        logger.info("[FORENSIC TRACE] Executing solver.step(state)...")
        solver.step(state)
        
        _log_field_max_abs("Post-solver.step() state", state)

        if hasattr(solver, "sync_fields") and callable(solver.sync_fields):
            logger.info("[FORENSIC TRACE] Executing solver.sync_fields(state)...")
            solver.sync_fields(state)
            
            if hasattr(state, "fields") and state.fields is not None:
                state.u = state.fields[0]
                state.v = state.fields[1]
                state.w = state.fields[2]
                state.p = state.fields[3]

            _log_field_max_abs("Post-sync_fields state", state)
        else:
            raise RuntimeError(
                "FATAL ERROR: C++ NavierStokesSolver instance is missing required callable 'sync_fields' method."
            )
    except Exception as e:
        logger.error(f"C++ step execution failed at iteration {getattr(state, 'current_iteration', 0)}: {e}")
        raise RuntimeError(f"C++ execution failure during solver step: {e}") from e

    try:
        dt = float(state.dt)
    except (AttributeError, TypeError):
        try:
            dt = float(state.input_data["simulation_parameters"]["time_step"])
        except (AttributeError, KeyError, TypeError) as inner_err:
            raise KeyError(
                "FATAL ERROR: Simulation time step 'dt' or 'simulation_parameters.time_step' must be explicitly provided."
            ) from inner_err

    state.current_iteration += 1
    state.current_time += dt
    logger.info(f"[FORENSIC TRACE] === Exiting step_simulation successfully. New iteration: {state.current_iteration}, time: {state.current_time} ===")