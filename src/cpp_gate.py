"""
src/cpp_gate.py
C++ Interaction Wrapper Module.
Bridges Python SolverState with the compiled C++ Navier-Stokes engine via Pybind11,
passing the sovereign container directly to establish zero-copy memory binding 
and eliminate parameter drift.
"""

import logging
from typing import Any

from src.state import SolverState

try:
    import navier_stokes_cpp
except ImportError as e:
    raise ImportError(
        "Failed to import compiled C++ module 'navier_stokes_cpp'. "
        "Ensure the C++ library is built and Python path includes the build directory."
    ) from e

logger = logging.getLogger("Solver.CppGate")


def _dict_to_boundary_condition(bc_dict: dict) -> Any:
    """Instantiates and populates a C++ BoundaryCondition object from a Python dict."""
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

    return bc_obj


def _apply_initial_boundary_conditions(state: SolverState) -> None:
    """Enforces initial boundary condition values (e.g., inflow velocity = 0.1) onto array boundary faces."""
    raw_bcs = None
    if hasattr(state, "input_data") and isinstance(state.input_data, dict):
        raw_bcs = state.input_data.get("boundary_conditions")
    if not raw_bcs:
        raw_bcs = getattr(state, "boundary_conditions", None)

    if not raw_bcs:
        raise KeyError("FATAL ERROR: Boundary conditions missing from state and input_data.")

    for bc in raw_bcs:
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

        # Case-insensitive and substring match for inflow/prescribed types across Pybind11 objects and Python dicts
        if "inflow" in bc_type or "prescribed" in bc_type or bc_type in ["inflow", "prescribed"]:
            missing_vals = [k for k in ["u", "v", "w", "p"] if k not in vals]
            if missing_vals:
                raise KeyError(f"Inflow boundary condition '{loc}' missing required values {missing_vals}.")

            u_val = float(vals["u"])
            v_val = float(vals["v"])
            w_val = float(vals["w"])
            p_val = float(vals["p"])

            if "x_min" in loc or "xmin" in loc:
                state.u[0, :, :] = u_val
                state.v[0, :, :] = v_val
                state.w[0, :, :] = w_val
                state.p[0, :, :] = p_val
            elif "x_max" in loc or "xmax" in loc:
                state.u[-1, :, :] = u_val
                state.v[-1, :, :] = v_val
                state.w[-1, :, :] = w_val
                state.p[-1, :, :] = p_val
            elif "y_min" in loc or "ymin" in loc:
                state.u[:, 0, :] = u_val
                state.v[:, 0, :] = v_val
                state.w[:, 0, :] = w_val
                state.p[:, 0, :] = p_val
            elif "y_max" in loc or "ymax" in loc:
                state.u[:, -1, :] = u_val
                state.v[:, -1, :] = v_val
                state.w[:, -1, :] = w_val
                state.p[:, -1, :] = p_val
            elif "z_min" in loc or "zmin" in loc:
                state.u[:, :, 0] = u_val
                state.v[:, :, 0] = v_val
                state.w[:, :, 0] = w_val
                state.p[:, :, 0] = p_val
            elif "z_max" in loc or "zmax" in loc:
                state.u[:, :, -1] = u_val
                state.v[:, :, -1] = v_val
                state.w[:, :, -1] = w_val
                state.p[:, :, -1] = p_val


def _convert_boundary_conditions(state: SolverState) -> None:
    """Converts boundary conditions to C++ BoundaryCondition objects in-place on state.boundary_conditions."""
    _apply_initial_boundary_conditions(state)

    raw_bcs = getattr(state, "boundary_conditions", None)
    if not raw_bcs and hasattr(state, "input_data") and isinstance(state.input_data, dict):
        raw_bcs = state.input_data.get("boundary_conditions")

    if not raw_bcs:
        raise KeyError("FATAL ERROR: Boundary conditions configuration missing from SolverState or input_data.")

    state.boundary_conditions = [
        _dict_to_boundary_condition(bc) if isinstance(bc, dict) else bc
        for bc in raw_bcs
    ]


def _get_or_create_cpp_solver(state: SolverState) -> Any:
    """
    Instance-bound initializer for the underlying C++ NavierStokesSolver engine.
    Attaches the engine directly to the provided SolverState instance.
    """
    if state is None:
        raise ValueError("FATAL ERROR: state must be explicitly provided.")

    if not hasattr(state, "_cpp_solver") or state._cpp_solver is None:
        _convert_boundary_conditions(state)
        logger.info("Initializing instance-bound C++ NavierStokesSolver engine for SolverState...")
        state._cpp_solver = navier_stokes_cpp.NavierStokesSolver(state)
        # Re-apply initial boundary values AFTER C++ constructor initialization
        _apply_initial_boundary_conditions(state)

    return state._cpp_solver


def step_simulation(state: SolverState) -> None:
    """
    Executes a single time-integration step through the C++ bridge interface.
    """
    if state is None:
        raise ValueError("FATAL ERROR: state must be explicitly provided.")

    solver = _get_or_create_cpp_solver(state)

    try:
        solver.step(state)

        if hasattr(solver, "sync_fields") and callable(solver.sync_fields):
            solver.sync_fields(state)
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
