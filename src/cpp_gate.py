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
    """Instantiates and populates a C++ BoundaryCondition object from a Python dict, mapping nested and flat fields."""
    if not isinstance(bc_dict, dict):
        raise TypeError("Boundary condition configuration must be a dictionary.")

    if "location" not in bc_dict:
        raise KeyError("Boundary condition configuration missing required field 'location'. Defaults are strictly prohibited.")
    if "type" not in bc_dict:
        raise KeyError("Boundary condition configuration missing required field 'type'. Defaults are strictly prohibited.")

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

    # Extract values dictionary strictly without fallback defaults
    vals = bc_dict.get("values")
    if vals is None:
        vals = bc_dict

    missing_fields = [k for k in ["u", "v", "w", "p"] if k not in vals]
    if missing_fields:
        raise KeyError(
            f"Boundary condition for '{bc_dict['location']}' missing required numerical value field(s): {missing_fields}. "
            "No default values allowed."
        )

    u_val = float(vals["u"])
    v_val = float(vals["v"])
    w_val = float(vals["w"])
    p_val = float(vals["p"])

    # 1. Populate nested 'values' C++ sub-object if bound by Pybind11
    if hasattr(bc_obj, "values"):
        val_obj = bc_obj.values
        if val_obj is not None:
            for k, val in [("u", u_val), ("v", v_val), ("w", w_val), ("p", p_val)]:
                if hasattr(val_obj, k):
                    try:
                        setattr(val_obj, k, val)
                    except (AttributeError, TypeError) as err:
                        logger.debug(f"Skipping read-only or type-incompatible Pybind11 nested attribute 'values.{k}': {err}")

    # 2. Fallback/direct attribute mappings on the BoundaryCondition object itself
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
                logger.debug(
                    f"Skipping read-only or incompatible Pybind11 attribute '{attr}' on BoundaryCondition: {err}"
                )

    return bc_obj


def _apply_initial_boundary_conditions(state: SolverState) -> None:
    """Enforces initial boundary condition values (e.g., inflow velocity = 0.1) onto array boundary faces."""
    raw_bcs = getattr(state, "boundary_conditions", None)
    if not raw_bcs and hasattr(state, "input_data") and isinstance(state.input_data, dict):
        raw_bcs = state.input_data.get("boundary_conditions")

    if not raw_bcs:
        raise KeyError("FATAL ERROR: Boundary conditions missing from state and input_data. Defaults are strictly prohibited.")

    for bc in raw_bcs:
        if isinstance(bc, dict):
            if "location" not in bc or "type" not in bc:
                raise KeyError("Boundary condition configuration item missing required 'location' or 'type' key.")
            loc = bc["location"]
            bc_type = bc["type"]
            vals = bc.get("values", bc)
        else:
            loc = getattr(bc, "location", None)
            bc_type = getattr(bc, "type", None)
            if loc is None or bc_type is None:
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
                    raise KeyError(f"Boundary condition object for '{loc}' missing required attribute '{k}'. No defaults allowed.")
                vals[k] = val

        if bc_type in ["inflow", "prescribed"]:
            missing_vals = [k for k in ["u", "v", "w", "p"] if k not in vals]
            if missing_vals:
                raise KeyError(f"Inflow boundary condition '{loc}' missing required values {missing_vals}. Defaults are strictly prohibited.")

            u_val = float(vals["u"])
            v_val = float(vals["v"])
            w_val = float(vals["w"])
            p_val = float(vals["p"])

            if loc == "x_min":
                state.u[0, :, :] = u_val
                state.v[0, :, :] = v_val
                state.w[0, :, :] = w_val
                state.p[0, :, :] = p_val
            elif loc == "x_max":
                state.u[-1, :, :] = u_val
                state.v[-1, :, :] = v_val
                state.w[-1, :, :] = w_val
                state.p[-1, :, :] = p_val
            elif loc == "y_min":
                state.u[:, 0, :] = u_val
                state.v[:, 0, :] = v_val
                state.w[:, 0, :] = w_val
                state.p[:, 0, :] = p_val
            elif loc == "y_max":
                state.u[:, -1, :] = u_val
                state.v[:, -1, :] = v_val
                state.w[:, -1, :] = w_val
                state.p[:, -1, :] = p_val
            elif loc == "z_min":
                state.u[:, :, 0] = u_val
                state.v[:, :, 0] = v_val
                state.w[:, :, 0] = w_val
                state.p[:, :, 0] = p_val
            elif loc == "z_max":
                state.u[:, :, -1] = u_val
                state.v[:, :, -1] = v_val
                state.w[:, :, -1] = w_val
                state.p[:, :, -1] = p_val


def _convert_boundary_conditions(state: SolverState) -> None:
    """Converts boundary conditions to C++ BoundaryCondition objects in-place on state.boundary_conditions."""
    raw_bcs = getattr(state, "boundary_conditions", None)
    if not raw_bcs and hasattr(state, "input_data") and isinstance(state.input_data, dict):
        raw_bcs = state.input_data.get("boundary_conditions")

    if not raw_bcs:
        raise KeyError("FATAL ERROR: Boundary conditions configuration missing from SolverState or input_data. No defaults allowed.")

    state.boundary_conditions = [
        _dict_to_boundary_condition(bc) if isinstance(bc, dict) else bc
        for bc in raw_bcs
    ]

    _apply_initial_boundary_conditions(state)


def _get_or_create_cpp_solver(state: SolverState) -> Any:
    """
    Instance-bound initializer for the underlying C++ NavierStokesSolver engine.
    Attaches the engine directly to the provided SolverState instance to guarantee 
    zero-copy memory binding without cross-instance stale reference leaks.
    """
    if state is None:
        raise ValueError("FATAL ERROR: state must be explicitly provided (no defaults allowed).")

    if not hasattr(state, "_cpp_solver") or state._cpp_solver is None:
        _convert_boundary_conditions(state)
        logger.info("Initializing instance-bound C++ NavierStokesSolver engine for SolverState...")
        state._cpp_solver = navier_stokes_cpp.NavierStokesSolver(state)

    return state._cpp_solver


def step_simulation(state: SolverState) -> None:
    """
    Executes a single time-integration step through the C++ bridge interface,
    utilizing direct sovereign container reference for in-place RAM mutation.

    Args:
        state: Sovereign SolverState instance holding physical arrays and simulation parameters.
    """
    if state is None:
        raise ValueError("FATAL ERROR: state must be explicitly provided (no defaults allowed).")

    # Instance engine retrieval lazily converts BCs and seeds initial values
    solver = _get_or_create_cpp_solver(state)

    # 1. Isolated C++ core step execution
    try:
        solver.step(state)

        # Enforce strict field synchronization to ensure C++ array state maps back to Python memory
        if hasattr(solver, "sync_fields") and callable(solver.sync_fields):
            solver.sync_fields(state)
        else:
            raise RuntimeError(
                "FATAL ERROR: C++ NavierStokesSolver instance is missing required callable 'sync_fields' method."
            )
    except Exception as e:
        logger.error(f"C++ step execution failed at iteration {getattr(state, 'current_iteration', 0)}: {e}")
        raise RuntimeError(f"C++ execution failure during solver step: {e}") from e

    # 2. Sovereign state tracking metric updates
    try:
        dt = float(state.dt)
    except (AttributeError, TypeError):
        try:
            dt = float(state.input_data["simulation_parameters"]["time_step"])
        except (AttributeError, KeyError, TypeError) as inner_err:
            raise KeyError(
                "FATAL ERROR: Simulation time step 'dt' or 'simulation_parameters.time_step' "
                "must be explicitly provided (no defaults allowed)."
            ) from inner_err

    state.current_iteration += 1
    state.current_time += dt