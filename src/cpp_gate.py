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
    bc_obj = navier_stokes_cpp.BoundaryCondition()

    if "location" in bc_dict and hasattr(bc_obj, "location"):
        setattr(bc_obj, "location", str(bc_dict["location"]))
    if "type" in bc_dict and hasattr(bc_obj, "type"):
        setattr(bc_obj, "type", str(bc_dict["type"]))

    vals = bc_dict.get("values", {})
    if not isinstance(vals, dict):
        vals = {}

    u_val = float(vals.get("u", bc_dict.get("u", 0.0)))
    v_val = float(vals.get("v", bc_dict.get("v", 0.0)))
    w_val = float(vals.get("w", bc_dict.get("w", 0.0)))
    p_val = float(vals.get("p", bc_dict.get("p", 0.0)))

    # Set attributes across all potential C++ binding field naming conventions
    for attr in ["u_val", "u"]:
        if hasattr(bc_obj, attr):
            setattr(bc_obj, attr, u_val)
    for attr in ["v_val", "v"]:
        if hasattr(bc_obj, attr):
            setattr(bc_obj, attr, v_val)
    for attr in ["w_val", "w"]:
        if hasattr(bc_obj, attr):
            setattr(bc_obj, attr, w_val)
    for attr in ["scalar_p", "p", "p_val"]:
        if hasattr(bc_obj, attr):
            setattr(bc_obj, attr, p_val)

    return bc_obj


def _apply_initial_boundary_conditions(state: SolverState) -> None:
    """Enforces initial boundary condition values (e.g., inflow velocity = 0.1) onto array boundaries."""
    raw_bcs = getattr(state, "input_data", {}).get("boundary_conditions", [])
    if not raw_bcs:
        raw_bcs = getattr(state, "boundary_conditions", [])

    for bc in raw_bcs:
        if isinstance(bc, dict):
            loc = bc.get("location", "")
            bc_type = bc.get("type", "")
            vals = bc.get("values", {})
        else:
            loc = getattr(bc, "location", "")
            bc_type = getattr(bc, "type", "")
            vals = {
                "u": getattr(bc, "u_val", getattr(bc, "u", 0.0)),
                "v": getattr(bc, "v_val", getattr(bc, "v", 0.0)),
                "w": getattr(bc, "w_val", getattr(bc, "w", 0.0)),
                "p": getattr(bc, "scalar_p", getattr(bc, "p", 0.0)),
            }

        if bc_type in ["inflow", "prescribed"]:
            u_val = float(vals.get("u", 0.0))
            v_val = float(vals.get("v", 0.0))
            w_val = float(vals.get("w", 0.0))
            p_val = float(vals.get("p", 0.0))

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
        raw_bcs = state.input_data.get("boundary_conditions", [])

    if raw_bcs:
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

    # Instance engine retrieval lazily converts BCs and applies initial boundary values
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