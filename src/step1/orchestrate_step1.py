# src/step1/orchestrate_step1.py

from __future__ import annotations

import numpy as np

from src.solver_input import SolverInput
from src.solver_state import SolverState

from .helpers import allocate_staggered_fields, generate_3d_masks, parse_bc_lookup


def orchestrate_step1(input_data: SolverInput, **kwargs) -> SolverState:
    """
    Direct Ingestion Orchestrator.
    Maps SolverInput (Contract) to SolverState (Truth) with Zero-Debt logic.
    """
    state = SolverState()

    # --- 1. Geometric Context (Grid) ---
    grid_in = input_data.grid
    state.grid.nx, state.grid.ny, state.grid.nz = grid_in.nx, grid_in.ny, grid_in.nz
    state.grid.x_min, state.grid.x_max = grid_in.x_min, grid_in.x_max
    state.grid.y_min, state.grid.y_max = grid_in.y_min, grid_in.y_max
    state.grid.z_min, state.grid.z_max = grid_in.z_min, grid_in.z_max

    # --- 2. Physical Context (Fluid & Forces) ---
    state.fluid.rho = input_data.fluid_properties.density
    state.fluid.mu = input_data.fluid_properties.viscosity
    
    # Direct mapping for config elements
    state.config.external_forces = {
        "force_vector": input_data.external_forces.force_vector
    }
    state.config.boundary_conditions = [
        {"location": bc.location, "type": bc.type, "values": bc.values}
        for bc in input_data.boundary_conditions.items
    ]

    # --- 3. Memory Allocation (Fields) ---
    raw_fields = allocate_staggered_fields(grid_in)
    state.fields.P = raw_fields["P"]
    state.fields.U = raw_fields["U"]
    state.fields.V = raw_fields["V"]
    state.fields.W = raw_fields["W"]

    # Initial Condition Priming
    if input_data.initial_conditions.pressure is not None:
        state.fields.P.fill(input_data.initial_conditions.pressure)
    
    if input_data.initial_conditions.velocity is not None:
        u0, v0, w0 = input_data.initial_conditions.velocity
        state.fields.U.fill(u0)
        state.fields.V.fill(v0)
        state.fields.W.fill(w0)

    # --- 4. Topology (Masks) ---
    # CONSTITUTIONAL ALIGNMENT: We must persist the fluid/boundary distinction
    # for the Step 2 Operators to build valid sparse matrices.
    mask_3d, is_fluid, is_boundary = generate_3d_masks(input_data.mask.data, grid_in)
    
    state.masks.mask = mask_3d
    state.masks.is_fluid = is_fluid 
    state.masks.is_boundary = is_boundary

    # --- 5. Global Metadata (Deterministic Policy) ---
    if "iteration" not in kwargs or "time" not in kwargs:
        raise KeyError("Step 1 requires explicit iteration and time values.")

    state.iteration = int(kwargs["iteration"])
    state.time = float(kwargs["time"])

    state.ready_for_time_loop = False
    
    # Internal Lookup Tables (for math performance)
    state.boundary_lookup = parse_bc_lookup(input_data.boundary_conditions.items)
    
    # --- Config Hydration for Security Guard ---
    state.config._simulation_parameters = input_data.simulation_parameters
    state.config._fluid_properties = input_data.fluid_properties

    return state