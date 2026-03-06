# src/step1/orchestrate_step1.py

from __future__ import annotations

from src.solver_input import SolverInput
from src.solver_state import SolverState

from .helpers import allocate_fields, generate_3d_masks, parse_bc_lookup


def orchestrate_step1(input_data: SolverInput, **kwargs) -> SolverState:
    """
    Direct Ingestion Orchestrator.
    Maps SolverInput (Contract) to SolverState (Truth) with Zero-Debt logic.
    
    Theory Alignment:
    - Uses Collocated Grid (Section 3).
    - Uses Mask-Based Geometry (Section 6).
    """
    state = SolverState()

    # --- 1. Geometric Context (Grid) ---
    # Rule 5: Accessing properties directly. 
    # If GridInput is missing these, it will raise AttributeError (Strict Failure).
    g = input_data.grid
    state.grid.nx, state.grid.ny, state.grid.nz = g.nx, g.ny, g.nz
    state.grid.x_min, state.grid.x_max = g.x_min, g.x_max
    state.grid.y_min, state.grid.y_max = g.y_min, g.y_max
    state.grid.z_min, state.grid.z_max = g.z_min, g.z_max

    # --- 2. Physical Context (Fluid & Forces) ---
    # Rule 5: No defaults. Fluid properties must exist in input.
    state.fluid.rho = input_data.fluid_properties.density
    state.fluid.mu = input_data.fluid_properties.viscosity
    
    state.config.external_forces = {"force_vector": input_data.external_forces.force_vector}
    
    # --- 3. Memory Allocation (Collocated Fields) ---
    # Aligned with Section 3: All fields share (nx, ny, nz)
    raw_fields = allocate_fields(g)
    state.fields.P = raw_fields["P"]
    state.fields.U = raw_fields["U"]
    state.fields.V = raw_fields["V"]
    state.fields.W = raw_fields["W"]

    # Explicit Initialization (Strict adherence to Rule 5)
    # Removing 'is not None' checks; if they are not in the contract, the input is invalid.
    state.fields.P.fill(input_data.initial_conditions.pressure)
    u0, v0, w0 = input_data.initial_conditions.velocity
    state.fields.U.fill(u0)
    state.fields.V.fill(v0)
    state.fields.W.fill(w0)

    # --- 4. Topology (Masks) ---
    # Aligned with Section 6: Mask-Based Geometry
    mask_3d, is_fluid, is_boundary = generate_3d_masks(input_data.mask.data, g)
    state.masks.mask = mask_3d
    state.masks.is_fluid = is_fluid 
    state.masks.is_boundary = is_boundary

    # --- 5. Global Metadata (Deterministic Policy) ---
    # Explicitly derived from kwargs to prevent silent defaults
    state.iteration = int(kwargs["iteration"])
    state.time = float(kwargs["time"])
    state.ready_for_time_loop = False
    
    # Internal Lookup Tables
    state.boundary_lookup = parse_bc_lookup(input_data.boundary_conditions.items)
    
    # --- 6. Config Hydration ---
    state.config._simulation_parameters = input_data.simulation_parameters
    state.config._fluid_properties = input_data.fluid_properties

    return state