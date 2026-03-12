# src/step1/orchestrate_step1.py

from __future__ import annotations

import numpy as np

from src.common.simulation_context import SimulationContext
from src.common.solver_state import (
    BoundaryConditionManager,
    DomainManager,
    ExternalForceManager,
    FieldManager,
    FluidPropertiesManager,
    GridManager,
    InitialConditionManager,
    MaskManager,
    SimulationParameterManager,
    SolverState,
)
from src.common.field_schema import FI
from .helpers import generate_3d_masks, parse_bc_lookup

# Rule 7: Granular Traceability
DEBUG = True

def orchestrate_step1(context: SimulationContext) -> SolverState:
    """
    Direct Ingestion Orchestrator (Phase C Compliant).
    Assembles the SolverState via strict container initialization. 
    Rule 5: No hardcoded defaults; every value is derived from context.
    """
    if DEBUG:
        print(f"DEBUG [Step 1]: Starting State Assembly...")

    input_data = context.input_data
    state = SolverState()

    # --- 1. Grid & Domain (Rule 4: Geometric Context) ---
    state.grid = GridManager(
        x_min=float(input_data.grid.x_min), x_max=float(input_data.grid.x_max),
        y_min=float(input_data.grid.y_min), y_max=float(input_data.grid.y_max),
        z_min=float(input_data.grid.z_min), z_max=float(input_data.grid.z_max),
        nx=int(input_data.grid.nx), ny=int(input_data.grid.ny), nz=int(input_data.grid.nz)
    )
    
    state.domain = DomainManager(
        type=str(input_data.domain_configuration.type),
        reference_velocity=np.array(input_data.domain_configuration.reference_velocity, dtype=np.float64)
    )

    # --- 2. Physical Context (Rule 4: Physical Context) ---
    state.fluid = FluidPropertiesManager(
        density=float(input_data.fluid_properties.density),
        viscosity=float(input_data.fluid_properties.viscosity)
    )
    
    state.external_forces = ExternalForceManager(
        force_vector=np.array(input_data.external_forces.force_vector, dtype=np.float64)
    )

    # --- 3. Initial Conditions ---
    state.initial_conditions = InitialConditionManager(
        velocity=np.array(input_data.initial_conditions.velocity, dtype=np.float64),
        pressure=float(input_data.initial_conditions.pressure)
    )

    # --- 4. Simulation Parameters (Rule 4: Config Context) ---
    state.sim_params = SimulationParameterManager(
        time_step=float(input_data.simulation_parameters.time_step),
        total_time=float(input_data.simulation_parameters.total_time),
        output_interval=int(input_data.simulation_parameters.output_interval)
    )

    # --- 5. Topology & Foundation (Rule 9: Hybrid Memory) ---
    state.masks = MaskManager()
    mask_3d, _, _ = generate_3d_masks(input_data.mask.data, input_data.grid)
    state.masks.mask = mask_3d
    
    # Pre-allocation of FieldManager (Foundation Layer)
    state.fields = FieldManager()
    n_cells = state.grid.nx * state.grid.ny * state.grid.nz
    state.fields.allocate(n_cells) 

    # --- 6. Boundary Conditions ---
    # Rule 8: Singular Access - Parse via helper, store in manager
    bc_lookup = parse_bc_lookup(input_data.boundary_conditions.items)
    state.boundary_conditions = BoundaryConditionManager(lookup_table=bc_lookup)
        
    if DEBUG:
        print(f"DEBUG [Step 1]: State assembly complete.")
        print(f"  > Grid Resolution: {state.grid.nx}x{state.grid.ny}x{state.grid.nz}")
        print(f"  > Foundation: {n_cells} cells pre-allocated.")

    return state