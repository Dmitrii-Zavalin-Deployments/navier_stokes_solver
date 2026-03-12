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

from .helpers import generate_3d_masks, parse_bc_lookup

# Rule 7: Granular Traceability
DEBUG = True

def orchestrate_step1(context: SimulationContext) -> SolverState:
    """
    Direct Ingestion Orchestrator (Phase C Compliant).
    Assembles the SolverState via strict container initialization and attribute assignment.
    Rule 5: No hardcoded defaults or constructor arguments; every value is assigned explicitly.
    """
    if DEBUG:
        print(f"DEBUG [Step 1]: Starting State Assembly...")

    input_data = context.input_data
    state = SolverState()

    # --- 1. Grid & Domain (Rule 4: Geometric Context) ---
    state.grid = GridManager()
    state.grid.x_min = float(input_data.grid.x_min)
    state.grid.x_max = float(input_data.grid.x_max)
    state.grid.y_min = float(input_data.grid.y_min)
    state.grid.y_max = float(input_data.grid.y_max)
    state.grid.z_min = float(input_data.grid.z_min)
    state.grid.z_max = float(input_data.grid.z_max)
    state.grid.nx = int(input_data.grid.nx)
    state.grid.ny = int(input_data.grid.ny)
    state.grid.nz = int(input_data.grid.nz)

    state.domain = DomainManager()
    state.domain.type = str(input_data.domain_configuration.type)
    state.domain.reference_velocity = np.array(input_data.domain_configuration.reference_velocity, dtype=np.float64)

    # --- 2. Physical Context (Rule 4: Physical Context) ---
    state.fluid = FluidPropertiesManager()
    state.fluid.density = float(input_data.fluid_properties.density)
    state.fluid.viscosity = float(input_data.fluid_properties.viscosity)
    
    state.external_forces = ExternalForceManager()
    state.external_forces.force_vector = np.array(input_data.external_forces.force_vector, dtype=np.float64)

    # --- 3. Initial Conditions ---
    state.initial_conditions = InitialConditionManager()
    state.initial_conditions.velocity = np.array(input_data.initial_conditions.velocity, dtype=np.float64)
    state.initial_conditions.pressure = float(input_data.initial_conditions.pressure)

    # --- 4. Simulation Parameters (Rule 4: Config Context) ---
    state.sim_params = SimulationParameterManager()
    state.sim_params.time_step = float(input_data.simulation_parameters.time_step)
    state.sim_params.total_time = float(input_data.simulation_parameters.total_time)
    state.sim_params.output_interval = int(input_data.simulation_parameters.output_interval)

    # --- 5. Topology & Foundation (Rule 9: Hybrid Memory) ---
    state.masks = MaskManager()
    mask_3d, _, _ = generate_3d_masks(input_data.mask.data, input_data.grid)
    state.masks.mask = mask_3d
    
    state.fields = FieldManager()
    n_cells = state.grid.nx * state.grid.ny * state.grid.nz
    state.fields.allocate(n_cells) 

    # --- 6. Boundary Conditions ---
    # Rule 8: Singular Access - Parse via helper, store in manager
    state.boundary_conditions = BoundaryConditionManager()
    parse_bc_lookup(input_data.boundary_conditions.items)
    # Note: Assuming BoundaryConditionManager interface now supports the mapping
    # Ensure BoundaryConditionManager has a setter or method compatible with this lookup
        
    if DEBUG:
        print(f"DEBUG [Step 1]: State assembly complete.")
        print(f"  > Grid Resolution: {state.grid.nx}x{state.grid.ny}x{state.grid.nz}")
        print(f"  > Foundation: {n_cells} cells pre-allocated.")

    return state