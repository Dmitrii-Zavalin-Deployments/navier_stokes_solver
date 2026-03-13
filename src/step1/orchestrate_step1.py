# src/step1/orchestrate_step1.py

from __future__ import annotations

import numpy as np

from src.common.field_schema import FI
from src.common.simulation_context import SimulationContext
from src.common.solver_state import (
    BoundaryCondition,
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

from .helpers import generate_3d_masks

# Rule 7: Granular Traceability
DEBUG = True

def orchestrate_step1(context: SimulationContext) -> SolverState:
    """
    Direct Ingestion Orchestrator (Phase C Compliant).
    Assembles the SolverState via strict container initialization and attribute assignment.
    """
    if DEBUG:
        print(f"DEBUG [Step 1]: Starting State Assembly...")

    input_data = context.input_data
    state = SolverState()

    # --- 1. Grid & Domain ---
    state.grid = GridManager()
    state.grid.x_min, state.grid.x_max = float(input_data.grid.x_min), float(input_data.grid.x_max)
    state.grid.y_min, state.grid.y_max = float(input_data.grid.y_min), float(input_data.grid.y_max)
    state.grid.z_min, state.grid.z_max = float(input_data.grid.z_min), float(input_data.grid.z_max)
    state.grid.nx, state.grid.ny, state.grid.nz = int(input_data.grid.nx), int(input_data.grid.ny), int(input_data.grid.nz)

    state.domain = DomainManager()
    state.domain.type = str(input_data.domain_configuration.type)
    state.domain.reference_velocity = np.array(input_data.domain_configuration.reference_velocity, dtype=np.float64)

    # --- 2. Physical Context ---
    state.fluid = FluidPropertiesManager()
    state.fluid.density, state.fluid.viscosity = float(input_data.fluid_properties.density), float(input_data.fluid_properties.viscosity)
    
    state.external_forces = ExternalForceManager()
    state.external_forces.force_vector = np.array(input_data.external_forces.force_vector, dtype=np.float64)

    # --- 3. Initial Conditions ---
    state.initial_conditions = InitialConditionManager()
    state.initial_conditions.velocity = np.array(input_data.initial_conditions.velocity, dtype=np.float64)
    state.initial_conditions.pressure = float(input_data.initial_conditions.pressure)

    # --- 4. Simulation Parameters ---
    state.sim_params = SimulationParameterManager()
    state.sim_params.time_step = float(input_data.simulation_parameters.time_step)
    state.sim_params.total_time = float(input_data.simulation_parameters.total_time)
    state.sim_params.output_interval = int(input_data.simulation_parameters.output_interval)

    # --- 5. Topology & Foundation ---
    mask_3d, _, _ = generate_3d_masks(input_data.mask.data, input_data.grid)
    state.fields = FieldManager()
    n_cells = state.grid.nx * state.grid.ny * state.grid.nz
    state.fields.allocate(n_cells) 
    state.fields.data[:, FI.MASK] = mask_3d.flatten()
    
    state.masks = MaskManager()
    state.masks.mask = mask_3d

    # --- 6. Boundary Conditions ---
    state.boundary_conditions = BoundaryConditionManager()
    
    # Strictly adhere to dictionary input structure as hydrated by SolverInput
    for item in input_data.boundary_conditions:
        bc = BoundaryCondition()
        bc.location = str(item["location"])
        bc.type = str(item["type"])
        bc.values = item["values"]
        
        state.boundary_conditions.add_condition(bc)
        
    if DEBUG:
        print(f"DEBUG [Step 1]: State assembly complete.")
        print(f"  > Foundation: {n_cells} cells allocated with {FI.num_fields()} fields.")

    return state