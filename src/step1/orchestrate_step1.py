# src/step1/orchestrate_step1.py

from __future__ import annotations
import numpy as np

from src.common.field_schema import FI
from src.common.solver_state import (
    SolverState,
    GridManager,
    DomainManager,
    FluidPropertiesManager,
    ExternalForceManager,
    InitialConditionManager,
    SimulationParameterManager,
    MaskManager,
    FieldManager,
    BoundaryConditionManager,
    BoundaryCondition
)
from src.solver_input import SolverInput
from .helpers import generate_3d_masks, parse_bc_lookup

# Rule 7: Granular Traceability
DEBUG = True

def orchestrate_step1(input_data: SolverInput) -> SolverState:
    """
    Direct Ingestion Orchestrator (Phase C Compliant).
    Assembles the SolverState via strict container initialization.
    """
    if DEBUG:
        print(f"DEBUG [Step 1]: Starting State Assembly...")

    state = SolverState()

    # --- 1. Grid & Domain ---
    # SSoT: All grid properties explicitly mapped to the GridManager
    state.grid = GridManager(
        x_min=float(input_data.grid.x_min), x_max=float(input_data.grid.x_max),
        y_min=float(input_data.grid.y_min), y_max=float(input_data.grid.y_max),
        z_min=float(input_data.grid.z_min), z_max=float(input_data.grid.z_max),
        nx=int(input_data.grid.nx), ny=int(input_data.grid.ny), nz=int(input_data.grid.nz)
    )
    
    state.domain = DomainManager(
        type=str(input_data.domain.type),
        reference_velocity=np.array(input_data.domain.reference_velocity, dtype=np.float64)
    )

    # --- 2. Physical Context (Fluid & Forces) ---
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

    # --- 4. Simulation Parameters ---
    state.sim_params = SimulationParameterManager(
        time_step=float(input_data.simulation_parameters.time_step),
        total_time=float(input_data.simulation_parameters.total_time),
        output_interval=int(input_data.simulation_parameters.output_interval)
    )

    # --- 5. Topology & Foundation ---
    state.masks = MaskManager()
    mask_3d, _, _ = generate_3d_masks(input_data.mask.data, input_data.grid)
    state.masks.mask = mask_3d
    
    # Foundation Initialization (Rule 9)
    state.fields = FieldManager()
    n_cells = state.grid.nx * state.grid.ny * state.grid.nz
    state.fields.allocate(n_cells)

    # --- 6. Boundary Conditions ---
    # Use helper to parse BC lookup (Rule 8: Singular Access)
    bc_lookup = parse_bc_lookup(input_data.boundary_conditions)
    state.boundary_conditions = BoundaryConditionManager(lookup_table=bc_lookup)
        
    if DEBUG:
        print(f"DEBUG [Step 1]: State assembly complete.")
        print(f"  > Domain: {state.grid.nx}x{state.grid.ny}x{state.grid.nz}")
        print(f"  > Foundation: {n_cells} cells allocated.")

    return state