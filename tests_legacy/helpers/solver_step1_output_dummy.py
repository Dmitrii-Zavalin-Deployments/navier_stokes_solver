# tests/helpers/solver_step1_output_dummy.py

import numpy as np

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


def make_step1_output_dummy(nx=4, ny=4, nz=4):
    state = SolverState()
    
    # 1. Geometry & Domain (Setters trigger validation)
    state.grid = GridManager()
    state.grid.x_min, state.grid.x_max = 0.0, 1.0
    state.grid.y_min, state.grid.y_max = 0.0, 1.0
    state.grid.z_min, state.grid.z_max = 0.0, 1.0
    state.grid.nx, state.grid.ny, state.grid.nz = nx, ny, nz
    
    state.domain = DomainManager(type="INTERNAL", reference_velocity=np.array([0.0, 0.0, 0.0]))
    
    # 2. Physics & Foundation
    state.fluid = FluidPropertiesManager(density=1000.0, viscosity=0.001)
    state.initial_conditions = InitialConditionManager(
        velocity=np.array([0.0, 0.0, 0.0]), pressure=0.0
    )
    state.sim_params = SimulationParameterManager(
        time_step=0.001, total_time=1.0, output_interval=1
    )
    state.external_forces = ExternalForceManager(
        force_vector=np.array([0.0, 0.0, -9.81])
    )
    
    # 3. Foundation Allocation (Rule 9)
    state.fields = FieldManager()
    state.fields.allocate(nx * ny * nz)
    
    # 4. Topology
    state.masks = MaskManager()
    state.masks.mask = np.ones((nx, ny, nz), dtype=int)
    
    # 5. Boundary Condition Setup
    bcs = [
        BoundaryCondition(location='x_min', type='inflow', values={'u': 1.0, 'v': 0.0, 'w': 0.0, 'p': 1.0}),
        BoundaryCondition(location='x_max', type='outflow', values={'p': 0.0}),
        BoundaryCondition(location='y_min', type='no-slip', values={'u': 0.0, 'v': 0.0, 'w': 0.0}),
        BoundaryCondition(location='y_max', type='no-slip', values={'u': 0.0, 'v': 0.0, 'w': 0.0}),
        BoundaryCondition(location='z_min', type='no-slip', values={'u': 0.0, 'v': 0.0, 'w': 0.0}),
        BoundaryCondition(location='z_max', type='no-slip', values={'u': 0.0, 'v': 0.0, 'w': 0.0}),
        BoundaryCondition(location='wall', type='no-slip', values={'u': 0.0, 'v': 0.0, 'w': 0.0})
    ]
    state.boundary_conditions = BoundaryConditionManager(conditions=bcs)
    
    # 6. Engine state
    state.iteration = 0
    state.time = 0.0
    state.ready_for_time_loop = False

    return state