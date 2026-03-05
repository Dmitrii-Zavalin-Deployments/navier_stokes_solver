# tests/helpers/solver_step1_output_dummy.py

import numpy as np
from src.solver_state import SolverState

def make_step1_output_dummy(nx=4, ny=4, nz=4):
    state = SolverState()
    
    # --- MANDATORY HYDRATION STEP ---
    # Initialize the internal dictionaries so the ValidatedContainer allows access
    state.config._simulation_parameters = {'time_step': 0.001, 'total_time': 1.0, 'output_interval': 1, 'g': 9.81, 'advection_weight_base': 0.125}
    state.config._fluid_properties = {}
    state.config._external_forces = {}
    state.config._initial_conditions = {}
    # --------------------------------

    # Now dot-notation will work without triggering the RuntimeError
    state.fluid._rho = 1000.0
    state.fluid._mu = 0.001
    
    state.config.case_name = "dummy_verification"
    state.config.simulation_parameters["time_step"] = 0.001
    state.config.simulation_parameters["total_time"] = 1.0
    state.config.simulation_parameters["output_interval"] = 1
    
    state.config.fluid_properties["density"] = 1000.0
    state.config.fluid_properties["viscosity"] = 0.001
    
    state.config.external_forces["force_vector"] = [0.0, 0.0, -9.81]
    state.config.initial_conditions["velocity"] = [0.0, 0.0, 0.0]
    state.config.initial_conditions["pressure"] = 0.0
    
    state.config._boundary_conditions = [
        {'location': 'x_min', 'type': 'no-slip', 'values': {'u': 0.0, 'v': 0.0, 'w': 0.0}},
        {'location': 'x_max', 'type': 'outflow', 'values': {'p': 0.0}},
        {'location': 'y_min', 'type': 'no-slip', 'values': {'u': 0.0, 'v': 0.0, 'w': 0.0}},
        {'location': 'y_max', 'type': 'no-slip', 'values': {'u': 0.0, 'v': 0.0, 'w': 0.0}},
        {'location': 'z_min', 'type': 'no-slip', 'values': {'u': 0.0, 'v': 0.0, 'w': 0.0}},
        {'location': 'z_max', 'type': 'no-slip', 'values': {'u': 0.0, 'v': 0.0, 'w': 0.0}}]

    # Standard Grid/Field Setup
    state.grid.nx, state.grid.ny, state.grid.nz = nx, ny, nz
    state.grid.x_min, state.grid.x_max = 0.0, 1.0
    state.grid.y_min, state.grid.y_max = 0.0, 1.0
    state.grid.z_min, state.grid.z_max = 0.0, 1.0
    
    state.fields.P = np.zeros((nx, ny, nz))
    state.fields.U = np.zeros((nx + 1, ny, nz))
    state.fields.V = np.zeros((nx, ny + 1, nz))
    state.fields.W = np.zeros((nx, ny, nz + 1))
    state.masks.mask = np.ones((nx, ny, nz), dtype=int)

    state.iteration = 0
    state.time = 0.0
    state.ready_for_time_loop = False
    state.health.is_stable = True 
    state.ppe.status = "initialized"

    return state