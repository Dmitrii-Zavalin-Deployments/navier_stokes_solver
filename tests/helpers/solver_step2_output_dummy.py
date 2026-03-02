# tests/helpers/solver_step2_output_dummy.py

import numpy as np
from scipy.sparse import csr_matrix
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

def make_step2_output_dummy(nx=4, ny=4, nz=4):
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    dof_p = nx * ny * nz
    dof_u = (nx + 1) * ny * nz
    dof_v = nx * (ny + 1) * nz
    dof_w = nx * ny * (nz + 1)
    total_vel_dof = dof_u + dof_v + dof_w

    # Attributes strictly matching the frozen OperatorStorage
    state.operators.divergence = csr_matrix((dof_p, total_vel_dof))
    state.operators.grad_x = csr_matrix((dof_u, dof_p))
    state.operators.grad_y = csr_matrix((dof_v, dof_p))
    state.operators.grad_z = csr_matrix((dof_w, dof_p))
    state.operators.laplacian = csr_matrix((dof_p, dof_p))
    
    # Advection data storage alignment
    state.advection._weights = np.zeros((total_vel_dof, 8))
    state.advection._indices = np.zeros((total_vel_dof, 8), dtype=int)
    
    state.ppe._A = state.operators.laplacian
    state.health.is_stable = True
    
    return state