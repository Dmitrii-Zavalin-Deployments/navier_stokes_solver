# tests/helpers/solver_step2_output_dummy.py

import numpy as np
from scipy.sparse import csr_matrix, eye
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

def make_step2_output_dummy(nx=4, ny=4, nz=4):
    """
    Zero-Debt Dummy Generator for Step 2 Output.
    Satisfies Rule 5 (Explicit Config) and avoids Matrix Voids.
    """
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    dof_p = nx * ny * nz
    dof_u = (nx + 1) * ny * nz
    dof_v = nx * (ny + 1) * nz
    dof_w = nx * ny * (nz + 1)
    total_vel_dof = dof_u + dof_v + dof_w

    # 1. Operators: Use eye() for Laplacian to allow matrix inversion in tests
    # Using CSR format as required by the scipy.sparse.linalg.cg solver
    state.operators.divergence = csr_matrix((dof_p, total_vel_dof))
    state.operators.grad_x = csr_matrix((dof_u, dof_p))
    state.operators.grad_y = csr_matrix((dof_v, dof_p))
    state.operators.grad_z = csr_matrix((dof_w, dof_p))
    state.operators.laplacian = eye(dof_p, format='csr')
    
    # 2. Advection data storage alignment
    state.advection._weights = np.zeros((total_vel_dof, 8))
    state.advection._indices = np.zeros((total_vel_dof, 8), dtype=int)
    
    # 3. Mandatory Handshake & Configuration (Rule 5 Compliance)
    state.ppe._A = state.operators.laplacian
    
    # Hydrating the configuration to prevent _get_safe Access Errors
    state.config.ppe_tolerance = 1e-10
    state.config.ppe_atol = 1e-12
    state.config.ppe_max_iter = 1000
    
    # 4. State Baseline
    state.health.is_stable = True
    state.health.max_u = 0.0
    state.health.divergence_norm = 0.0
    state.health.post_correction_divergence_norm = 0.0
    
    state.ready_for_time_loop = True
    
    return state