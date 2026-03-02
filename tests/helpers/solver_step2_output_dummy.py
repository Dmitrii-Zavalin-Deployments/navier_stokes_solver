import numpy as np
from scipy.sparse import csr_matrix
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

def make_step2_output_dummy(nx=4, ny=4, nz=4):
    """
    Step 2 Dummy: Mimics the exact output of orchestrate_step2.
    
    Constitutional Role: 
    Inherits Step 1 data and populates the 'Mathematical' safes:
    - Operators (Sparse Matrices)
    - PPE System (A and Preconditioner)
    - Initial Health (Vitals)
    - Advection Stencils
    """
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Precise staggered grid DOF calculation
    dof_p = nx * ny * nz
    dof_u = (nx + 1) * ny * nz
    dof_v = nx * (ny + 1) * nz
    dof_w = nx * ny * (nz + 1)
    total_vel_dof = dof_u + dof_v + dof_w

    # Hydrate the operators with correct shapes to avoid matmul mismatch
    state.operators._divergence = csr_matrix((dof_p, total_vel_dof))
    state.operators._grad_x = csr_matrix((dof_u, dof_p))
    state.operators._grad_y = csr_matrix((dof_v, dof_p))
    state.operators._grad_z = csr_matrix((dof_w, dof_p))
    state.operators._laplacian = csr_matrix((dof_p, dof_p))
    
    # PPE and Advection hydration
    state.ppe._A = state.operators._laplacian
    state.advection._weights = np.zeros((total_vel_dof, 8))
    state.advection._indices = np.zeros((total_vel_dof, 8), dtype=int)

    return state
