# src/step2/operators.py

import scipy.sparse as sp
from src.solver_state import SolverState

# Global Debug Toggle
DEBUG = True

def build_numerical_operators(state: SolverState) -> None:
    """
    Step 2 Logic: Populate Discrete Calculus Operators.
    Implements Staggered Divergence/Gradient and a Composite Laplacian.
    Rule 5 Compliance: No placeholders. L = D @ G.
    """
    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz
    dx, dy, dz = state.grid.dx, state.grid.dy, state.grid.dz
    
    dof_p = nx * ny * nz
    dof_u = (nx + 1) * ny * nz
    dof_v = nx * (ny + 1) * nz
    dof_w = nx * ny * (nz + 1)
    total_vel_dof = dof_u + dof_v + dof_w

    if DEBUG:
        print(f"DEBUG [Step 2]: Grid dimensions {nx}x{ny}x{nz}")
        print(f"DEBUG [Step 2]: Target DOFs - P:{dof_p}, U:{dof_u}, V:{dof_v}, W:{dof_w}")

    # --- 1. BUILD GRADIENTS (P -> U, V, W faces) ---
    Gx = sp.lil_matrix((dof_u, dof_p))
    Gy = sp.lil_matrix((dof_v, dof_p))
    Gz = sp.lil_matrix((dof_w, dof_p))

    for k in range(nz):
        for j in range(ny):
            for i in range(1, nx):
                idx_u = i + j*(nx+1) + k*(nx+1)*ny
                Gx[idx_u, i + j*nx + k*nx*ny] = 1.0 / dx
                Gx[idx_u, (i-1) + j*nx + k*nx*ny] = -1.0 / dx

    for k in range(nz):
        for j in range(1, ny):
            for i in range(nx):
                idx_v = i + j*nx + k*nx*(ny+1)
                Gy[idx_v, i + j*nx + k*nx*ny] = 1.0 / dy
                Gy[idx_v, i + (j-1)*nx + k*nx*ny] = -1.0 / dy

    for k in range(1, nz):
        for j in range(ny):
            for i in range(nx):
                idx_w = i + j*nx + k*nx*ny
                Gz[idx_w, i + j*nx + k*nx*ny] = 1.0 / dz
                Gz[idx_w, i + j*nx + (k-1)*nx*ny] = -1.0 / dz

    if DEBUG:
        print(f"DEBUG [Step 2]: Gx non-zeros: {Gx.nnz}, Gy nnz: {Gy.nnz}, Gz nnz: {Gz.nnz}")

    # --- 2. BUILD DIVERGENCE (U, V, W -> P) ---
    D = sp.lil_matrix((dof_p, total_vel_dof))
    u_off, v_off, w_off = 0, dof_u, dof_u + dof_v

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx_p = i + j*nx + k*nx*ny
                D[idx_p, u_off + (i+1) + j*(nx+1) + k*(nx+1)*ny] += 1.0 / dx
                D[idx_p, u_off + i + j*(nx+1) + k*(nx+1)*ny] -= 1.0 / dx
                D[idx_p, v_off + i + (j+1)*nx + k*nx*(ny+1)] += 1.0 / dy
                D[idx_p, v_off + i + j*nx + k*nx*(ny+1)] -= 1.0 / dy
                D[idx_p, w_off + i + j*nx + (k+1)*nx*ny] += 1.0 / dz
                D[idx_p, w_off + i + j*nx + k*nx*ny] -= 1.0 / dz

    if DEBUG:
        print(f"DEBUG [Step 2]: Divergence matrix built. Shape: {D.shape}, nnz: {D.nnz}")

    # --- 3. BUILD COMPOSITE LAPLACIAN (L = D * G) ---
    state.operators.grad_x = Gx.tocsr()
    state.operators.grad_y = Gy.tocsr()
    state.operators.grad_z = Gz.tocsr()
    state.operators.divergence = D.tocsr()

    G_total = sp.vstack([state.operators.grad_x, state.operators.grad_y, state.operators.grad_z])
    
    if DEBUG:
        print(f"DEBUG [Step 2]: Global Gradient V-Stack shape: {G_total.shape}")

    state.operators.laplacian = state.operators.divergence @ G_total
    
    if DEBUG:
        nnz_L = state.operators.laplacian.nnz
        print(f"DEBUG [Step 2]: Laplacian (D@G) nnz: {nnz_L}")
        if nnz_L == 0:
            print("!!! CRITICAL: Laplacian is empty. Check DOF mapping !!!")

    state.ppe._A = state.operators.laplacian