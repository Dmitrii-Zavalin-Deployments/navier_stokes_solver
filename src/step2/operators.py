# src/step2/operators.py

import scipy.sparse as sp
from src.solver_state import SolverState

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

    # --- 1. BUILD GRADIENTS (P -> U, V, W faces) ---
    Gx = sp.lil_matrix((dof_u, dof_p))
    Gy = sp.lil_matrix((dof_v, dof_p))
    Gz = sp.lil_matrix((dof_w, dof_p))

    # Gx maps P-centers to U-faces (i indices 1 to nx-1)
    for k in range(nz):
        for j in range(ny):
            for i in range(1, nx):
                idx_u = i + j*(nx+1) + k*(nx+1)*ny
                Gx[idx_u, i + j*nx + k*nx*ny] = 1.0 / dx
                Gx[idx_u, (i-1) + j*nx + k*nx*ny] = -1.0 / dx

    # Gy maps P-centers to V-faces (j indices 1 to ny-1)
    for k in range(nz):
        for j in range(1, ny):
            for i in range(nx):
                idx_v = i + j*nx + k*nx*(ny+1)
                Gy[idx_v, i + j*nx + k*nx*ny] = 1.0 / dy
                Gy[idx_v, i + (j-1)*nx + k*nx*ny] = -1.0 / dy

    # Gz maps P-centers to W-faces (k indices 1 to nz-1)
    for k in range(1, nz):
        for j in range(ny):
            for i in range(nx):
                idx_w = i + j*nx + k*nx*ny
                Gz[idx_w, i + j*nx + k*nx*ny] = 1.0 / dz
                Gz[idx_w, i + j*nx + (k-1)*nx*ny] = -1.0 / dz

    # --- 2. BUILD DIVERGENCE (U, V, W -> P) ---
    D = sp.lil_matrix((dof_p, total_vel_dof))
    u_offset = 0
    v_offset = dof_u
    w_offset = dof_u + dof_v

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx_p = i + j*nx + k*nx*ny
                
                # U contrib (East/West faces)
                D[idx_p, u_offset + (i+1) + j*(nx+1) + k*(nx+1)*ny] += 1.0 / dx
                D[idx_p, u_offset + i + j*(nx+1) + k*(nx+1)*ny] -= 1.0 / dx
                
                # V contrib (North/South faces)
                D[idx_p, v_offset + i + (j+1)*nx + k*nx*(ny+1)] += 1.0 / dy
                D[idx_p, v_offset + i + j*nx + k*nx*(ny+1)] -= 1.0 / dy
                
                # W contrib (Top/Bottom faces)
                D[idx_p, w_offset + i + j*nx + (k+1)*nx*ny] += 1.0 / dz
                D[idx_p, w_offset + i + j*nx + k*nx*ny] -= 1.0 / dz

    # --- 3. BUILD COMPOSITE LAPLACIAN (L = D * G) ---
    # This ensures the Pressure Solve perfectly "undoes" the Divergence.
    state.operators.grad_x = Gx.tocsr()
    state.operators.grad_y = Gy.tocsr()
    state.operators.grad_z = Gz.tocsr()
    state.operators.divergence = D.tocsr()

    # Create the global gradient vector operator
    G_total = sp.vstack([state.operators.grad_x, state.operators.grad_y, state.operators.grad_z])
    
    # Fundamental Projection Identity: L = D @ G
    state.operators.laplacian = state.operators.divergence @ G_total
    
    # Handshake: The PPE solver A-matrix is the Laplacian
    state.ppe._A = state.operators.laplacian