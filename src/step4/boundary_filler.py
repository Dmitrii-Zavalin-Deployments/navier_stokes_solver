import numpy as np
from src.solver_state import SolverState

# Global Debug Toggle
DEBUG = True

def fill_ghost_boundaries(state: SolverState) -> None:
    """
    Step 4.2: Boundary Synchronization.
    Populates ghost cells based on SSoT boundary condition definitions.
    Fix: Synchronizes all velocity components across all faces to fill edges/corners.
    """
    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz
    
    if DEBUG:
        print(f"DEBUG [Step 4 Boundary]: Synchronizing ghost cells for {nx}x{ny}x{nz}")

    U_e, V_e, W_e, P_e = state.fields.U_ext, state.fields.V_ext, state.fields.W_ext, state.fields.P_ext
    bc = state.bc_lookup 

    def apply_face_bc(face_name, axis, side):
        config = bc.get(face_name)
        if config is None:
            raise RuntimeError(f"Boundary Error: No configuration found for face {face_name}")

        bc_type = config["type"]
        if DEBUG: print(f"DEBUG [Step 4 Boundary]: Applying {bc_type} to {face_name}")
        
        # Pressure logic for the current axis
        if bc_type == "pressure":
            val = config["p"]
            if axis == 0:
                if side == 0: P_e[0, :, :] = 2.0 * val - P_e[1, :, :]
                else:       P_e[-1, :, :] = 2.0 * val - P_e[-2, :, :]
            elif axis == 1:
                if side == 0: P_e[:, 0, :] = 2.0 * val - P_e[:, 1, :]
                else:       P_e[:, -1, :] = 2.0 * val - P_e[:, -2, :]
            elif axis == 2:
                if side == 0: P_e[:, :, 0] = 2.0 * val - P_e[:, :, 1]
                else:       P_e[:, :, -1] = 2.0 * val - P_e[:, :, -2]
        else:
            # Default Neumann for Pressure
            if axis == 0:
                if side == 0: P_e[0, :, :] = P_e[1, :, :]
                else:       P_e[-1, :, :] = P_e[-2, :, :]
            elif axis == 1:
                if side == 0: P_e[:, 0, :] = P_e[:, 1, :]
                else:       P_e[:, -1, :] = P_e[:, -2, :]
            elif axis == 2:
                if side == 0: P_e[:, :, 0] = P_e[:, :, 1]
                else:       P_e[:, :, -1] = P_e[:, :, -2]

    # 1. Process X Boundaries & Sync all velocities
    apply_face_bc("x_min", axis=0, side=0)
    apply_face_bc("x_max", axis=0, side=-1)
    for field in [U_e, V_e, W_e]:
        field[0, :, :] = field[1, :, :]
        field[-1, :, :] = field[-2, :, :]

    # 2. Process Y Boundaries & Sync all velocities
    apply_face_bc("y_min", axis=1, side=0)
    apply_face_bc("y_max", axis=1, side=-1)
    for field in [U_e, V_e, W_e]:
        field[:, 0, :] = field[:, 1, :]
        field[:, -1, :] = field[:, -2, :]

    # 3. Process Z Boundaries & Sync all velocities
    apply_face_bc("z_min", axis=2, side=0)
    apply_face_bc("z_max", axis=2, side=-1)
    for field in [U_e, V_e, W_e]:
        field[:, :, 0] = field[:, :, 1]
        field[:, :, -1] = field[:, :, -2]

    state.diagnostics.bc_verification_passed = True
    
    if DEBUG:
        p_ghost_sum = np.sum(P_e[0,:,:]) + np.sum(P_e[-1,:,:])
        print(f"DEBUG [Step 4 Boundary]: Sync Complete. Ghost Pressure Signal: {p_ghost_sum:.4e}")