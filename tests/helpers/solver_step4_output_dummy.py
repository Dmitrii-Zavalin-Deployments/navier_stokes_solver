# tests/helpers/solver_step4_output_dummy.py

from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def make_step4_output_dummy(nx=4, ny=4, nz=4):
    """
    Generates a valid SolverState representing the system 
    immediately after orchestrate_step4 has finished.
    """
    # 1. Start with the established Step 3 state
    state = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)

    # 2. Apply Boundary Enforcement (In-place modification)
    # This reflects exactly what orchestrate_step4 does to the blocks.
    for block in state.stencil_matrix:
        if not block.center.is_ghost:
            # Enforce physics: Solid (mask 0) and Wall (mask -1)
            # mapping schema u, v, w to Cell attributes vx, vy, vz
            if block.center.mask == 0 or block.center.mask == -1:
                block.center.vx = 0.0
                block.center.vy = 0.0
                block.center.vz = 0.0

    return state