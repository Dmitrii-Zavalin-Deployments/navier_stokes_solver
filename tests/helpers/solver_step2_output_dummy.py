# tests/helpers/solver_step2_output_dummy.py

from src.common.solver_state import FieldManager  # Added import
from src.step2.factory import get_initialization_context
from src.step2.stencil_assembler import assemble_stencil_matrix
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


def make_step2_output_dummy(nx=4, ny=4, nz=4):
    """
    Generates a valid SolverState with a populated FieldManager foundation 
    and a stencil_matrix wiring.
    """
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 1. Allocate the Foundation (Rule 9)
    # We must allocate the data before the assembler expects the cells to have indices
    n_cells = nx * ny * nz
    state.fields = FieldManager()
    state.fields.allocate(n_cells)
    
    # 2. Prepare minimal context and physics params
    ctx = get_initialization_context(state)
    physics_params = {
        "dx": 0.1, "dy": 0.1, "dz": 0.1, 
        "dt": 0.01, "rho": 1.0, "mu": 0.001, 
        "f_vals": (0.0, 0.0, 0.0)
    }
    
    # 3. Populate the stencil_matrix (Wiring)
    state.stencil_matrix = assemble_stencil_matrix(
        state, nx, ny, nz, ctx, physics_params
    )
    
    # 4. State Baseline
    state.ready_for_time_loop = True
    
    return state