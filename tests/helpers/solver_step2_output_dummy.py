# tests/helpers/solver_step2_output_dummy.py

import numpy as np
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from src.step2.cell import Cell
from src.step2.stencil_block import StencilBlock
from src.step2.stencil_assembler import assemble_stencil_matrix
from src.step2.factory import get_initialization_context

def make_step2_output_dummy(nx=4, ny=4, nz=4):
    """
    Generates a valid SolverState with a populated stencil_matrix.
    """
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 1. Prepare minimal context and physics params (matching orchestrator logic)
    ctx = get_initialization_context(state)
    physics_params = {
        "dx": 0.1, "dy": 0.1, "dz": 0.1, 
        "dt": 0.01, "rho": 1.0, "mu": 0.001, 
        "f_vals": (0.0, 0.0, 0.0)
    }
    
    # 2. Populate the stencil_matrix using the assembler
    # This ensures your dummy state perfectly mirrors the production logic
    state.stencil_matrix = assemble_stencil_matrix(
        state, nx, ny, nz, ctx, physics_params
    )
    
    # 3. State Baseline
    state.ready_for_time_loop = True
    
    return state