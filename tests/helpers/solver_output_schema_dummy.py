# tests/helpers/solver_output_schema_dummy.py

import numpy as np
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

def make_output_schema_dummy(nx=4, ny=4, nz=4):
    state = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)

    state.time = 1.0
    state.iteration = 1000 
    
    if hasattr(state, "step_index"):
        state.step_index = 1000

    state.manifest.output_directory = "output/simulation_results"
    state.manifest.saved_snapshots = [
        "output/snapshot_0000.vtk",
        "output/snapshot_0500.vtk",
        f"output/snapshot_{state.iteration:04d}.vtk"
    ]
    state.manifest.final_checkpoint = f"output/checkpoint_final_{state.iteration}.npy"
    state.manifest.log_file = "output/solver_convergence.log"

    state.health.is_stable = True
    state.health.post_correction_divergence_norm = 1e-15
    state.health.max_u = 1.2 

    state.ready_for_time_loop = False
    return state