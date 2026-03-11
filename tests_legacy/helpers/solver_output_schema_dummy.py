# tests/helpers/solver_output_schema_dummy.py

from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy


def make_output_schema_dummy(nx=4, ny=4, nz=4):
    """
    Simulates the terminal state of the entire main_solver.py pipeline.
    This acts as the 'Gold Standard' for integration tests.
    """
    state = make_step5_output_dummy(nx=nx, ny=ny, nz=nz)

    # 1. Final state markers (End of simulation)
    state.ready_for_time_loop = False
    
    # 2. Complete Archive Manifest (All iterations saved)
    # This reflects the end-of-simulation state of the manifest
    state.manifest.saved_snapshots = [
        "output/snapshot_0000.h5",
        "output/snapshot_0500.h5",
        "output/snapshot_1000.h5"
    ]
    
    return state