# tests/helpers/solver_step5_output_dummy.py

from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy


def make_step5_output_dummy(nx=4, ny=4, nz=4):
    """
    Generates state immediately after orchestrate_step5.
    Verifies that the manifest has been updated.
    """
    state = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # State reflects the iteration, typically the first snapshot
    state.iteration = 0
    
    # Simulating that Step 5 has run and updated the manifest
    state.manifest.saved_snapshots.append("output/snapshot_0000.h5")
    
    return state