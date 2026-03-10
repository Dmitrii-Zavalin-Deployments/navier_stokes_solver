# tests/helpers/solver_output_schema_dummy.py

from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

def make_output_schema_dummy(nx=4, ny=4, nz=4):
    """
    Generates a minimal, clean state representing the system
    after the final orchestration step (Step 5).
    """
    # 1. Start with the established Step 4 state
    state = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)

    # 2. Archivist Manifest
    # Only populate what the Archivist explicitly updates.
    state.manifest.output_directory = "output"
    state.manifest.saved_snapshots = [
        "output/snapshot_0000.h5",
        "output/snapshot_0500.h5",
        "output/snapshot_1000.h5"
    ]
    
    return state