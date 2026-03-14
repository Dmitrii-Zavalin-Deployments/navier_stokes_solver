# tests/helpers/solver_step5_output_dummy.py

from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy


def make_step5_output_dummy(nx: int = 4, ny: int = 4, nz: int = 4):
    """
    Returns a 'frozen' prototype representing the system state after 
    at least one iteration of the Step 5 archival process.
    """
    # 1. Hydrate foundation
    state = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 2. Set Iteration/Time context
    state.iteration = 1
    state.time = 0.01 
    
    # 3. Manifest Integrity (Correctly appending to mimic main_solver loop)
    if not hasattr(state, 'manifest'):
        raise RuntimeError("CRITICAL: Manifest container missing from state.")
    
    # Ensure initialized as an empty list if not already
    if not hasattr(state.manifest, 'saved_snapshots'):
        state.manifest.saved_snapshots = []
        
    # Use append to reflect the actual iterative history build-up
    state.manifest.saved_snapshots.append("output/snapshot_0000.h5")
    state.manifest.output_directory = "output/"
    
    return state