# tests/helpers/solver_step5_output_dummy.py

"""
Archivist Testing: Snapshot-based Test Baseline (Step 5).

Compliance:
- Rule 4: SSoT (Hierarchy over convenience).
- Rule 8: Law of Singular Access (Use established state containers).
"""

from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy


def make_step5_output_dummy(nx: int = 4, ny: int = 4, nz: int = 4):
    """
    Returns a 'frozen' prototype representing the system state immediately 
    after Step 5 (Manifest Archival).
    """
    # 1. Hydrate foundation from Step 4 baseline
    state = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 2. Update State (SSoT Compliance)
    # Ensure attributes are updated directly within the state hierarchy.
    # Rule 5: No defaults. We define the iteration explicitly.
    state.iteration = 0
    
    # 3. Manifest Update
    # The manifest is a container within the state. We verify the archival
    # trigger by checking the existence of the snapshot path.
    # Rule 4: Do not create aliases or facade properties to access the manifest.
    if not hasattr(state, 'manifest'):
        raise RuntimeError("CRITICAL: Manifest container missing from state.")
        
    state.manifest.saved_snapshots.append("output/snapshot_0000.h5")
    
    return state