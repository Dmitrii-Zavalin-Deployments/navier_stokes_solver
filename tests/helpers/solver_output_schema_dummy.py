# tests/helpers/solver_output_schema_dummy.py

"""
Archivist Testing: Explicit Terminal State Hydration.

Compliance:
- Rule 5: Deterministic Initialization (No implicit defaults).
- Rule 8: API Minimalism (Primary state only).
"""

from src.common.solver_state import SolverState
from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy


def create_terminal_state_dummy(nx: int = 4, ny: int = 4, nz: int = 4) -> SolverState:
    """
    Hydrates a terminal SolverState object.
    Ensures all sub-containers are explicitly defined per Rule 4.
    """
    # 1. Initialize from valid intermediate state
    state = make_step5_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 2. Terminal Markers: 
    # Using the existing flag, but ensuring it is set explicitly 
    # to maintain strict control over the solver life-cycle.
    state.ready_for_time_loop = False
    
    # 3. Complete Archive Manifest
    # SSoT: Manifest exists within its own dedicated container
    # This prevents the solver state from holding redundant metadata.
    state.manifest.update_snapshots([
        "output/snapshot_0000.h5",
        "output/snapshot_0500.h5",
        "output/snapshot_1000.h5"
    ])
    
    return state