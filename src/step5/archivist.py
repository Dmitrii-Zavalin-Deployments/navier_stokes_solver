# src/step5/archivist.py

import os
import numpy as np
from src.solver_state import SolverState

# Global Debug Toggle
DEBUG = True

def record_snapshot(state: SolverState) -> None:
    """
    Step 5.1: Archivist. Writes physical artifacts to disk.
    Rule 5 Compliance: No defaults. Uses explicit config paths.
    """
    # 1. EXPLICIT PATH RESOLUTION (No getattr/hasattr defaults)
    # We use the SSoT from state.config directly.
    base_dir = state.config.output_directory
    case_name = state.config.case_name
    out_dir = os.path.join(base_dir, case_name)
    
    if DEBUG:
        print(f"DEBUG [Step 5 Archivist]: Preparing output at {out_dir}")

    # 2. Filesystem Management
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        if DEBUG: print(f"DEBUG [Step 5 Archivist]: Created directory {out_dir}")
    
    # 3. Artifact Generation
    snap_name = f"snapshot_{state.iteration:04d}.vtk"
    snap_path = os.path.join(out_dir, snap_name)
    
    # Logic for VTK Export (Strictly using current state data)
    # This replaces the text placeholder with actual field metadata for validation
    with open(snap_path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"Navier-Stokes Solver Snapshot - Iteration {state.iteration}\n")
        f.write("ASCII\nDATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {state.grid.nx} {state.grid.ny} {state.grid.nz}\n")
        f.write(f"METADATA: TIME={state.time:.6f}, DIV_NORM={state.health.divergence_norm:.2e}\n")

    if DEBUG:
        print(f"DEBUG [Step 5 Archivist]: Snapshot saved to {snap_path}")

    # 4. Manifest Update (SSoT Sync)
    state.manifest.output_directory = out_dir
    
    # Pythonic list update without safe-check placeholders
    state.manifest.saved_snapshots.append(snap_path)
    state.manifest.log_file = os.path.join(out_dir, "solver_convergence.log")

    if DEBUG:
        print(f"DEBUG [Step 5 Archivist]: Manifest updated. Total snapshots: {len(state.manifest.saved_snapshots)}")