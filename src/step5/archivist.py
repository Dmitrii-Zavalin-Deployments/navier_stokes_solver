# src/step5/archivist.py

import os
from src.solver_state import SolverState

def record_snapshot(state: SolverState) -> None:
    """
    Step 5.1: Archivist. Writes physical artifacts to disk.
    """
    # Safeguard: Pull output_directory or default to case_name
    out_dir = getattr(state.config, 'output_directory', 'output')
    if out_dir == 'output' and hasattr(state.config, 'case_name'):
        out_dir = os.path.join('output', state.config.case_name)
    
    # Ensure directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # Generate standard file names
    snap_name = f"snapshot_{state.iteration:04d}.vtk"
    snap_path = os.path.join(out_dir, snap_name)
    
    # Write Artifacts (Placeholder for real VTK export)
    with open(snap_path, "w") as f:
        f.write(f"Step: {state.iteration}\nTime: {state.time}")

    # Update Manifest
    state.manifest.output_directory = out_dir
    
    # Handle list persistence safely
    current_snapshots = list(state.manifest.saved_snapshots)
    if snap_path not in current_snapshots:
        current_snapshots.append(snap_path)
        state.manifest.saved_snapshots = current_snapshots
    
    state.manifest.log_file = os.path.join(out_dir, "solver_convergence.log")
