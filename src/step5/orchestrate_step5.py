def orchestrate_step5(state: SolverState) -> SolverState:
    # 1. Apply Physical Constraints (Ghost Cell Synchronization)
    # This uses the masks generated in Step 2 to ensure we don't 
    # touch internal fluid cells, only boundary-associated ghost cells.
    apply_boundary_conditions(state)
    
    # 2. Finalize and Archive
    # Only now is the state "Physical" enough to be saved for analysis.
    if state.iteration % state.config.archive_frequency == 0:
        state.archive.save_snapshot(state)
        
    return state