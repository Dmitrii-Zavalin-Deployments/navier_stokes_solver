def orchestrate_step4(state: SolverState, dt: float) -> SolverState:
    # 1. Predictor: Solve for v* (Momentum)
    v_star = predictor.compute(state, dt)
    
    # 2. Rhie-Chow Corrected PPE: Solve for p^{n+1}
    # Uses corrected divergence: div(v*) - div(M_rc * p^n)
    p_next = ppe_solver.solve(state, v_star, dt)
    
    # 3. Corrector: Project onto divergence-free space
    v_next = corrector.compute(state, v_star, p_next, dt)
    
    # 4. State Update (Active Working Set)
    state.fields.U, state.fields.V, state.fields.W = v_next
    state.fields.P = p_next
    
    # 5. Archive Trigger: Snapshot the matrix state if iteration matches frequency
    if state.iteration % state.config.archive_frequency == 0:
        state.archive.save_snapshot(state)
        
    return state