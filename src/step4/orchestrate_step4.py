def orchestrate_step4(state: SolverState, dt: float) -> SolverState:
    # 1. Predictor: Solve for intermediate velocity v*
    v_star = predictor.compute(state, dt)
    
    # 2. Rhie-Chow Corrected PPE: Solve for p^{n+1}
    p_next = ppe_solver.solve(state, v_star, dt)
    
    # 3. Corrector: Project onto divergence-free space to get v^{n+1}
    v_next = corrector.compute(state, v_star, p_next, dt)
    
    # 4. State Update: Commit the new physical state to the Active Working Set
    # This overwrites the buffers, keeping memory footprint constant.
    state.fields.U, state.fields.V, state.fields.W = v_next
    state.fields.P = p_next
    
    # Note: No archive trigger here. We defer archival to Step 5 
    # to ensure the snapshot is physically bounded (BCs applied).
    
    return state