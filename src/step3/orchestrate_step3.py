# src/step3/orchestrate_step3.py

from src.solver_state import SolverState
from .predictor import predict_velocity
from .solver import solve_pressure
from .corrector import correct_velocity

def orchestrate_step3(state: SolverState) -> SolverState:
    """
    Step 3 Orchestrator: Physical Integration.
    Sequence: Predict (V*) -> Solve (P) -> Correct (V_new).
    """
    # 1. Execute Numerical Steps
    predict_velocity(state)
    status = solve_pressure(state)
    correct_velocity(state)

    # 2. Update Progression 
    # Using state.dt facade for consistency with the corrected SolverConfig
    state.iteration += 1
    state.time += state.dt
    
    # 3. History Persistence (SSoT Rule 4)
    # Using getattr for health metrics to ensure robustness during the first iteration
    state.history.times.append(state.time)
    state.history.divergence_norms.append(getattr(state.health, 'divergence_norm', 0.0))
    state.history.max_velocity_history.append(getattr(state.health, 'max_u', 0.0))
    state.history.ppe_status_history.append(status)
    
    # Flip the loop-readiness bit now that one full iteration is valid
    state.ready_for_time_loop = True 
    
    return state
