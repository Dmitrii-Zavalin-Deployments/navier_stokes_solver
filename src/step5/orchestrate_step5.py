# src/step5/orchestrate_step5.py

from src.common.simulation_context import SimulationContext
from src.common.solver_state import SolverState
from src.step5.io_archivist import save_snapshot

def orchestrate_step5(state: SolverState, context: SimulationContext) -> SolverState:
    """
    Step 5: The Archivist Orchestration.
    
    Compliance:
    - Rule 4 (SSoT): Accesses output interval exclusively via context.config.
    - Rule 5 (Deterministic Init): Relies on explicit iteration counts from the configuration.
    - Rule 9 (Hybrid Memory): Logic-layer remains thin; archiving is delegated.
    """
    
    # Rule 5: Accessing configuration explicitly from the unified context.
    # This prevents the silent failure of using a hardcoded default.
    interval = context.config.output_interval
    
    # Logic-layer operation: Decision to archive
    # state.iteration is a property managed within the SolverState lifecycle
    if state.iteration % interval == 0:
        # Rule 4: Data persistence delegated to the Archivist.
        # No serialization logic here; orchestration stays thin and focused.
        save_snapshot(state)
        
    return state