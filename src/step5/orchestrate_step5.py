# src/step5/orchestrate_step5.py

from src.common.simulation_context import SimulationContext
from src.common.solver_state import SolverState
from src.step5.io_archivist import save_snapshot


def orchestrate_step5(state: SolverState, context: SimulationContext) -> SolverState:
    print('DEBUG: Explicitly flushing field buffers to archive...')
    """
    Step 5: The Archivist Orchestration.
    
    Compliance:
    - Rule 4 (SSoT): Accesses output interval exclusively via simulation_parameters.
    - Rule 5 (Deterministic Init): Relies on explicit iteration counts from the input schema.
    - Rule 9 (Hybrid Memory): Logic-layer remains thin; archiving is delegated.
    """
    
    # Rule 4: SSoT Compliance
    # Output frequency is a simulation parameter defined in the Input Schema,
    # not an algorithmic tuning parameter (SolverConfig).
    interval = context.input_data.simulation_parameters.output_interval
    
    # Logic-layer operation: Decision to archive
    # state.iteration is a property managed within the SolverState lifecycle
    if state.iteration % interval == 0:
        # Rule 4: Data persistence delegated to the Archivist.
        # No serialization logic here; orchestration stays thin and focused.
        save_snapshot(state)
        
    return state