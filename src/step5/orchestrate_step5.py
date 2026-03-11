# src/step5/orchestrate_step5.py

from src.step5.io_archivist import save_snapshot

def orchestrate_step5(state) -> object:
    """
    Step 5: The Archivist Orchestration.
    
    Compliance:
    - Rule 4 (SSoT): Accesses output interval exclusively via state.config.
    - Rule 5 (Deterministic Init): Relies on explicit iteration counts; no default interval logic.
    - Rule 9 (Hybrid Memory): Maintains zero-copy interaction with the Foundation buffer.
    """
    
    # Rule 5: Accessing configuration explicitly.
    # If output_interval is missing, this will raise an AttributeError,
    # ensuring the simulation does not proceed with unverified defaults.
    interval = state.config.output_interval
    
    # Logic-layer operation: Decision to archive
    if state.iteration % interval == 0:
        # Rule 4: Data persistence delegated to the Archivist.
        # No serialization logic here; orchestration stays thin and focused.
        save_snapshot(state)
        
    return state