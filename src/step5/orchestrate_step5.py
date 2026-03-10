# src/step5/orchestrate_step5.py
from src.step5.io_archivist import save_snapshot

def orchestrate_step5(state) -> object:
    """
    Step 5: The Archivist.
    Decides when to trigger data persistence.
    """
    if state.iteration % state.config.output_interval == 0:
        save_snapshot(state)
        
    return state