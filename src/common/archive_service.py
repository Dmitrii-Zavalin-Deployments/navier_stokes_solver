# src/common/archive_service.py

import json
import shutil
from pathlib import Path
from src.common.solver_state import SolverState

# Toggle DEBUG based on global settings
DEBUG = True 

def get_json_schema_compliant_dict(state: SolverState) -> dict:
    """
    Acts as the 'View' layer. Per Rule 8 (Singular Access), this function
    serves as the sole authorized interface for transforming internal 
    containers into the persistent output schema.
    """
    # Each manager now handles its own serialization via the 
    # ValidatedContainer.to_dict() interface, ensuring SSoT integrity.
    return {
        "time": float(state.time),
        "iteration": int(state.iteration),
        "ready_for_time_loop": bool(state.ready_for_time_loop),
        
        # Rule 4: Data must reside in its assigned container.
        # We access these through the validated property gates.
        "config": state.sim_params.to_dict(),
        "grid": {
            "nx": state.grid.nx, "ny": state.grid.ny, "nz": state.grid.nz,
            "dx": state.grid.dx, "dy": state.grid.dy, "dz": state.grid.dz
        },
        "fields": {
            "data": state.fields.data.tolist() # Foundation Buffer -> List
        },
        "masks": {
            "mask": state.masks.mask.tolist()   # Foundation Mask -> List
        },
        "manifest": state._manifest # Accessing internal manifest state
    }

def archive_simulation_artifacts(state: SolverState) -> str:
    """
    Rule 4: SSoT Archiving. Performs final snapshotting.
    This service is decoupled from the physics integration loop to 
    ensure zero performance impact (Rule 0).
    """
    base_dir = Path(".")
    
    # Per Rule 5 (Deterministic Initialization), the case_name must be 
    # present in the sim_params or domain; otherwise, raise explicit error.
    case_name = getattr(state.domain, 'case_name', None)
    if case_name is None:
        raise AttributeError("CRITICAL: case_name missing from DomainManager.")
        
    zip_base_name = base_dir / f"navier_stokes_{case_name}_output"
    source_dir = Path("output")
    
    # Ensure cleanup/init of output sink
    source_dir.mkdir(parents=True, exist_ok=True)

    # 1. Map state to schema and persist (The 'Atomic' write)
    state_json_path = source_dir / "final_state_snapshot.json"
    with open(state_json_path, "w") as f:
        json.dump(get_json_schema_compliant_dict(state), f, indent=4)
    
    # 2. Package into single archive
    result_path = shutil.make_archive(str(zip_base_name), 'zip', str(source_dir))
    
    if DEBUG:
        print(f"DEBUG [Archive]: Artifact successfully packaged at: {result_path}")
    
    return str(result_path)