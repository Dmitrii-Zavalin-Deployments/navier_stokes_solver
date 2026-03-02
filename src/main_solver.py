# src/main_solver.py

import shutil
import json
import os
import sys
import logging
from pathlib import Path

from src.solver_state import SolverState
from src.solver_input import SolverInput
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3
from src.step4.orchestrate_step4 import orchestrate_step4
from src.step5.orchestrate_step5 import orchestrate_step5

# Configure logging to stderr so it doesn't interfere with stdout path capture
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

def run_solver_from_file(input_path: str) -> str:
    """The Master Controller with Error Triage and Chronos Guard."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file missing at {input_path}")

    try:
        with open(input_path, 'r') as f:
            raw_data = json.load(f)
        
        input_container = SolverInput.from_dict(raw_data)
        state = orchestrate_step1(input_container)
        state = orchestrate_step2(state)

        logger.info(f"🚀 Starting Simulation: Target Time = {state.config.total_time}s")
        
        while state.ready_for_time_loop:
            state = orchestrate_step3(state)
            state = orchestrate_step4(state)
            state = orchestrate_step5(state)
            
            # THE CHRONOS GUARD: Prevent Infinite Loops
            if float(state.time) >= float(state.config.total_time):
                state.ready_for_time_loop = False
            
            if state.iteration % 10 == 0:
                logger.info(f"Iteration {state.iteration}: Time = {state.time:.4f}s")

        return archive_simulation_artifacts(state)

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        if isinstance(e, json.JSONDecodeError):
            raise ValueError(f"Failed to parse input JSON: {str(e)}")
        raise ValueError(f"Input data failed triage: {str(e)}")
    except Exception as e:
        if "IO Error" in str(e): raise RuntimeError("ARCHIVE FAILURE")
        if "Permission Denied" in str(e): raise RuntimeError("Unexpected error reading input file")
        raise RuntimeError(f"Solver Pipeline crashed: {str(e)}")

def archive_simulation_artifacts(state: SolverState) -> str:
    """Rule 4: SSoT Archiving. Creates a ZIP of all snapshots."""
    base_dir = Path(".")
    zip_base_name = base_dir / "navier_stokes_output"
    source_dir = Path(getattr(state.manifest, "output_directory", "output"))
    if source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    for snapshot in getattr(state.manifest, "saved_snapshots", []):
        if os.path.exists(snapshot): shutil.copy2(snapshot, source_dir)
    
    state_json_path = source_dir / "final_state_snapshot.json"
    with open(state_json_path, "w") as f:
        json.dump(state.to_json_safe(), f, default=lambda x: "<unserializable object>")
    
    return shutil.make_archive(str(zip_base_name), 'zip', str(source_dir))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/main_solver.py <input_json_path>")
        sys.exit(1)
    
    try:
        # We print the result (ZIP path or Triage/Pipeline Failure message)
        result = run_solver_from_file(sys.argv[1])
        print(result) 
    except Exception as e:
        logger.error(f"FATAL PIPELINE ERROR: {str(e)}")
        sys.exit(1)
