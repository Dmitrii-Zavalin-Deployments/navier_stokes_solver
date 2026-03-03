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

# Global Debug Toggle
DEBUG = True

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

def run_solver_from_file(input_path: str) -> str:
    """Master Controller: Orchestrates the pipeline with strict logic separation."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file missing at {input_path}")

    try:
        with open(input_path, 'r') as f:
            raw_data = json.load(f)
        
        # 1. INITIALIZATION
        input_container = SolverInput.from_dict(raw_data)
        state = orchestrate_step1(input_container)
        state = orchestrate_step2(state)

        if DEBUG:
            logger.info(f"🚀 Starting Simulation: {state.config.case_name}")
        
        # 2. MAIN EXECUTION LOOP
        while state.ready_for_time_loop:
            # A. Physics & Boundaries (Steps 3 & 4)
            state = orchestrate_step3(state)
            state = orchestrate_step4(state)
            
            # B. ODOMETER UPDATE (Authority check happens after this)
            state.iteration += 1
            state.time += state.dt
            
            # C. FINALIZATION & GUARD (Step 5)
            # Step 5 now decides if state.ready_for_time_loop remains True
            state = orchestrate_step5(state)
            
            if DEBUG and state.iteration % 10 == 0:
                logger.info(f"Iter {state.iteration}: t={state.time:.4f}s | Div={state.health.divergence_norm:.2e}")

        if DEBUG:
            logger.info("DEBUG [Main]: Loop exit detected. Finalizing artifacts.")
        
        return archive_simulation_artifacts(state)

    except Exception as e:
        logger.error(f"FATAL PIPELINE ERROR: {str(e)}")
        raise RuntimeError(f"Solver Pipeline crashed: {str(e)}")

def archive_simulation_artifacts(state: SolverState) -> str:
    """Rule 4: SSoT Archiving. Moves manifest snapshots into a single ZIP."""
    base_dir = Path(".")
    zip_base_name = base_dir / f"navier_stokes_{state.config.case_name}_output"
    source_dir = Path(state.manifest.output_directory)

    # Final metadata capture
    state_json_path = source_dir / "final_state_snapshot.json"
    with open(state_json_path, "w") as f:
        json.dump(state.to_json_safe(), f, indent=4)
    
    result_path = shutil.make_archive(str(zip_base_name), 'zip', str(source_dir))
    
    if DEBUG:
        logger.info(f"DEBUG [Main]: Artifact created: {result_path}")
    
    return result_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    try:
        print(run_solver_from_file(sys.argv[1])) 
    except Exception:
        sys.exit(1)