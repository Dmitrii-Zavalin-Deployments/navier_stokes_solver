# src/main_solver.py

import json
import logging
import os
import shutil
import sys
from pathlib import Path

import jsonschema

from src.solver_input import SolverInput
from src.solver_state import SolverState
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

def _load_solver_config() -> dict:
    config_path = Path("config.json")
    with open(config_path, "r") as f:
        return json.load(f)["solver_settings"]

def run_solver_from_file(input_path: str) -> str:
    """Master Controller: Orchestrates the pipeline with iterative PPE-Boundary coupling."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file missing at {input_path}")

    # Load configuration once (Hoisting)
    cfg = _load_solver_config()
    omega = cfg["ppe_omega"]
    max_iter = cfg["ppe_max_iter"]
    tol = cfg["ppe_tolerance"]

    SCHEMA_PATH = Path("schema/solver_input_schema.json")

    try:
        with open(input_path) as f:
            raw_data = json.load(f)
        
        # 1. INITIALIZATION & STATE ASSEMBLY
        input_container = SolverInput.from_dict(raw_data)
        state = orchestrate_step1(input_container)
        state = orchestrate_step2(state)

        # FIREWALL: Ensure the fully initialized state matches the master contract
        # before the simulation enters the physics loop.
        try:
            state.validate_against_schema(str(SCHEMA_PATH))
        except jsonschema.exceptions.ValidationError as e:
            print("\n" + "!" * 60)
            print("CONTRACT VIOLATION: Solver State does not match Schema")
            print(f"Path to error: {'.'.join([str(p) for p in e.path])}")
            print(f"Error message: {e.message}")
            print("!" * 60 + "\n")
            raise  # Re-raise to stop the execution

        if DEBUG:
            logger.info(f"🚀 Starting Simulation: {state.config.case_name}")
        
        # 2. MAIN EXECUTION LOOP
        while state.ready_for_time_loop:
            # A. PREDICTOR PASS (Once per time step)
            for block in state.stencil_matrix:
                # Calculate v*
                orchestrate_step3(block, omega, is_first_pass=True)
                # Enforce BCs on the predicted velocity v*
                orchestrate_step4(block)
            
            # B. ITERATIVE SOLVER & BOUNDARY PASS
            for _ in range(max_iter):
                max_delta = 0.0
                for block in state.stencil_matrix:
                    # Solve (SOR) -> Correct (v^{n+1}) -> Sync (p^{n+1})
                    block, delta = orchestrate_step3(block, omega, is_first_pass=False)
                    
                    # Enforce BCs on the newly corrected velocity v^{n+1}
                    block = orchestrate_step4(block)
                    
                    max_delta = max(max_delta, delta)
                
                if max_delta < tol:
                    break
            
            # C. ODOMETER UPDATE
            state.iteration += 1
            state.time += state.dt
            
            # D. FINALIZATION & GUARD
            state = orchestrate_step5(state)
            
            if DEBUG and state.iteration % 10 == 0:
                logger.info(f"Iter {state.iteration}: t={state.time:.4f}s | Delta={max_delta:.2e}")

        if DEBUG:
            logger.info("DEBUG [Main]: Loop exit detected. Finalizing artifacts.")

        return archive_simulation_artifacts(state)

    except Exception as e:
        logger.error(f"FATAL PIPELINE ERROR: {str(e)}")
        raise RuntimeError(f"Solver Pipeline crashed: {str(e)}") from e

# ... [archive_simulation_artifacts and if __name__ == "__main__" remain the same] ...