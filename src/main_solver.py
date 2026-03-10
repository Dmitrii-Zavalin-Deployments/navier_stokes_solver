# src/main_solver.py

import json
import logging
import os
import shutil
import sys
from pathlib import Path

import jsonschema

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

def _load_solver_config() -> dict:
    config_path = Path("config.json")
    with open(config_path) as f:
        return json.load(f)["solver_settings"]

def run_solver_from_file(input_path: str) -> str:
    """Master Controller: Orchestrates the pipeline with iterative PPE-Boundary coupling."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file missing at {input_path}")

    # Load configuration once
    cfg = _load_solver_config()
    max_iter = cfg["ppe_max_iter"]
    tol = cfg["ppe_tolerance"]
    omega = cfg["ppe_omega"]

    SCHEMA_PATH = Path("schema/solver_input_schema.json")

    try:
        with open(input_path) as f:
            raw_data = json.load(f)
        
        # 1. INITIALIZATION & STATE ASSEMBLY
        input_container = SolverInput.from_dict(raw_data)
        state = orchestrate_step1(input_container)
        state = orchestrate_step2(state)

        # FIREWALL: Contract Validation
        try:
            state.validate_against_schema(str(SCHEMA_PATH))
            logger.info("✅ State validation passed.")
        except jsonschema.exceptions.ValidationError as e:
            logger.error(f"CONTRACT VIOLATION at {'.'.join([str(p) for p in e.path])}")
            logger.error(f"Message: {e.message}")
            raise

        if DEBUG:
            logger.info(f"🚀 Starting Simulation: {state.config.case_name}")
        
        # 2. MAIN EXECUTION LOOP
        while state.ready_for_time_loop:
            # A. PREDICTOR PASS
            for block in state.stencil_matrix:
                block, _ = orchestrate_step3(block, omega=omega, is_first_pass=True)
                block = orchestrate_step4(block, state.config.boundary_conditions, state.grid)
            
            # B. ITERATIVE SOLVER & BOUNDARY PASS
            for _ in range(max_iter):
                max_delta = 0.0
                for block in state.stencil_matrix:
                    block, delta = orchestrate_step3(block, omega=omega, is_first_pass=False)
                    block = orchestrate_step4(block, state.config.boundary_conditions, state.grid)
                    max_delta = max(max_delta, delta)
                
                # Convergence check for the Pressure-Poisson equation
                if max_delta < tol:
                    if DEBUG:
                        logger.info(
                            f"PPE Converged: Iter={_ + 1} | "
                            f"Delta={max_delta:.2e} < Tol={tol:.2e}"
                        )
                    break
            
            # C. ODOMETER UPDATE
            state.iteration += 1
            state.time += state.config.time_step
            
            # D. ARCHIVING (Step 5)
            state = orchestrate_step5(state)
            
            # E. TEMPORAL GUARD
            if state.time >= state.config.total_time:
                state.ready_for_time_loop = False
            
            if DEBUG and state.iteration % 10 == 0:
                logger.info(f"Iter {state.iteration}: t={state.time:.4f}s | PPE Delta={max_delta:.2e}")

        if DEBUG:
            logger.info("DEBUG [Main]: Loop exit detected. Finalizing artifacts.")

        return archive_simulation_artifacts(state)

    except Exception as e:
        logger.error(f"FATAL PIPELINE ERROR: {str(e)}")
        raise RuntimeError(f"Solver Pipeline crashed: {str(e)}") from e

def archive_simulation_artifacts(state) -> str:
    """Rule 4: SSoT Archiving. Moves manifest snapshots into a single ZIP."""
    base_dir = Path(".")
    zip_base_name = base_dir / f"navier_stokes_{state.config.case_name}_output"
    source_dir = Path("output")

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