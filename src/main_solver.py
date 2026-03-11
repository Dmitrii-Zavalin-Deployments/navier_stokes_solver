# src/main_solver.py

import json
import os
import sys
from pathlib import Path
import jsonschema

from src.solver_input import SolverInput
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3
from src.step4.orchestrate_step4 import orchestrate_step4
from src.step5.orchestrate_step5 import orchestrate_step5
from src.common.archive_service import archive_simulation_artifacts

# Global Debug Toggle
DEBUG = True

def _load_solver_config() -> dict:
    config_path = Path("config.json")
    if not config_path.exists():
        raise FileNotFoundError("config.json required for solver orchestration.")
    with open(config_path) as f:
        return json.load(f)["solver_settings"]

def run_solver(input_path: str):
    """Orchestrates the physics pipeline: Initialization through Time-Loop."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file missing at {input_path}")

    # Load configuration
    cfg = _load_solver_config()
    # Rule 5: No hardcoded defaults, pull explicitly from config
    max_iter = cfg["ppe_max_iter"]
    tol = cfg["ppe_tolerance"]
    omega = cfg["ppe_omega"]

    SCHEMA_PATH = Path("schema/solver_input_schema.json")

    with open(input_path) as f:
        raw_data = json.load(f)
    
    # 1. INITIALIZATION & STATE ASSEMBLY
    input_container = SolverInput.from_dict(raw_data)
    state = orchestrate_step1(input_container)
    state = orchestrate_step2(state)

    # FIREWALL: Contract Validation
    try:
        # Assuming Validation is a method on the Container per Architecture
        # Otherwise, move to a separate Validator utility to keep state clean.
        state.validate_against_schema(str(SCHEMA_PATH))
        if DEBUG:
            print("DEBUG [Main]: ✅ State validation passed.")
    except jsonschema.exceptions.ValidationError as e:
        print(f"!!! CONTRACT VIOLATION at {'.'.join([str(p) for p in e.path])}")
        raise

    # 2. ACTIVATE SYSTEM SENTINEL
    # Setting this to True triggers the POST and Physical Sanity checks 
    # via the SolverState property setter (Rule 9).
    state.ready_for_time_loop = True

    if DEBUG:
        print(f"🚀 Starting Simulation: {state.domain.type}")
    
    # 3. MAIN EXECUTION LOOP
    while state.ready_for_time_loop:
        # A. PREDICTOR PASS
        for block in state.stencil_matrix:
            block, _ = orchestrate_step3(block, omega=omega, is_first_pass=True)
            block = orchestrate_step4(block, state.boundary_conditions, state.grid)
        
        # B. ITERATIVE SOLVER & BOUNDARY PASS
        for _ in range(max_iter):
            max_delta = 0.0
            for block in state.stencil_matrix:
                block, delta = orchestrate_step3(block, omega=omega, is_first_pass=False)
                block = orchestrate_step4(block, state.boundary_conditions, state.grid)
                max_delta = max(max_delta, delta)
            
            if max_delta < tol:
                if DEBUG:
                    print(f"DEBUG [Main]: PPE Converged: Iter={_ + 1} | Delta={max_delta:.2e} < Tol={tol:.2e}")
                break
        
        # C. ODOMETER UPDATE
        state.iteration += 1
        state.time += state.sim_params.time_step
        
        # D. ARCHIVING (Orchestrate step5 now purely handles logic, not packaging)
        state = orchestrate_step5(state)
        
        # E. TEMPORAL GUARD
        if state.time >= state.sim_params.total_time:
            state.ready_for_time_loop = False

    if DEBUG:
        print("DEBUG [Main]: Loop exit detected.")
        
    return state

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Input file path required.")
        sys.exit(1)
    
    try:
        # Step 1: Run the simulation physics
        final_state = run_solver(sys.argv[1])
        
        # Step 2: Use Archive Service (SSoT) for result packaging
        zip_path = archive_simulation_artifacts(final_state)
        
        print(zip_path)
        sys.exit(0)
        
    except Exception as e:
        print(f"FATAL PIPELINE ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)