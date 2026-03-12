# src/main_solver.py

import json
import os
import sys
from pathlib import Path
import jsonschema

from src.common.archive_service import archive_simulation_artifacts
from src.common.solver_input import SolverInput
from src.common.solver_config import SolverConfig
from src.common.simulation_context import SimulationContext
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3
from src.step4.orchestrate_step4 import orchestrate_step4
from src.step5.orchestrate_step5 import orchestrate_step5

# Global Debug Toggle
DEBUG = True

def _load_simulation_context(input_path: str) -> SimulationContext:
    """Assembles physical input and numerical config into a unified context."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file missing at {input_path}")
    
    config_path = Path("config.json")
    if not config_path.exists():
        raise FileNotFoundError("config.json required for solver orchestration.")

    with open(input_path) as f:
        input_data = json.load(f)
    with open(config_path) as f:
        config_data = json.load(f)
        
    return SimulationContext.create(input_data, config_data)

def run_solver(input_path: str):
    """Orchestrates the physics pipeline using the unified SimulationContext."""
    
    # 1. INITIALIZATION & CONTEXT ASSEMBLY
    context = _load_simulation_context(input_path)
    
    # Assembly via Orchestrators
    state = orchestrate_step1(context.input_data)
    state = orchestrate_step2(state)

    # FIREWALL: Contract Validation
    SCHEMA_PATH = Path("schema/solver_input_schema.json")
    try:
        state.validate_against_schema(str(SCHEMA_PATH))
        if DEBUG:
            print("DEBUG [Main]: ✅ State validation passed.")
    except jsonschema.exceptions.ValidationError as e:
        print(f"!!! CONTRACT VIOLATION at {'.'.join([str(p) for p in e.path])}")
        raise

    # 2. ACTIVATE SYSTEM SENTINEL
    state.ready_for_time_loop = True
    if DEBUG:
        print(f"🚀 Starting Simulation: {state.domain.type}")
    
    # 3. MAIN EXECUTION LOOP
    while state.ready_for_time_loop:
        # A. PREDICTOR PASS
        for block in state.stencil_matrix:
            block, _ = orchestrate_step3(block, context=context, is_first_pass=True)
            block = orchestrate_step4(block, state.boundary_conditions, state.grid)
        
        # B. ITERATIVE SOLVER & BOUNDARY PASS
        for _ in range(context.config.ppe_max_iter):
            max_delta = 0.0
            for block in state.stencil_matrix:
                block, delta = orchestrate_step3(block, context=context, is_first_pass=False)
                block = orchestrate_step4(block, state.boundary_conditions, state.grid)
                max_delta = max(max_delta, delta)
            
            if max_delta < context.config.ppe_tolerance:
                if DEBUG:
                    print(f"DEBUG [Main]: PPE Converged: Iter={_ + 1} | Delta={max_delta:.2e} < Tol={context.config.ppe_tolerance:.2e}")
                break
        
        # C. ODOMETER UPDATE
        state.iteration += 1
        state.time += state.sim_params.time_step
        
        # D. ARCHIVING
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
        final_state = run_solver(sys.argv[1])
        zip_path = archive_simulation_artifacts(final_state)
        print(zip_path)
        sys.exit(0)
    except Exception as e:
        print(f"FATAL PIPELINE ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)