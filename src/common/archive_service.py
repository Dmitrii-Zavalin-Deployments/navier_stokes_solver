# src/common/archive_service.py

import shutil
import os
from pathlib import Path
from src.common.solver_state import SolverState

def archive_simulation_artifacts(state: SolverState) -> str:
    """
    Context-aware archiving that adapts to both CI/CD runners and local tests.
    Uses Dynamic Lookup of BASE_DIR to anchor the 'data/' folder correctly.
    """
    # 1. Resolve Dynamic Paths via Dynamic Lookup
    # Rule 5: Explicit or Error. We pull the live BASE_DIR from the main_solver
    # to ensure consistency between simulation run and archival.
    import src.main_solver
    current_base = Path(src.main_solver.BASE_DIR)

    # Source: Where the solver just wrote files (Resolved against current env)
    source_dir = Path(state.manifest.output_directory).resolve()
    
    # Target: Always anchored to the current project/test root
    target_dir = current_base / "data" / "testing-input-output"
    
    # Staging: Keep temporary folders in the current working directory to avoid root clutter
    renamed_dir = Path.cwd() / "navier_stokes_output"
    
    # 2. Safety Check (Rule 5: Explicit or Error)
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Archiver Critical Error: Source directory '{source_dir}' not found. "
            f"Expected at: {source_dir}"
        )

    # 3. Ensure Target Infrastructure exists
    target_dir.mkdir(parents=True, exist_ok=True)

    # 4. Atomic Staging (Move 'output' -> 'navier_stokes_output')
    if renamed_dir.exists():
        shutil.rmtree(renamed_dir)
    
    shutil.move(str(source_dir), str(renamed_dir))

    # 5. Package into Archive
    # Note: Using renamed_dir as the base ensures a clean internal ZIP structure
    temp_zip_path = shutil.make_archive(str(renamed_dir), 'zip', str(renamed_dir))

    # 6. Final Placement
    final_destination = target_dir / "navier_stokes_output.zip"
    if final_destination.exists():
        final_destination.unlink()

    shutil.move(temp_zip_path, str(final_destination))

    return str(final_destination)