# src/common/archive_service.py

import shutil
import os
from pathlib import Path
from src.common.solver_state import SolverState

def archive_simulation_artifacts(state: SolverState) -> str:
    """
    Context-aware archiving that adapts to both CI/CD runners and local tests.
    Derives paths from SolverState to ensure the 'output' folder is found 
    wherever the simulation actually ran.
    """
    # 1. Resolve Dynamic Paths
    # Use state.manifest.output_directory (Rule 9: Single Source of Truth)
    # .resolve() ensures we handle the os.chdir(test_dir) correctly.
    source_dir = Path(state.manifest.output_directory).resolve()
    
    # We place the final archive and staging area in the Current Working Directory
    cwd = Path.cwd()
    target_dir = cwd / "data" / "testing-input-output"
    renamed_dir = cwd / "navier_stokes_output"
    
    # 2. Safety Check (Rule 5: Explicit or Error)
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Archiver Critical Error: Source directory '{source_dir}' not found. "
            f"Check if the solver successfully initialized the output folder."
        )

    # 3. Ensure Target Infrastructure exists
    target_dir.mkdir(parents=True, exist_ok=True)

    # 4. Atomic Staging (Move 'output' -> 'navier_stokes_output')
    if renamed_dir.exists():
        shutil.rmtree(renamed_dir)
    
    shutil.move(str(source_dir), str(renamed_dir))

    # 5. Package into Archive
    # make_archive returns the absolute path to the .zip file created
    archive_base = str(renamed_dir)
    temp_zip_path = shutil.make_archive(archive_base, 'zip', str(renamed_dir))

    # 6. Final Placement
    final_destination = target_dir / "navier_stokes_output.zip"
    if final_destination.exists():
        final_destination.unlink()

    shutil.move(temp_zip_path, str(final_destination))

    return str(final_destination)