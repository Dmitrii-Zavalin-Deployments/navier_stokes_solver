# src/common/archive_service.py

import shutil
from pathlib import Path

from src.common.solver_state import SolverState


def archive_simulation_artifacts(state: SolverState) -> str:
    """
    Performs robust archiving using absolute paths to ensure consistency 
    across local development and CI/CD runners.
    """
    # Anchor to the absolute project root: src/common/archive_service.py -> ../..
    project_root = Path(__file__).resolve().parent.parent.parent
    
    source_dir = project_root / "output"
    target_dir = project_root / "data" / "testing-input-output"
    
    # 1. Ensure target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Rename 'output' to 'navier_stokes_output'
    # We use a temporary staging area to ensure atomic moves
    renamed_dir = project_root / "navier_stokes_output"
    if renamed_dir.exists():
        shutil.rmtree(renamed_dir)
    
    # Move contents from 'output' to the staged 'navier_stokes_output'
    shutil.move(str(source_dir), str(renamed_dir))
    
    # 3. Package into an archive
    # Creates 'navier_stokes_output.zip' at project_root
    archive_base = project_root / "navier_stokes_output"
    archive_path = shutil.make_archive(str(archive_base), 'zip', str(renamed_dir))
    
    # 4. Move the resulting archive to the target location
    final_destination = target_dir / "navier_stokes_output.zip"
    if final_destination.exists():
        final_destination.unlink()
    shutil.move(archive_path, str(final_destination))
    
    return str(final_destination)