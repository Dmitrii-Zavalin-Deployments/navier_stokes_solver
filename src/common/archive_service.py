import shutil
from pathlib import Path
from src.common.solver_state import SolverState

def archive_simulation_artifacts(state: SolverState) -> str:
    """
    Simplest form: Rename 'output' to 'navier_stokes_output', 
    archive it, and move to 'data/testing-input-output'.
    """
    base_dir = Path(".")
    source_dir = base_dir / "output"
    target_dir = base_dir / "data" / "testing-input-output"
    
    # 1. Ensure target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Rename 'output' to 'navier_stokes_output'
    # Note: If 'navier_stokes_output' already exists, handle it
    renamed_dir = base_dir / "navier_stokes_output"
    if renamed_dir.exists():
        shutil.rmtree(renamed_dir)
    shutil.move(str(source_dir), str(renamed_dir))
    
    # 3. Package into an archive
    # Creates 'navier_stokes_output.zip' in the current directory
    archive_path = shutil.make_archive("navier_stokes_output", 'zip', str(renamed_dir))
    
    # 4. Move the resulting archive to the target location
    final_destination = target_dir / "navier_stokes_output.zip"
    if final_destination.exists():
        final_destination.unlink()
    shutil.move(archive_path, str(final_destination))
    
    return str(final_destination)