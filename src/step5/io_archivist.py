# src/step5/io_archivist.py

from pathlib import Path
import h5py
from src.common.field_schema import FI

def save_snapshot(state) -> None:
    """
    Exports the physical 3D domain state to HDF5.
    
    Compliance:
    - Rule 4 (SSoT): Accesses grid and state data from authorized sub-containers.
    - Rule 9 (Hybrid Memory): Direct Foundation slicing via FI schema.
    """
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Rule 5: Explicit derivation of state properties.
    filename = output_dir / f"snapshot_{state.iteration:04d}.h5"
    
    # Retrieve dimensions from the authorized Grid container
    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz
    
    # Access the contiguous Foundation buffer (The "Sink")
    data = state.fields_buffer 

    with h5py.File(filename, 'w') as h5f:
        # Physical Fields: Direct, schema-locked slicing (Rule 9)
        h5f.create_dataset('vx', data=data[:, FI.VX].reshape(nx, ny, nz))
        h5f.create_dataset('vy', data=data[:, FI.VY].reshape(nx, ny, nz))
        h5f.create_dataset('vz', data=data[:, FI.VZ].reshape(nx, ny, nz))
        h5f.create_dataset('p',  data=data[:, FI.P].reshape(nx, ny, nz))
        
        # Spatial metadata: Accessing authorized sub-containers (Rule 4)
        h5f.create_dataset('x', data=state.grid.x_mesh)
        h5f.create_dataset('y', data=state.grid.y_mesh)
        h5f.create_dataset('z', data=state.grid.z_mesh)
        h5f.create_dataset('mask', data=state.grid.mask_mesh)
        
        # Global Metadata: Explicit attribution
        h5f.attrs['time'] = state.time
        h5f.attrs['iteration'] = state.iteration
        h5f.attrs['dx'] = state.grid.dx
        h5f.attrs['dy'] = state.grid.dy
        h5f.attrs['dz'] = state.grid.dz
    
    # Update manifest via the state object
    state.manifest.saved_snapshots.append(str(filename))