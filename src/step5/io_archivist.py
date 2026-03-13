# src/step5/io_archivist.py

from pathlib import Path

import h5py
import numpy as np

from src.common.field_schema import FI


def save_snapshot(state) -> None:
    """
    Exports the physical 3D domain state to HDF5.
    
    Compliance:
    - Rule 4 (SSoT): Accesses grid and state data from authorized sub-containers.
    - Rule 8 (Law of Singular Access): Coordinates computed locally to avoid 'God Object' properties in GridManager.
    - Rule 9 (Hybrid Memory): Direct Foundation slicing via FI schema.
    """
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Rule 5: Explicit derivation of state properties.
    filename = output_dir / f"snapshot_{state.iteration:04d}.h5"
    
    # Retrieve dimensions from the authorized Grid container
    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz
    
    # Compute coordinate meshes on-the-fly (Rule 8 compliance)
    x = np.linspace(state.grid.x_min, state.grid.x_max, nx)
    y = np.linspace(state.grid.y_min, state.grid.y_max, ny)
    z = np.linspace(state.grid.z_min, state.grid.z_max, nz)
    
    # Access the contiguous Foundation buffer (The "Sink") 
    # via the authorized FieldManager (Rule 9 compliance)
    data = state.fields.data 

    with h5py.File(filename, 'w') as h5f:
        # Physical Fields: Direct, schema-locked slicing (Rule 9)
        h5f.create_dataset('vx', data=data[:, FI.VX].reshape(nx, ny, nz))
        h5f.create_dataset('vy', data=data[:, FI.VY].reshape(nx, ny, nz))
        h5f.create_dataset('vz', data=data[:, FI.VZ].reshape(nx, ny, nz))
        h5f.create_dataset('p',  data=data[:, FI.P].reshape(nx, ny, nz))
        
        # Spatial metadata: Using computed arrays
        h5f.create_dataset('x', data=x)
        h5f.create_dataset('y', data=y)
        h5f.create_dataset('z', data=z)
        
        # Grid mask retrieved from MaskManager
        h5f.create_dataset('mask', data=state.masks.mask)
        
        # Global Metadata: Explicit attribution
        h5f.attrs['time'] = state.time
        h5f.attrs['iteration'] = state.iteration
        h5f.attrs['dx'] = (state.grid.x_max - state.grid.x_min) / nx
        h5f.attrs['dy'] = (state.grid.y_max - state.grid.y_min) / ny
        h5f.attrs['dz'] = (state.grid.z_max - state.grid.z_min) / nz
    
    # Update manifest via the state object
    state.manifest.saved_snapshots.append(str(filename))