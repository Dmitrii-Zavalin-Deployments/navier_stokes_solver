# src/step5/io_archivist.py
import h5py
import numpy as np
from pathlib import Path

def save_snapshot(state):
    """
    Exports the physical 3D domain state to HDF5 with spatial verification.
    """
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = output_dir / f"snapshot_{state.iteration:04d}.h5"
    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz
    
    # Helper to extract field attributes into a 3D NumPy array
    def get_field(attr_name):
        return np.array([getattr(b.center, attr_name) for b in state.stencil_matrix]).reshape(nx, ny, nz)

    with h5py.File(filename, 'w') as h5f:
        # Spatial Coordinates (Verification Layer)
        h5f.create_dataset('x', data=get_field('x'))
        h5f.create_dataset('y', data=get_field('y'))
        h5f.create_dataset('z', data=get_field('z'))
        
        # Physical Fields
        h5f.create_dataset('vx', data=get_field('vx'))
        h5f.create_dataset('vy', data=get_field('vy'))
        h5f.create_dataset('vz', data=get_field('vz'))
        h5f.create_dataset('p',  data=get_field('p'))
        
        # Topology
        h5f.create_dataset('mask', data=get_field('mask'))
        
        # Global Metadata
        h5f.attrs['time'] = state.time
        h5f.attrs['iteration'] = state.iteration
        h5f.attrs['dx'] = state.grid.dx
        h5f.attrs['dy'] = state.grid.dy
        h5f.attrs['dz'] = state.grid.dz
    
    state.manifest.saved_snapshots.append(str(filename))