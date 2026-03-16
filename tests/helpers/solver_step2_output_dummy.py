# tests/helpers/solver_step2_output_dummy.py

"""
Archivist Testing: Snapshot-based Test Baseline (Step 2).

Compliance:
- Rule 6: Zero-Redundancy (Independent of Factory logic).
- Rule 7: Atomic Numerical Truth (Fixed data for verification).
- Rule 9: Sentinel Integrity (Mocks must mirror pointer-to-buffer behavior).
"""

import numpy as np
from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


class SimpleCellMock:
    """
    Lightweight Mock Cell that mimics the pointer-behavior of the real Cell class.
    This ensures that integrity checks on the global fields_buffer reflect 
    correctly through the 'cell' views.
    """
    def __init__(self, index, is_ghost, fields_buffer):
        self.index = index
        self.is_ghost = is_ghost
        self.fields_buffer = fields_buffer

    # Primary Velocity Fields
    @property
    def vx(self) -> float: return self.fields_buffer[self.index, FI.VX]
    @vx.setter
    def vx(self, val): self.fields_buffer[self.index, FI.VX] = val

    @property
    def vy(self) -> float: return self.fields_buffer[self.index, FI.VY]
    @vy.setter
    def vy(self, val): self.fields_buffer[self.index, FI.VY] = val

    @property
    def vz(self) -> float: return self.fields_buffer[self.index, FI.VZ]
    @vz.setter
    def vz(self, val): self.fields_buffer[self.index, FI.VZ] = val

    # Pressure Field (The cause of the 'Memory Swap' error)
    @property
    def p(self) -> float: return self.fields_buffer[self.index, FI.P]
    @p.setter
    def p(self, val): self.fields_buffer[self.index, FI.P] = val

    # Mask Field
    @property
    def mask(self) -> float: return self.fields_buffer[self.index, FI.MASK]
    @mask.setter
    def mask(self, val): self.fields_buffer[self.index, FI.MASK] = val


def make_step2_output_dummy(nx: int = 4, ny: int = 4, nz: int = 4):
    """
    Returns a 'frozen' prototype representing a successful Step 2 completion.
    Wires mock cells to the real monolithic buffer to satisfy integrity validations.
    """
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    physics_params = {
        "dx": 0.25, "dy": 0.25, "dz": 0.25,
        "dt": 0.001, "rho": 1000.0, "mu": 0.001,
        "f_vals": (0.0, 0.0, -9.81)
    }
    
    state.stencil_matrix = []
    nx_buf, ny_buf = nx + 2, ny + 2
    
    # Grab the actual buffer from the state created by Step 1
    buffer = state.fields.data
    
    for k in range(nz + 2):
        for j in range(ny + 2):
            for i in range(nx + 2):
                # Manual index calculation
                index = i + nx_buf * (j + ny_buf * k)
                is_ghost = (i == 0 or i == nx + 1 or 
                            j == 0 or j == ny + 1 or 
                            k == 0 or k == nz + 1)
                
                # Mock cell construction - now passing the buffer reference
                cell = SimpleCellMock(index=index, is_ghost=is_ghost, fields_buffer=buffer)
                
                # Manual StencilBlock construction
                block = StencilBlock(
                    center=cell,
                    i_minus=cell, i_plus=cell,
                    j_minus=cell, j_plus=cell,
                    k_minus=cell, k_plus=cell,
                    **physics_params
                )
                state.stencil_matrix.append(block)
    
    # This trigger now performs the verify_foundation_integrity() call
    state.ready_for_time_loop = True
    
    return state