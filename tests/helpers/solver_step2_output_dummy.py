# tests/helpers/solver_step2_output_dummy.py

"""
Archivist Testing: Snapshot-based Test Baseline (Step 2).

Compliance:
- Rule 6: Zero-Redundancy (Independent of Factory logic).
- Rule 7: Atomic Numerical Truth (Fixed data for verification).
- Rule 9: Sentinel Integrity (Mocks must mirror pointer-to-buffer behavior).
"""

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


# Around line 17 in tests/helpers/solver_step2_output_dummy.py
class SimpleCellMock:
    """
    Lightweight Mock Cell that mimics the pointer-behavior of the real Cell class.
    
    Compliance:
    - Rule 0: __slots__ mandatory for memory footprint parity.
    - Rule 9: Hybrid Memory Foundation (Pointer-to-Buffer).
    """
    __slots__ = ['index', 'is_ghost', 'fields_buffer', 'nx_buf', 'ny_buf']

    def __init__(self, index, is_ghost, fields_buffer, nx_buf=6, ny_buf=6):
        self.index = index
        self.is_ghost = is_ghost
        self.fields_buffer = fields_buffer
        self.nx_buf = nx_buf
        self.ny_buf = ny_buf

    def set_field(self, field_idx, value): 
        self.fields_buffer[self.index, field_idx] = value 

    def get_field(self, field_idx):
        return self.fields_buffer[self.index, field_idx]

    # --- Field Accessors (Rule 9 Bridge) ---

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

    @property
    def p(self) -> float: return self.fields_buffer[self.index, FI.P]
    @p.setter
    def p(self, val): self.fields_buffer[self.index, FI.P] = val

    @property
    def mask(self) -> float: return self.fields_buffer[self.index, FI.MASK]
    @mask.setter
    def mask(self, val): self.fields_buffer[self.index, FI.MASK] = val

    @property
    def vx_star(self): return self.get_field(FI.VX_STAR)
    @property
    def vy_star(self): return self.get_field(FI.VY_STAR)
    @property
    def vz_star(self): return self.get_field(FI.VZ_STAR)

    def to_dict(self):
        return {
            "index": self.index,
            "is_ghost": self.is_ghost,
            "fields_buffer": self.fields_buffer.tolist(),
            "nx_buf": self.nx_buf,
            "ny_buf": self.ny_buf
        }

def make_step2_output_dummy(nx: int = 4, ny: int = 4, nz: int = 4):
    """
    Fixed Step 2 Dummy: Complies with Rule 7 (Offset 1 Topology).
    Maps Core [0...nx-1] to Memory Indices [1...nx].
    """
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    physics_params = {
        "dx": 0.25, "dy": 0.25, "dz": 0.25,
        "dt": 0.001, "rho": 1000.0, "mu": 0.001,
        "f_vals": (0.0, 0.0, -9.81)
    }
    
    state.stencil_matrix = []
    # Rule 7.2: 4x4x4 core maps to 6x6x6 memory structure
    nx_buf, ny_buf, nz_buf = nx + 2, ny + 2, nz + 2
    buffer = state.fields.data
    
    def get_idx(i, j, k):
        return i + nx_buf * (j + ny_buf * k)

    # We iterate ONLY over the CORE cells (Memory Indices 1 to nx)
    # This aligns with Rule 7.3: Memory Index 1 is Coordinate 0.
    for k in range(1, nz + 1):
        for j in range(1, ny + 1):
            for i in range(1, nx + 1):
                
                # Center is always a Core cell in this loop
                cell_c = SimpleCellMock(get_idx(i, j, k), False, buffer)
                
                # Neighbors pull from the Unified Foundation
                # If i-1 == 0, it correctly hits the Ghost at Coordinate -1
                cell_im = SimpleCellMock(get_idx(i-1, j, k), (i-1 == 0), buffer)
                cell_ip = SimpleCellMock(get_idx(i+1, j, k), (i+1 == nx+1), buffer)
                
                cell_jm = SimpleCellMock(get_idx(i, j-1, k), (j-1 == 0), buffer)
                cell_jp = SimpleCellMock(get_idx(i, j+1, k), (j+1 == ny+1), buffer)
                
                cell_km = SimpleCellMock(get_idx(i, j, k-1), (k-1 == 0), buffer)
                cell_kp = SimpleCellMock(get_idx(i, j, k+1), (k+1 == nz+1), buffer)
                
                block = StencilBlock(
                    center=cell_c,
                    i_minus=cell_im, i_plus=cell_ip,
                    j_minus=cell_jm, j_plus=cell_jp,
                    k_minus=cell_km, k_plus=cell_kp,
                    **physics_params
                )
                state.stencil_matrix.append(block)
    
    state.ready_for_time_loop = True
    return state