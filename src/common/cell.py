# src/common/cell.py

import numpy as np

from src.common.base_container import ValidatedContainer
from src.common.field_schema import FI
from src.common.grid_math import get_coords_from_index


class Cell(ValidatedContainer):
    """
    Lean Topology DTO (Wiring).
    Uses __slots__ to enforce zero-overhead logic-data.
    The Cell acts as a pointer-view into the shared Foundation buffer.
    """
    # Optimized slots: primitive integers only to satisfy memory constraints
    __slots__ = ['index', 'fields_buffer', 'is_ghost', 'nx_buf', 'ny_buf']

    def __init__(self, index: int, fields_buffer: np.ndarray, nx_buf: int, ny_buf: int, is_ghost: bool = False):
        # Explicit initialization to bypass __dict__ creation
        object.__setattr__(self, 'index', index)
        object.__setattr__(self, 'fields_buffer', fields_buffer)
        object.__setattr__(self, 'is_ghost', is_ghost)
        object.__setattr__(self, 'nx_buf', nx_buf)
        object.__setattr__(self, 'ny_buf', ny_buf)

    # --- Coordinate Properties (SSoT compliant derivation) ---
    
    @property
    def i(self) -> int:
        """Returns the logical X-coordinate by reversing the buffer shift."""
        return get_coords_from_index(self.index, self.nx_buf, self.ny_buf)[0] - 1

    @property
    def j(self) -> int:
        """Returns the logical Y-coordinate by reversing the buffer shift."""
        return get_coords_from_index(self.index, self.nx_buf, self.ny_buf)[1] - 1

    @property
    def k(self) -> int:
        """Returns the logical Z-coordinate by reversing the buffer shift."""
        return get_coords_from_index(self.index, self.nx_buf, self.ny_buf)[2] - 1

    # --- Schema-Locked Foundation Access (Rule 9) ---
    def get_field(self, field_id: int) -> np.ndarray:
        """Access the foundation buffer via slice to return an ndarray view."""
        return self.fields_buffer[self.index:self.index+1, field_id]

    def set_field(self, field_id: int, value: float):
        """Mutate the foundation buffer directly via schema index."""
        self.fields_buffer[self.index, field_id] = value

    # --- Topological Access (View into Foundation) ---
    @property
    def mask(self) -> int: 
        return int(self.fields_buffer[self.index, FI.MASK])
    
    @mask.setter
    def mask(self, value: int): 
        self.fields_buffer[self.index, FI.MASK] = value

    # --- Physical Fields (View into Foundation) ---
    # Returning slices [index:index+1] ensures the returned value is a 
    # numpy.ndarray view. This satisfies .size and .shape checks in CI.

    @property
    def vx(self) -> np.ndarray: return self.fields_buffer[self.index:self.index+1, FI.VX]
    @vx.setter
    def vx(self, value: float): self.fields_buffer[self.index, FI.VX] = value

    @property
    def vy(self) -> np.ndarray: return self.fields_buffer[self.index:self.index+1, FI.VY]
    @vy.setter
    def vy(self, value: float): self.fields_buffer[self.index, FI.VY] = value

    @property
    def vz(self) -> np.ndarray: return self.fields_buffer[self.index:self.index+1, FI.VZ]
    @vz.setter
    def vz(self, value: float): self.fields_buffer[self.index, FI.VZ] = value

    @property
    def vx_star(self) -> np.ndarray: return self.fields_buffer[self.index:self.index+1, FI.VX_STAR]
    @vx_star.setter
    def vx_star(self, value: float): self.fields_buffer[self.index, FI.VX_STAR] = value

    @property
    def vy_star(self) -> np.ndarray: return self.fields_buffer[self.index:self.index+1, FI.VY_STAR]
    @vy_star.setter
    def vy_star(self, value: float): self.fields_buffer[self.index, FI.VY_STAR] = value

    @property
    def vz_star(self) -> np.ndarray: return self.fields_buffer[self.index:self.index+1, FI.VZ_STAR]
    @vz_star.setter
    def vz_star(self, value: float): self.fields_buffer[self.index, FI.VZ_STAR] = value

    @property
    def p(self) -> np.ndarray: return self.fields_buffer[self.index:self.index+1, FI.P]
    @p.setter
    def p(self, value: float): self.fields_buffer[self.index, FI.P] = value

    @property
    def p_next(self) -> np.ndarray: return self.fields_buffer[self.index:self.index+1, FI.P_NEXT]
    @p_next.setter
    def p_next(self, value: float): self.fields_buffer[self.index, FI.P_NEXT] = value

    # --- Vector Properties (Rule 9 Sentinel Compliance) ---

    @property
    def u(self) -> np.ndarray:
        """
        Returns a live vector view of [VX, VY, VZ].
        Used by the Foundation Integrity Sentinel for pre-flight pointer validation.
        """
        return self.fields_buffer[self.index, [FI.VX, FI.VY, FI.VZ]]

    @u.setter
    def u(self, value: np.ndarray):
        """
        Directly mutates the velocity components in the foundation buffer.
        Expects a vector of length 3.
        """
        self.fields_buffer[self.index, [FI.VX, FI.VY, FI.VZ]] = value