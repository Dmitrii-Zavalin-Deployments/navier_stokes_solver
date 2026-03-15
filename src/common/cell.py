# src/common/cell.py

import numpy as np

from src.common.base_container import ValidatedContainer
from src.common.field_schema import FI


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
        return (self.index % (self.nx_buf * self.ny_buf)) % self.nx_buf - 1

    @property
    def j(self) -> int:
        return ((self.index % (self.nx_buf * self.ny_buf)) // self.nx_buf) - 1

    @property
    def k(self) -> int:
        return (self.index // (self.nx_buf * self.ny_buf)) - 1

    # --- Schema-Locked Foundation Access (Rule 9) ---
    def get_field(self, field_id: int) -> float:
        """Access the foundation buffer directly via schema index."""
        return self.fields_buffer[self.index, field_id]

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
    @property
    def vx(self) -> float: return self.fields_buffer[self.index, FI.VX]
    @vx.setter
    def vx(self, value: float): self.fields_buffer[self.index, FI.VX] = value

    @property
    def vy(self) -> float: return self.fields_buffer[self.index, FI.VY]
    @vy.setter
    def vy(self, value: float): self.fields_buffer[self.index, FI.VY] = value

    @property
    def vz(self) -> float: return self.fields_buffer[self.index, FI.VZ]
    @vz.setter
    def vz(self, value: float): self.fields_buffer[self.index, FI.VZ] = value

    @property
    def vx_star(self) -> float: return self.fields_buffer[self.index, FI.VX_STAR]
    @vx_star.setter
    def vx_star(self, value: float): self.fields_buffer[self.index, FI.VX_STAR] = value

    @property
    def vy_star(self) -> float: return self.fields_buffer[self.index, FI.VY_STAR]
    @vy_star.setter
    def vy_star(self, value: float): self.fields_buffer[self.index, FI.VY_STAR] = value

    @property
    def vz_star(self) -> float: return self.fields_buffer[self.index, FI.VZ_STAR]
    @vz_star.setter
    def vz_star(self, value: float): self.fields_buffer[self.index, FI.VZ_STAR] = value

    @property
    def p(self) -> float: return self.fields_buffer[self.index, FI.P]
    @p.setter
    def p(self, value: float): self.fields_buffer[self.index, FI.P] = value

    @property
    def p_next(self) -> float: return self.fields_buffer[self.index, FI.P_NEXT]
    @p_next.setter
    def p_next(self, value: float): self.fields_buffer[self.index, FI.P_NEXT] = value