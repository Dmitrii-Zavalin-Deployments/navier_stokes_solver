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

    # --- Internal Helper for Type Safety ---
    def _to_scalar(self, value):
        """Collapses numpy arrays or sequences into a single float."""
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 1:
            return float(value[0])
        return float(value)

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
        """Mutate the foundation buffer directly with Sovereign Type Enforcement."""
        self.fields_buffer[self.index, field_id] = self._to_scalar(value)

    # --- Topological Access ---
    @property
    def mask(self) -> int: 
        return int(self.fields_buffer[self.index, FI.MASK])
    
    @mask.setter
    def mask(self, value: int): 
        self.fields_buffer[self.index, FI.MASK] = int(self._to_scalar(value))

    # --- Physical Fields (View into Foundation) ---

    @property
    def vx(self) -> np.ndarray: return self.fields_buffer[self.index:self.index+1, FI.VX]
    @vx.setter
    def vx(self, value: float): self.fields_buffer[self.index, FI.VX] = self._to_scalar(value)

    @property
    def vy(self) -> np.ndarray: return self.fields_buffer[self.index:self.index+1, FI.VY]
    @vy.setter
    def vy(self, value: float): self.fields_buffer[self.index, FI.VY] = self._to_scalar(value)

    @property
    def vz(self) -> np.ndarray: return self.fields_buffer[self.index:self.index+1, FI.VZ]
    @vz.setter
    def vz(self, value: float): self.fields_buffer[self.index, FI.VZ] = self._to_scalar(value)

    @property
    def vx_star(self) -> np.ndarray: return self.fields_buffer[self.index:self.index+1, FI.VX_STAR]
    @vx_star.setter
    def vx_star(self, value: float): self.fields_buffer[self.index, FI.VX_STAR] = self._to_scalar(value)

    @property
    def vy_star(self) -> np.ndarray: return self.fields_buffer[self.index:self.index+1, FI.VY_STAR]
    @vy_star.setter
    def vy_star(self, value: float): self.fields_buffer[self.index, FI.VY_STAR] = self._to_scalar(value)

    @property
    def vz_star(self) -> np.ndarray: return self.fields_buffer[self.index:self.index+1, FI.VZ_STAR]
    @vz_star.setter
    def vz_star(self, value: float): self.fields_buffer[self.index, FI.VZ_STAR] = self._to_scalar(value)

    @property
    def p(self) -> np.ndarray: return self.fields_buffer[self.index:self.index+1, FI.P]
    @p.setter
    def p(self, value: float): self.fields_buffer[self.index, FI.P] = self._to_scalar(value)

    @property
    def p_next(self) -> np.ndarray: return self.fields_buffer[self.index:self.index+1, FI.P_NEXT]
    @p_next.setter
    def p_next(self, value: float): self.fields_buffer[self.index, FI.P_NEXT] = self._to_scalar(value)

    # --- Vector Properties ---

    @property
    def u(self) -> np.ndarray:
        """
        Returns a live vector view of [VX, VY, VZ].
        Used by the Foundation Integrity Sentinel for pre-flight pointer validation.
        """
        return self.fields_buffer[self.index, [FI.VX, FI.VY, FI.VZ]]

    @u.setter
    def u(self, value: np.ndarray):
        # Vector properties expect a sequence of 3, handled by numpy's internal broadcasting
        self.fields_buffer[self.index, [FI.VX, FI.VY, FI.VZ]] = value