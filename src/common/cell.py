# src/common/cell.py

from src.common.base_container import ValidatedContainer


class Cell(ValidatedContainer):
    """
    Lean Data Transfer Object (DTO) for the Projection Method.
    Inherits runtime contract enforcement from ValidatedContainer.
    Optimized with __slots__ to eliminate dictionary overhead.
    """

    __slots__ = [
        '_x', '_y', '_z', 
        '_vx', '_vy', '_vz', 
        '_vx_star', '_vy_star', '_vz_star', 
        '_p', '_p_next', 
        '_mask', '_is_ghost'
    ]

    def __init__(self, x: int = None, y: int = None, z: int = None):
        for slot in self.__slots__:
            super().__setattr__(slot, None)

        if x is not None: self.x = x
        if y is not None: self.y = y
        if z is not None: self.z = z

    # --- Standard Properties ---
    @property
    def x(self) -> int: return self._get_safe("x")
    @x.setter
    def x(self, value: int): self._set_safe("x", value, int)

    @property
    def y(self) -> int: return self._get_safe("y")
    @y.setter
    def y(self, value: int): self._set_safe("y", value, int)

    @property
    def z(self) -> int: return self._get_safe("z")
    @z.setter
    def z(self, value: int): self._set_safe("z", value, int)

    # --- Velocity Fields (n) ---
    @property
    def vx(self) -> float: return self._get_safe("vx")
    @vx.setter
    def vx(self, value: float): self._set_safe("vx", value, (float, int))

    @property
    def vy(self) -> float: return self._get_safe("vy")
    @vy.setter
    def vy(self, value: float): self._set_safe("vy", value, (float, int))

    @property
    def vz(self) -> float: return self._get_safe("vz")
    @vz.setter
    def vz(self, value: float): self._set_safe("vz", value, (float, int))

    # --- Intermediate Velocity Fields (star) ---
    @property
    def vx_star(self) -> float: return self._get_safe("vx_star")
    @vx_star.setter
    def vx_star(self, value: float): self._set_safe("vx_star", value, (float, int))

    @property
    def vy_star(self) -> float: return self._get_safe("vy_star")
    @vy_star.setter
    def vy_star(self, value: float): self._set_safe("vy_star", value, (float, int))

    @property
    def vz_star(self) -> float: return self._get_safe("vz_star")
    @vz_star.setter
    def vz_star(self, value: float): self._set_safe("vz_star", value, (float, int))

    # --- Pressure Fields ---
    @property
    def p(self) -> float: return self._get_safe("p")
    @p.setter
    def p(self, value: float): self._set_safe("p", value, (float, int))

    @property
    def p_next(self) -> float: return self._get_safe("p_next")
    @p_next.setter
    def p_next(self, value: float): self._set_safe("p_next", value, (float, int))

    # --- Topology ---
    @property
    def mask(self) -> int: return self._get_safe("mask")
    @mask.setter
    def mask(self, value: int): self._set_safe("mask", value, int)

    @property
    def is_ghost(self) -> bool: return self._get_safe("is_ghost")
    @is_ghost.setter
    def is_ghost(self, value: bool): self._set_safe("is_ghost", value, bool)