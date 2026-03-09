# src/step2/cell.py

from src.common.base_container import ValidatedContainer


class Cell(ValidatedContainer):
    """
    Transient Data Transfer Object (DTO) for Step 2.
    Inherits runtime contract enforcement from ValidatedContainer.
    Optimized with __slots__ to eliminate dictionary overhead.
    """

    # __slots__ reserves fixed memory locations for these attributes.
    # Note: These names match the internal storage expected by _get_safe/_set_safe.
    __slots__ = ['_x', '_y', '_z', '_vx', '_vy', '_vz', '_p', '_mask', '_is_ghost']

    def __init__(self, x: int = None, y: int = None, z: int = None):
        # Explicitly initialize slots to None to allow the _get_safe 
        # uninitialized-access check to function correctly.
        for slot in self.__slots__:
            super().__setattr__(slot, None)

        # Set coordinates if provided; these use the property setters, 
        # which will perform the validation logic.
        if x is not None: self.x = x
        if y is not None: self.y = y
        if z is not None: self.z = z

    # --- Setters and Getters using _set_safe and _get_safe ---

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

    @property
    def p(self) -> float: return self._get_safe("p")
    @p.setter
    def p(self, value: float): self._set_safe("p", value, (float, int))

    @property
    def mask(self) -> int: return self._get_safe("mask")
    @mask.setter
    def mask(self, value: int): self._set_safe("mask", value, int)

    @property
    def is_ghost(self) -> bool: return self._get_safe("is_ghost")
    @is_ghost.setter
    def is_ghost(self, value: bool): self._set_safe("is_ghost", value, bool)