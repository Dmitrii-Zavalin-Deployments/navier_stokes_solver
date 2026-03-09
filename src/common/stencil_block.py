# src/step2/stencil_block.py

from src.common.base_container import ValidatedContainer

from .cell import Cell


class StencilBlock(ValidatedContainer):
    """
    Logical container representing a 7-point stencil + physical context.
    Inherits runtime contract enforcement from ValidatedContainer.
    Optimized with __slots__ to eliminate dictionary overhead.
    """

    __slots__ = [
        '_center', '_i_minus', '_i_plus', '_j_minus', '_j_plus', '_k_minus', '_k_plus',
        '_dx', '_dy', '_dz', '_dt', '_rho', '_mu', '_f_vals'
    ]

    def __init__(self, center: Cell, i_minus: Cell, i_plus: Cell, 
                 j_minus: Cell, j_plus: Cell, k_minus: Cell, k_plus: Cell,
                 dx: float, dy: float, dz: float, dt: float, 
                 rho: float, mu: float, f_vals: tuple):
        
        # Initialize all slots to None to allow _get_safe to detect uninitialized states.
        for slot in self.__slots__:
            super().__setattr__(slot, None)
        
        # Assign values through the validated setters to ensure contract enforcement
        self.center = center
        self.i_minus = i_minus
        self.i_plus = i_plus
        self.j_minus = j_minus
        self.j_plus = j_plus
        self.k_minus = k_minus
        self.k_plus = k_plus
        
        self.dx, self.dy, self.dz = dx, dy, dz
        self.dt = dt
        self.rho = rho
        self.mu = mu
        self.f_vals = f_vals

    # --- Cell Accessors ---
    @property
    def center(self) -> Cell: return self._get_safe("center")
    @center.setter
    def center(self, val: Cell): self._set_safe("center", val, Cell)

    @property
    def i_minus(self) -> Cell: return self._get_safe("i_minus")
    @i_minus.setter
    def i_minus(self, val: Cell): self._set_safe("i_minus", val, Cell)

    @property
    def i_plus(self) -> Cell: return self._get_safe("i_plus")
    @i_plus.setter
    def i_plus(self, val: Cell): self._set_safe("i_plus", val, Cell)

    @property
    def j_minus(self) -> Cell: return self._get_safe("j_minus")
    @j_minus.setter
    def j_minus(self, val: Cell): self._set_safe("j_minus", val, Cell)

    @property
    def j_plus(self) -> Cell: return self._get_safe("j_plus")
    @j_plus.setter
    def j_plus(self, val: Cell): self._set_safe("j_plus", val, Cell)

    @property
    def k_minus(self) -> Cell: return self._get_safe("k_minus")
    @k_minus.setter
    def k_minus(self, val: Cell): self._set_safe("k_minus", val, Cell)

    @property
    def k_plus(self) -> Cell: return self._get_safe("k_plus")
    @k_plus.setter
    def k_plus(self, val: Cell): self._set_safe("k_plus", val, Cell)

    # --- Physics Parameters Accessors ---
    @property
    def dx(self) -> float: return self._get_safe("dx")
    @dx.setter
    def dx(self, val: float): self._set_safe("dx", val, float)

    @property
    def dy(self) -> float: return self._get_safe("dy")
    @dy.setter
    def dy(self, val: float): self._set_safe("dy", val, float)

    @property
    def dz(self) -> float: return self._get_safe("dz")
    @dz.setter
    def dz(self, val: float): self._set_safe("dz", val, float)

    @property
    def dt(self) -> float: return self._get_safe("dt")
    @dt.setter
    def dt(self, val: float): self._set_safe("dt", val, float)

    @property
    def rho(self) -> float: return self._get_safe("rho")
    @rho.setter
    def rho(self, val: float): self._set_safe("rho", val, float)

    @property
    def mu(self) -> float: return self._get_safe("mu")
    @mu.setter
    def mu(self, val: float): self._set_safe("mu", val, float)

    @property
    def f_vals(self) -> tuple: return self._get_safe("f_vals")
    @f_vals.setter
    def f_vals(self, val: tuple): self._set_safe("f_vals", val, tuple)