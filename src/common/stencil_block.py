# src/common/stencil_block.py

from src.common.base_container import ValidatedContainer
from .cell import Cell

class StencilBlock(ValidatedContainer):
    """
    Logical Wiring: Represents the 7-point stencil topology.
    Acts as the graph node connecting neighboring Cells.
    
    Physics parameters are provided via properties (facades) that 
    reference the SolverState, keeping memory usage O(1) per block.
    """

    __slots__ = [
        '_center', '_i_minus', '_i_plus', '_j_minus', '_j_plus', '_k_minus', '_k_plus',
        '_state'
    ]

    def __init__(self, state, center: Cell, i_minus: Cell, i_plus: Cell, 
                 j_minus: Cell, j_plus: Cell, k_minus: Cell, k_plus: Cell):
        
        # Zero-Debt Policy: Initialize all slots to None
        for slot in self.__slots__:
            super().__setattr__(slot, None)
        
        # Assign parent state and topology
        self.state = state
        self.center = center
        self.i_minus = i_minus
        self.i_plus = i_plus
        self.j_minus = j_minus
        self.j_plus = j_plus
        self.k_minus = k_minus
        self.k_plus = k_plus

    # --- Parent State Reference ---
    @property
    def state(self): return self._get_safe("state")
    @state.setter
    def state(self, val): self._set_safe("state", val, object)

    # --- Topological Accessors ---
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

    # --- Physics Facades (Computed on-the-fly) ---
    @property
    def dx(self) -> float: return self.state.grid.dx
    
    @property
    def dy(self) -> float: return self.state.grid.dy
    
    @property
    def dz(self) -> float: return self.state.grid.dz
    
    @property
    def dt(self) -> float: return self.state.simulation_parameters.time_step
    
    @property
    def rho(self) -> float: return self.state.fluid_properties.density
    
    @property
    def mu(self) -> float: return self.state.fluid_properties.viscosity
    
    @property
    def f_vals(self) -> tuple: 
        """Returns the specific slice of the global field buffer for this Cell."""
        return self.center.fields_buffer[self.center.index, :]