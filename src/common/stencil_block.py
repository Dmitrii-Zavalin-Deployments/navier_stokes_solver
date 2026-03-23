# src/common/stencil_block.py

from src.common.base_container import ValidatedContainer

from .cell import Cell


class StencilBlock(ValidatedContainer):
    """
    Logical Wiring: Represents the 7-point stencil topology.
    Acts as the graph node connecting neighboring Cells.
    
    Physics parameters are injected during assembly to maintain O(1) access
    within the inner simulation loops.
    """

    __slots__ = [
        '_center', '_i_minus', '_i_plus', '_j_minus', '_j_plus', '_k_minus', '_k_plus',
        '_dx', '_dy', '_dz', '_dt', '_rho', '_mu', '_f_vals', '_id'
    ]

    def __init__(self, center: Cell, i_minus: Cell, i_plus: Cell, 
                 j_minus: Cell, j_plus: Cell, k_minus: Cell, k_plus: Cell,
                 dx: float, dy: float, dz: float, dt: float, 
                 rho: float, mu: float, f_vals: tuple):
        
        # Explicit initialization (Rule 5: Deterministic Initialization)
        object.__setattr__(self, '_center', center)
        object.__setattr__(self, '_i_minus', i_minus)
        object.__setattr__(self, '_i_plus', i_plus)
        object.__setattr__(self, '_j_minus', j_minus)
        object.__setattr__(self, '_j_plus', j_plus)
        object.__setattr__(self, '_k_minus', k_minus)
        object.__setattr__(self, '_k_plus', k_plus)
        
        # Physics attributes (Cached at assembly for performance)
        object.__setattr__(self, '_dx', float(dx))
        object.__setattr__(self, '_dy', float(dy))
        object.__setattr__(self, '_dz', float(dz))
        object.__setattr__(self, '_dt', float(dt))
        object.__setattr__(self, '_rho', float(rho))
        object.__setattr__(self, '_mu', float(mu))
        object.__setattr__(self, '_f_vals', tuple(f_vals))
        
        # Unique ID for tracking in logs
        object.__setattr__(self, '_id', f"Block_{id(self) % 1000}")

    # --- Topological Accessors ---
    @property
    def id(self) -> str: return self._id

    @property
    def center(self) -> Cell: return self._center
    
    @property
    def i_minus(self) -> Cell: return self._i_minus
    
    @property
    def i_plus(self) -> Cell: return self._i_plus
    
    @property
    def j_minus(self) -> Cell: return self._j_minus
    
    @property
    def j_plus(self) -> Cell: return self._j_plus
    
    @property
    def k_minus(self) -> Cell: return self._k_minus
    
    @property
    def k_plus(self) -> Cell: return self._k_plus

    # --- Physics Facades (High-speed cached access) ---
    @property
    def dx(self) -> float: return self._dx
    
    @property
    def dy(self) -> float: return self._dy
    
    @property
    def dz(self) -> float: return self._dz
    
    @property
    def dt(self) -> float: 
        return self._dt

    @dt.setter
    def dt(self, value: float):
        """
        Rule 4 Sync Gate: Allows ElasticityManager to update the block's time-step.
        """
        if value <= 0:
            raise ValueError(f"Numerical Instability: dt must be positive, got {value}")
        
        # --- [STRATEGIC DIAGNOSTIC LOG] ---
        # Verifies that Elasticity is actually pushing the new dt down to the compute kernels.
        print(f"DEBUG [StencilBlock]: {self._id} Syncing dt -> {value:.4e}")
        
        object.__setattr__(self, '_dt', float(value))
    
    @property
    def rho(self) -> float: return self._rho
    
    @property
    def mu(self) -> float: return self._mu
    
    @property
    def f_vals(self) -> tuple: return self._f_vals

    def __repr__(self):
        return f"<{self._id} dt={self._dt:.2e}>"