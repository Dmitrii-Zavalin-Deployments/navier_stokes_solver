# src/common/solver_config.py

from dataclasses import dataclass
from typing import Any

from src.common.base_container import ValidatedContainer


@dataclass
class SolverConfig(ValidatedContainer):
    """
    Static numerical configuration for the Navier-Stokes solver.
    Separated from physical inputs to maintain schema integrity.
    
    Phase C Compliance:
    - Rule 5: Deterministic Initialization. All parameters must be 
      explicitly provided or strictly validated at runtime.
    """
    __slots__ = ['_ppe_tolerance', '_ppe_atol', '_ppe_max_iter', '_ppe_omega']

    def __init__(self, **kwargs):
        """
        Constructor implements Deterministic Initialization.
        Missing keys trigger an AttributeError via the property setters,
        enforcing the 'Explicit or Error' rule.
        """
        # We invoke setters directly to trigger the validation logic 
        # defined in the properties below.
        self.ppe_tolerance = kwargs.get('ppe_tolerance')
        self.ppe_atol = kwargs.get('ppe_atol')
        self.ppe_max_iter = kwargs.get('ppe_max_iter')
        self.ppe_omega = kwargs.get('ppe_omega')

    @property
    def ppe_tolerance(self) -> float: 
        return self._get_safe("ppe_tolerance")

    @ppe_tolerance.setter
    def ppe_tolerance(self, v: float):
        if v is None: raise AttributeError("Coding Error: 'ppe_tolerance' must be explicitly defined.")
        if v <= 0: 
            raise ValueError(f"ppe_tolerance must be > 0, got {v}")
        self._set_safe("ppe_tolerance", v, float)

    @property
    def ppe_atol(self) -> float: 
        return self._get_safe("ppe_atol")

    @ppe_atol.setter
    def ppe_atol(self, v: float):
        if v is None: raise AttributeError("Coding Error: 'ppe_atol' must be explicitly defined.")
        if v < 0: 
            raise ValueError(f"ppe_atol must be >= 0, got {v}")
        self._set_safe("ppe_atol", v, float)

    @property
    def ppe_max_iter(self) -> int: 
        return self._get_safe("ppe_max_iter")

    @ppe_max_iter.setter
    def ppe_max_iter(self, v: int):
        if v is None: raise AttributeError("Coding Error: 'ppe_max_iter' must be explicitly defined.")
        if v < 1: 
            raise ValueError(f"ppe_max_iter must be >= 1, got {v}")
        self._set_safe("ppe_max_iter", v, int)

    @property
    def ppe_omega(self) -> float: 
        return self._get_safe("ppe_omega")

    @ppe_omega.setter
    def ppe_omega(self, v: float):
        if v is None: raise AttributeError("Coding Error: 'ppe_omega' must be explicitly defined.")
        # Successive Over-Relaxation (SOR) range verification
        if not (0 < v < 2): 
            raise ValueError(f"ppe_omega must be in (0, 2), got {v}")
        self._set_safe("ppe_omega", v, float)