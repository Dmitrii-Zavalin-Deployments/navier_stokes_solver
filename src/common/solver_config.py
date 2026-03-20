# src/common/solver_config.py

from dataclasses import dataclass

from src.common.base_container import ValidatedContainer


@dataclass
class SolverConfig(ValidatedContainer):
    """
    Static numerical configuration for the Navier-Stokes solver.
    Uses VerifiedContainer accessors to enforce deterministic initialization.
    """
    # Rule 4: Explicit slots for memory efficiency and structural rigor.
    # Note: _dt is included as the reference target for Elasticity.
    __slots__ = ['_ppe_tolerance', '_ppe_atol', '_ppe_max_iter', '_ppe_omega', '_dt', '_divergence_threshold']

    def __init__(self, **kwargs):
        """
        Constructor implements Deterministic Initialization via _set_safe.
        This triggers the property setters and validation logic, ensuring
        that the object is valid from the moment of instantiation.
        """
        # Assign directly using the validated setters to ensure 
        # type-checking and value-range constraints are enforced.
        self.dt = kwargs.get('dt')
        self.ppe_tolerance = kwargs.get('ppe_tolerance')
        self.ppe_atol = kwargs.get('ppe_atol')
        self.ppe_max_iter = kwargs.get('ppe_max_iter')
        self.ppe_omega = kwargs.get('ppe_omega')
        self.divergence_threshold = kwargs.get('divergence_threshold')
        
        # Post-initialization check: Ensure no fields were left as None
        # Rule 5: Explicit or Error. No fallbacks/defaults allowed here.
        required_fields = ['dt', 'ppe_tolerance', 'ppe_atol', 'ppe_max_iter', 'ppe_omega', 'divergence_threshold']
        for field in required_fields:
            if getattr(self, field) is None:
                raise AttributeError(f"CONTRACT VIOLATION: '{field}' must be explicitly defined in config.")

    @property
    def dt(self) -> float:
        return self._get_safe("dt")

    @dt.setter
    def dt(self, v: float):
        if v is not None and v <= 0:
            raise ValueError(f"dt must be > 0, got {v}")
        self._set_safe("dt", v, float)

    @property
    def ppe_tolerance(self) -> float: 
        return self._get_safe("ppe_tolerance")

    @ppe_tolerance.setter
    def ppe_tolerance(self, v: float):
        if v is not None and v <= 0: 
            raise ValueError(f"ppe_tolerance must be > 0, got {v}")
        self._set_safe("ppe_tolerance", v, float)

    @property
    def ppe_atol(self) -> float: 
        return self._get_safe("ppe_atol")

    @ppe_atol.setter
    def ppe_atol(self, v: float):
        if v is not None and v < 0: 
            raise ValueError(f"ppe_atol must be >= 0, got {v}")
        self._set_safe("ppe_atol", v, float)

    @property
    def ppe_max_iter(self) -> int: 
        return self._get_safe("ppe_max_iter")

    @ppe_max_iter.setter
    def ppe_max_iter(self, v: int):
        if v is not None and v < 1: 
            raise ValueError(f"ppe_max_iter must be >= 1, got {v}")
        self._set_safe("ppe_max_iter", v, int)

    @property
    def ppe_omega(self) -> float: 
        return self._get_safe("ppe_omega")

    @ppe_omega.setter
    def ppe_omega(self, v: float):
        if v is not None and not (0 < v < 2): 
            raise ValueError(f"ppe_omega must be in (0, 2), got {v}")
        self._set_safe("ppe_omega", v, float)

    @property
    def divergence_threshold(self) -> float:
        return self._get_safe("divergence_threshold")

    @divergence_threshold.setter
    def divergence_threshold(self, v: float):
        if v is not None and v <= 0:
            raise ValueError(f"divergence_threshold must be > 0, got {v}")
        self._set_safe("divergence_threshold", v, float)
