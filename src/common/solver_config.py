from dataclasses import dataclass

from src.common.base_container import ValidatedContainer


@dataclass
class SolverConfig(ValidatedContainer):
    """
    Static numerical configuration for the Navier-Stokes solver.
    Rule 4 & 0: ONLY solver-static limits. NO 'dt'.
    """
    __slots__ = [
        '_ppe_tolerance', '_ppe_atol', '_ppe_max_iter', 
        '_ppe_omega', '_dt_min_limit', '_divergence_threshold'
    ]

    def __init__(self, **kwargs):
        # Rule 5: Explicit initialization. We IGNORE 'dt' if passed.
        self.dt_min_limit = kwargs.get('dt_min_limit')
        self.ppe_tolerance = kwargs.get('ppe_tolerance')
        self.ppe_atol = kwargs.get('ppe_atol')
        self.ppe_max_iter = kwargs.get('ppe_max_iter')
        self.ppe_omega = kwargs.get('ppe_omega')
        self.divergence_threshold = kwargs.get('divergence_threshold')
        
        required_fields = [
            'dt_min_limit', 'ppe_tolerance', 'ppe_atol', 
            'ppe_max_iter', 'ppe_omega', 'divergence_threshold'
        ]
        for field in required_fields:
            if getattr(self, field) is None:
                raise AttributeError(f"CONTRACT VIOLATION: '{field}' must be in JSON.")

    @property
    def dt_min_limit(self) -> float: return self._get_safe("dt_min_limit")
    @dt_min_limit.setter
    def dt_min_limit(self, v: float): self._set_safe("dt_min_limit", v, float)

    @property
    def ppe_tolerance(self) -> float: return self._get_safe("ppe_tolerance")
    @ppe_tolerance.setter
    def ppe_tolerance(self, v: float): self._set_safe("ppe_tolerance", v, float)

    @property
    def ppe_atol(self) -> float: return self._get_safe("ppe_atol")
    @ppe_atol.setter
    def ppe_atol(self, v: float): self._set_safe("ppe_atol", v, float)

    @property
    def ppe_max_iter(self) -> int: return self._get_safe("ppe_max_iter")
    @ppe_max_iter.setter
    def ppe_max_iter(self, v: int): self._set_safe("ppe_max_iter", v, int)

    @property
    def ppe_omega(self) -> float: return self._get_safe("ppe_omega")
    @ppe_omega.setter
    def ppe_omega(self, v: float): self._set_safe("ppe_omega", v, float)

    @property
    def divergence_threshold(self) -> float: return self._get_safe("divergence_threshold")
    @divergence_threshold.setter
    def divergence_threshold(self, v: float): self._set_safe("divergence_threshold", v, float)
