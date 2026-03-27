# src/common/solver_config.py

from src.common.base_container import ValidatedContainer

class SolverConfig(ValidatedContainer):
    """
    Static numerical configuration. No dynamic state allowed.
    Compliance: Rule 0 (Memory Hardening), Rule 5 (Deterministic), Rule 8 (Minimalism).
    """
    __slots__ = [
        '_dt_min_limit', '_ppe_tolerance', '_ppe_atol', 
        '_ppe_max_iter', '_ppe_omega', '_divergence_threshold',
        '_ppe_max_retries'
    ]

    def __init__(self, **kwargs):
        """
        Direct mapping from JSON context to validated slots.
        No defaults allowed here to satisfy Rule 5.
        """
        # Explicit assignment triggers the property setters immediately
        self.dt_min_limit = kwargs.get('dt_min_limit')
        self.ppe_tolerance = kwargs.get('ppe_tolerance')
        self.ppe_atol = kwargs.get('ppe_atol')
        self.ppe_max_iter = kwargs.get('ppe_max_iter')
        self.ppe_omega = kwargs.get('ppe_omega')
        self.divergence_threshold = kwargs.get('divergence_threshold')
        self.ppe_max_retries = kwargs.get('ppe_max_retries')

    def __repr__(self) -> str:
        """
        Rule 8: Minimalist string representation for remote forensic audits.
        Excludes sensitive or redundant state.
        """
        return (
            f"SolverConfig("
            f"ppe_tol={self.ppe_tolerance:.1e}, "
            f"ppe_max_iter={self.ppe_max_iter}, "
            f"ppe_omega={self.ppe_omega}, "
            f"div_thresh={self.divergence_threshold}"
            f")"
        )

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

    @property
    def ppe_max_retries(self) -> int: return self._get_safe("ppe_max_retries")
    @ppe_max_retries.setter
    def ppe_max_retries(self, v: int): self._set_safe("ppe_max_retries", v, int)