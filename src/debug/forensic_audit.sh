# 1. Confirm the illegal assignment in Config
cat -n src/common/solver_config.py | sed -n '20,30p'

# 2. Confirm the unused variable in Context
cat -n src/common/simulation_context.py | sed -n '25,35p'

# 3. Verify Slots in Config
grep "__slots__" -A 5 src/common/solver_config.py

# 1. Strip 'dt' from SolverConfig __init__ (Lines 20-30)
# This removes the attempt to set an unslotted attribute.
cat << 'EOF' > src/common/solver_config.py
from dataclasses import dataclass
from src.common.base_container import ValidatedContainer

@dataclass
class SolverConfig(ValidatedContainer):
    """
    Static numerical configuration for the Navier-Stokes solver.
    Rule 4 & 0: ONLY solver-static limits. NO 'dt' here.
    """
    __slots__ = [
        '_ppe_tolerance', '_ppe_atol', '_ppe_max_iter', 
        '_ppe_omega', '_dt_min_limit', '_divergence_threshold'
    ]

    def __init__(self, **kwargs):
        # We explicitly ignore 'dt' if it is passed in the kwargs
        self.dt_min_limit = kwargs.get('dt_min_limit')
        self.ppe_tolerance = kwargs.get('ppe_tolerance')
        self.ppe_atol = kwargs.get('ppe_atol')
        self.ppe_max_iter = kwargs.get('ppe_max_iter')
        self.ppe_omega = kwargs.get('ppe_omega')
        self.divergence_threshold = kwargs.get('divergence_threshold')
        
        # Rule 5: Explicit validation
        for field in [f.strip('_') for f in self.__slots__]:
            if getattr(self, field) is None:
                raise AttributeError(f"CONTRACT VIOLATION: '{field}' missing in JSON.")

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
EOF

# 2. Fix SimulationContext: Remove the unused base_dt assignment and the 'dt=' injection
# This satisfies Ruff and prevents the AttributeError.
sed -i '/base_dt = input_data.simulation_parameters.time_step/d' src/common/simulation_context.py
sed -i 's/config = SolverConfig(dt=base_dt, \*\*config_dict)/config = SolverConfig(\*\*config_dict)/' src/common/simulation_context.py

# 3. Final Lint & Format Check
ruff check src/common/solver_config.py src/common/simulation_context.py --fix
