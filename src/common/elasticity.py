import logging
import numpy as np
from src.common.field_schema import FI

class ElasticManager:
    __slots__ = ['config', 'logger', '_dt', 'dt_floor', '_iteration', '_runs', '_dt_range', '_omega']

    def __init__(self, config, initial_dt: float):
        self.config = config
        self.logger = logging.getLogger("Elasticity")
        
        # Current operating states
        self._dt = initial_dt 
        self.dt_floor = self.config.dt_min_limit
        self._omega = self.config.ppe_omega
        self._iteration = 0
        self._runs = 10
        
        # Linear range from initial_dt down to dt_floor
        self._dt_range = [initial_dt + i * (self.dt_floor - initial_dt) / self._runs for i in range(self._runs + 1)]

    @property
    def dt(self) -> float: 
        return self._dt
    
    @property
    def omega(self) -> float: 
        return self._omega

    @property
    def max_iter(self) -> int:
        # Boost iterations during stabilized retries to help convergence
        return 5000 if self._iteration > 0 else self.config.ppe_max_iter

    def stabilization(self, is_needed: bool) -> None:
        if not is_needed:
            # Success path: Reset to full speed
            self._iteration = 0
            self._dt = self._dt_range[self._iteration]
            return

        # Failure path: Step down the dt range
        if self._iteration >= self._runs:
            # Already at dt_floor and still failing
            raise RuntimeError(
                f"Not found stable run within dt = {self._dt:.2e} and dt_floor = {self.dt_floor:.2e}. "
                "Update config and restart the run."
            )
        
        self._iteration += 1
        self._dt = self._dt_range[self._iteration]
        self.logger.warning(f"Instability detected. Reducing dt to {self._dt:.2e} (Attempt {self._iteration}/{self._runs})")