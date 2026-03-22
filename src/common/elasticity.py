# src/common/elasticity.py

import logging


class ElasticManager:
    __slots__ = ['config', 'logger', '_dt', 'dt_floor', '_iteration', '_runs', '_dt_range']

    def __init__(self, config, initial_dt: float):
        self.config = config
        self.logger = logging.getLogger("Elasticity")
        
        self._dt = initial_dt 
        self.dt_floor = self.config.dt_min_limit
        self._iteration = 0
        
        # Rule 5: No hardcoded defaults. Pulled from SSoT (Config).
        self._runs = self.config.ppe_max_retries
        
        # Linear range from initial_dt down to dt_floor
        self._dt_range = [
            initial_dt + i * (self.dt_floor - initial_dt) / self._runs 
            for i in range(self._runs + 1)
        ]

    @property
    def dt(self) -> float: 
        return self._dt

    def stabilization(self, is_needed: bool) -> None:
        if not is_needed:
            # Success: Reset to full speed
            self._iteration = 0
            self._dt = self._dt_range[self._iteration]
            return

        if self._iteration >= self._runs:
            raise RuntimeError(
                f"Unstable: reached dt_floor = {self.dt_floor:.2e}. "
                f"Exhausted {self._runs} retries. Update the run configs and restart the solver"
            )
        
        # Advance to the next (smaller) time step in the pre-calculated range
        self._iteration += 1
        self._dt = self._dt_range[self._iteration]
        
        # Rule 8: Singular logging call for audit trail visibility
        self.logger.warning(
            f"Instability. Reducing dt to {self._dt:.2e} "
            f"({self._iteration}/{self._runs})"
        )