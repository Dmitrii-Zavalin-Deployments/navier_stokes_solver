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
        self._runs = 10
        
        # Linear range calculation
        self._dt_range = [
            initial_dt + i * (self.dt_floor - initial_dt) / self._runs 
            for i in range(self._runs + 1)
        ]

    @property
    def dt(self) -> float: 
        return self._dt

    def stabilization(self, is_needed: bool) -> None:
        if not is_needed:
            self._iteration = 0
            self._dt = self._dt_range[self._iteration]
            return

        if self._iteration >= self._runs:
            raise RuntimeError(
                f"Not found stable run within dt = {self._dt:.2e} and dt_floor = {self.dt_floor:.2e}. "
                "Update config and restart the run."
            )
        
        self._iteration += 1
        self._dt = self._dt_range[self._iteration]
        self.logger.warning(f"Instability detected. Reducing dt to {self._dt:.2e} (Attempt {self._iteration}/{self._runs})")