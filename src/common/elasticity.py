import logging

import numpy as np

from src.common.field_schema import FI


class ElasticManager:
    __slots__ = ['config', 'logger', '_dt', '_omega', '_max_iter', 'is_in_panic', 'dt_floor', '_target_dt', '_iteration']

    def __init__(self, config, initial_dt: float):
        self.config = config
        self.logger = logging.getLogger("Elasticity")
        self._dt = initial_dt 
        self._target_dt = initial_dt
        self.dt_floor = self.config.dt_min_limit 
        self._omega = self.config.ppe_omega
        self._max_iter = self.config.ppe_max_iter
        self.is_in_panic = False
        self._iteration = 0

    @property
    def dt(self) -> float: return self._dt
    @property
    def omega(self) -> float: return self._omega
    @property
    def max_iter(self) -> int: return self._max_iter

    def sync_state(self, state) -> bool:
        limit = self.config.divergence_threshold 
        audit_fields = [FI.VX_STAR, FI.VY_STAR, FI.VZ_STAR, FI.P_NEXT]
        data_slice = state.fields.data[:, audit_fields]
        
        # 1. VALIDATE
        is_sane = np.isfinite(data_slice).all() and \
                  data_slice.max() <= limit and \
                  data_slice.min() >= -limit

        if not is_sane:
            # 2. TRIGGER PANIC
            self.is_in_panic = True
            self._iteration = 0 # Reset streak immediately
            self._dt *= 0.5
            
            if self._dt < self.dt_floor:
                raise RuntimeError(f"FATAL: dt ({self._dt:.2e}) dropped below floor {self.dt_floor:.2e}")
            
            self._omega = max(0.5, self._omega - 0.2)
            self._max_iter = 5000
            self.logger.warning(f"PANIC: dt reduced to {self._dt:.2e}. Retrying step...")
            return False

        # 3. COMMIT (Success Path)
        state.fields.data[:, [FI.VX, FI.VY, FI.VZ]] = state.fields.data[:, [FI.VX_STAR, FI.VY_STAR, FI.VZ_STAR]]
        state.fields.data[:, FI.P] = state.fields.data[:, FI.P_NEXT]
        
        self._iteration += 1
        
        # 4. CONSERVATIVE RECOVERY
        # We only start recovery if we have 10 (increased from 5) stable steps
        if self.is_in_panic and self._iteration >= 10:
            # Restore standard PPE performance immediately once stable
            self._max_iter = self.config.ppe_max_iter
            
            if self._dt < self._target_dt:
                self._dt = min(self._target_dt, self._dt * 1.05)
                self._omega = min(self.config.ppe_omega, self._omega + 0.02)
            else:
                self.is_in_panic = False
        return True
