# src/common/elasticity.py

import logging
import numpy as np
from src.common.field_schema import FI

class ElasticManager:
    """
    SSoT for Numerical Stability. 
    Acts as the dynamic authority for dt, omega, and max_iter.
    """
    def __init__(self, config):
        self.config = config # SimulationConfig object
        self.logger = logging.getLogger("Elasticity")
        
        # Internal Elastic State
        self._dt = self.config.dt
        self._omega = self.config.ppe_omega
        self._max_iter = self.config.ppe_max_iter
        
        self.is_in_panic = False
        self.stable_streak = 0
        self.cooldown_limit = 5

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def omega(self) -> float:
        return self._omega

    @property
    def max_iter(self) -> int:
        return self._max_iter

    def validate_and_commit(self, state) -> bool:
        """Audits trial fields. Returns True if math is sane and committed."""
        audit_fields = [FI.VX_STAR, FI.VY_STAR, FI.VZ_STAR, FI.P_NEXT]
        if not np.isfinite(state.fields_buffer[:, audit_fields]).all():
            return False

        # COMMIT: Star -> Foundation
        state.fields_buffer[:, [FI.VX, FI.VY, FI.VZ]] = state.fields_buffer[:, [FI.VX_STAR, FI.VY_STAR, FI.VZ_STAR]]
        state.fields_buffer[:, FI.P] = state.fields_buffer[:, FI.P_NEXT]
        return True

    def apply_panic_mode(self):
        self.is_in_panic = True
        self.stable_streak = 0
        self._dt *= 0.5
        self._omega = max(0.5, self._omega - 0.2)
        self._max_iter = 5000
        self.logger.warning(f"PANIC: dt reduced to {self._dt:.2e}")

    def gradual_recovery(self):
        if not self.is_in_panic: return
        self.stable_streak += 1
        if self.stable_streak >= self.cooldown_limit:
            self._dt = min(self.config.dt, self._dt * 1.1)
            self._omega = min(self.config.ppe_omega, self._omega + 0.05)
            if self._dt == self.config.dt and self._omega == self.config.ppe_omega:
                self.is_in_panic = False
                self._max_iter = self.config.ppe_max_iter