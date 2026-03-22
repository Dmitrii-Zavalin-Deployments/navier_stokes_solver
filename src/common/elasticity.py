# src/common/elasticity.py

import logging

from src.common.field_schema import FI
from src.common.solver_state import SolverState


class ElasticManager:
    __slots__ = ['config', 'logger', '_dt', 'dt_floor', '_iteration', '_runs', '_dt_range']

    def __init__(self, config, initial_dt: float):
        self.config = config
        self.logger = logging.getLogger("Solver.Main.Elasticity")
        
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

    def validate_and_commit(self, state: SolverState) -> None:
        """
        Rule 9: Unified Data Commitment.
        Optimized via Foundation-level bulk transfer to maintain O(N) scaling.
        """
        # Access the raw NumPy buffer from the state's FieldManager/Foundation
        data = state.fields.data 
        
        # Bulk commit using the FI Enum-locked mapping
        # This is significantly faster than an object-based loop
        data[:, FI.VX] = data[:, FI.VX_STAR]
        data[:, FI.VY] = data[:, FI.VY_STAR]
        data[:, FI.VZ] = data[:, FI.VZ_STAR]
        data[:, FI.P] = data[:, FI.P_NEXT]

    def stabilization(self, is_needed: bool, state: SolverState) -> None:
        """
        Orchestrates time-step recovery and data commitment.
        
        Rule 5 & 9: Explicit State commitment. 
        We do not allow a 'None' state; the solver must fail if the 
        Foundation-to-Logic bridge is missing.
        """
        if not is_needed:
            # Rule 9: Unified Data Commitment via Foundation bulk transfer
            # This must happen before resetting the iteration/dt logic
            self.validate_and_commit(state)

            # Success: Reset to full speed
            self._iteration = 0
            self._dt = self._dt_range[self._iteration]
            return

        if self._iteration >= self._runs:
            raise RuntimeError(
                f"Unstable: reached dt_floor = {self.dt_floor:.2e}. "
                f"Exhausted {self._runs} retries. Update the run configs and restart."
            )
        
        # Advance to the next (smaller) time step
        self._iteration += 1
        self._dt = self._dt_range[self._iteration]
        
        # Rule 8: Singular logging call for audit trail visibility
        self.logger.warning(
            f"Instability. Reducing dt to {self._dt:.2e} "
            f"({self._iteration}/{self._runs})"
        )