# src/engine/elasticity.py

import logging
import numpy as np
from src.common.field_schema import FI

class ElasticManager:
    """
    The 'Nervous System' and 'Validator' of the solver. 
    Manages physical stability and handles the RAM-Git 'Commit' of trial fields.
    
    Compliance:
    - Rule 4 (SSoT): Acts as the sole authority on time-step scaling.
    - Rule 9 (Hybrid Memory): Performs vectorized commits directly on Foundation.
    """
    def __init__(self, base_config):
        # Anchor Points (User-defined targets)
        self.base_dt = base_config.dt
        self.base_max_iter = base_config.ppe_max_iter
        self.base_omega = base_config.ppe_omega
        
        # Current Operating State (The 'Elastic' values)
        self.current_dt = base_config.dt
        self.current_max_iter = base_config.ppe_max_iter
        self.current_omega = base_config.ppe_omega
        
        self.is_in_panic = False
        self.stable_streak = 0
        self.cooldown_limit = 5
        self.logger = logging.getLogger("Elasticity")

    def validate_and_commit(self, state) -> bool:
        """
        The 'Pull Request' Merge. 
        Audits the VX_STAR and P_NEXT fields. If sane, commits to VX and P.
        
        Args:
            state: The SimulationState object containing the fields_buffer.
        Returns:
            bool: True if audit passed and commit succeeded, False otherwise.
        """
        # 1. Audit for NaNs or Infs in the workspace (Vectorized Check)
        # We check all star velocities and the next pressure field in one go.
        audit_fields = [FI.VX_STAR, FI.VY_STAR, FI.VZ_STAR, FI.P_NEXT]
        if not np.isfinite(state.fields_buffer[:, audit_fields]).all():
            self.logger.error("Numerical explosion (NaN/Inf) detected in workspace.")
            return False

        # 2. Physical Sanity Check
        # Threshold (1e5) prevents 'silent' divergence before it becomes a NaN.
        if np.max(np.abs(state.fields_buffer[:, FI.VX_STAR])) > 1e5:
            self.logger.error("Physical velocity limit exceeded. Triggering Panic.")
            return False

        # 3. THE COMMIT (The RAM-Git Merge)
        # Data is moved from 'Trial' columns to 'Truth' columns.
        state.fields_buffer[:, FI.VX] = state.fields_buffer[:, FI.VX_STAR]
        state.fields_buffer[:, FI.VY] = state.fields_buffer[:, FI.VY_STAR]
        state.fields_buffer[:, FI.VZ] = state.fields_buffer[:, FI.VZ_STAR]
        state.fields_buffer[:, FI.P]  = state.fields_buffer[:, FI.P_NEXT]
        
        return True

    def apply_panic_mode(self):
        """Reduces solver aggressiveness to stabilize the system."""
        self.is_in_panic = True
        self.stable_streak = 0
        
        # Aggressive reduction to stop the 'fire'
        self.current_dt *= 0.5
        self.current_omega = max(0.5, self.current_omega - 0.2)
        self.current_max_iter = 5000 
        
        self.logger.warning(f"Panic Mode: dt reduced to {self.current_dt:.2e}, omega to {self.current_omega}")

    def gradual_recovery(self):
        """Slowly heals the simulation parameters toward base configuration."""
        if not self.is_in_panic:
            return

        self.stable_streak += 1
        if self.stable_streak >= self.cooldown_limit:
            # Heal DT (5% growth toward target)
            if self.current_dt < self.base_dt:
                self.current_dt = min(self.base_dt, self.current_dt * 1.05)
            
            # Heal Omega (Step increase)
            if self.current_omega < self.base_omega:
                self.current_omega = min(self.base_omega, self.current_omega + 0.05)
            
            # Check if fully recovered
            if self.current_dt >= self.base_dt and self.current_omega >= self.base_omega:
                self.is_in_panic = False
                self.current_max_iter = self.base_max_iter
                self.logger.info("Simulation health fully restored.")