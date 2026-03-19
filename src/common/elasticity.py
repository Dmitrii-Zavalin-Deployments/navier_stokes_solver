# src/step3/elasticity.py

import logging


class ElasticManager:
    """
    The 'Nervous System' of the solver. 
    Manages physical stability and solver convergence through adaptive 
    backtracking and configuration elasticity.
    """
    def __init__(self, base_config):
        # High-Performance Defaults (Anchor Points)
        self.base_dt = base_config.dt
        self.base_max_iter = base_config.ppe_max_iter
        self.base_omega = base_config.ppe_omega
        
        # Current Elastic State
        self.current_dt = base_config.dt
        self.current_max_iter = base_config.ppe_max_iter
        self.current_omega = base_config.ppe_omega
        
        # Stability Tracking
        self.backtrack_count = 0
        self.stable_streak = 0
        self.cooldown_limit = 5  # Steps needed before ramping back up
        
        self.logger = logging.getLogger("Elasticity")

    def audit_and_adapt(self, state, ppe_converged: bool):
        """
        Evaluates the health of the iteration.
        
        Returns:
            should_backtrack (bool): If True, caller must restore state and retry.
            params (dict): Updated parameters for the next attempt.
        """
        # 1. PHYSICAL AUDIT: Check for NaNs or CFL violations
        # (CFL > 1.0 means fluid moved faster than the grid can resolve)
        max_cfl = state.get_max_cfl()
        if state.has_nans() or max_cfl > 1.0:
            self.logger.warning(f"Physical Instability! CFL: {max_cfl:.4f}. Backtracking...")
            self._apply_panic_mode()
            return True, self._get_current_params()

        # 2. CONVERGENCE AUDIT: Did the Pressure Poisson solver succeed?
        if not ppe_converged:
            self.logger.warning("PPE Solver failed to converge. Escalating...")
            # First, try more iterations
            if self.current_max_iter < 5000:
                self.current_max_iter += 1000
                return True, self._get_current_params()
            
            # If still failing, dampen the physics and the math
            self._apply_panic_mode()
            return True, self._get_current_params()

        # 3. RECOVERY: If iteration was successful, slowly heal
        self.stable_streak += 1
        if self.stable_streak >= self.cooldown_limit:
            self._gradual_recovery()
            
        return False, self._get_current_params()

    def _apply_panic_mode(self):
        """Immediate reduction of aggressiveness to stabilize the system."""
        self.stable_streak = 0
        self.backtrack_count += 1
        
        # Cut DT in half
        self.current_dt *= 0.5
        # Under-relax the solver (Calm down oscillations)
        self.current_omega = max(0.7, self.current_omega - 0.1)
        # Max out iterations
        self.current_max_iter = 5000

    def _gradual_recovery(self):
        """Slowly return to base configuration after a stability streak."""
        # Recover DT (5% growth)
        if self.current_dt < self.base_dt:
            self.current_dt = min(self.base_dt, self.current_dt * 1.05)
            
        # Recover Omega (0.02 growth)
        if self.current_omega < self.base_omega:
            self.current_omega = min(self.base_omega, self.current_omega + 0.02)
            
        # Reset Max Iterations immediately once stable
        self.current_max_iter = self.base_max_iter

    def _get_current_params(self):
        """Returns the current operating parameters for the orchestrator."""
        return {
            "dt": self.current_dt,
            "ppe_max_iter": self.current_max_iter,
            "ppe_omega": self.current_omega,
            "backtrack_count": self.backtrack_count
        }