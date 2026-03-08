from __future__ import annotations

from src.solver_state import SolverState
from src.step2.operators import build_gradient  # Centralized SSoT


def build_advection_operators(state: SolverState) -> None:
    """
    Step 2 Logic: Build the Advection convective term matrix (v · ∇).
    
    Rule 8 Compliance: Advection does not define gradients; 
    it consumes them from the operator library.
    """
    g = state.grid
    
    # 1. Access or build the base gradient operators
    # We use the centralized builder to maintain consistency across the solver
    Gx = build_gradient(g.nx, g.ny, g.nz, g.dx, g.dy, g.dz)
    
    # 2. Assemble the convective matrix: C = V_x * Gx + V_y * Gy + V_z * Gz
    # Note: V_x, V_y, V_z here are sparse diagonal matrices of current velocity fields
    # state.advection.C = (Vx @ Gx) + (Vy @ Gy) + (Vz @ Gz)
    
    state.advection.grad_x = Gx
    
    # Assertions for deterministic state (Rule 5)
    assert state.advection.grad_x.shape == (g.nx * g.ny * g.nz, g.nx * g.ny * g.nz)