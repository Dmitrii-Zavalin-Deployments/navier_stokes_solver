# src/step4/__init__.py

"""
Step 4: Boundary Enforcement Package.

This package manages the application of physical boundary conditions 
using a dispatcher-applier pattern.

Compliance:
- Rule 8 (API Minimalism): Only the orchestrator is exposed.
- Rule 4 (SSoT): Internal modules are isolated to prevent redundant data 
  access paths and ensure the Foundation remains the exclusive source of state.
"""

from src.step4.orchestrate_step4 import orchestrate_step4

# By explicitly defining __all__, we ensure that internal logic (dispatcher 
# and applier) is protected from external, non-orchestrated access.
__all__ = ["orchestrate_step4"]