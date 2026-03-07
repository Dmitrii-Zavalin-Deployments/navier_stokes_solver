# Example: src/step3/ops/projection.py

from .divergence import build_divergence
from .gradient import build_gradient
from .laplacian import build_laplacian


def build_projection_operator(state):
    G = build_gradient(state)
    D = build_divergence(state)
    L = build_laplacian(state)
    
    # Composition of operators
    # PPE: L * p = (rho/dt) * D * v*
    return {"G": G, "D": D, "L": L}