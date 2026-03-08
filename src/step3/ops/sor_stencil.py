# src/step3/ops/sor_stencil.py

def compute_sor_stencil(p, dx2, dy2, dz2, stencil_denom, rhs):
    """
    Computes the interior Laplacian stencil and subtracts the RHS.
    Matches: [Laplacian(p) - RHS]
    """
    laplacian = (
        (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]) / dx2 +
        (p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]) / dy2 +
        (p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]) / dz2
    )
    return laplacian - rhs