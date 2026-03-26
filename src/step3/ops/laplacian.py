# src/step3/ops/laplacian.py

import logging
import numpy as np

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock

# Rule 7: Granular Traceability for Numerical Diffusion Kernels
logger = logging.getLogger("Solver.Ops.Laplacian")

def compute_local_laplacian(block: StencilBlock, field_id: FI) -> float:
    """
    Computes the discrete Laplacian ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
    
    Compliance:
    - Rule 7: Forensic Traceability & Fail-Fast math audit.
    - Rule 8: Centralized logic prevents drift between velocity and pressure ops.
    - Rule 9: Unified Foundation Access via get_field().item().
    """
    
    # 1. Access center and neighbors via Foundation schema
    # Using .item() to ensure scalar precision and prevent NumPy array leaks
    try:
        f_c = block.center.get_field(field_id).item()
        
        f_ip = block.i_plus.get_field(field_id).item()
        f_im = block.i_minus.get_field(field_id).item()
        
        f_jp = block.j_plus.get_field(field_id).item()
        f_jm = block.j_minus.get_field(field_id).item()
        
        f_kp = block.k_plus.get_field(field_id).item()
        f_km = block.k_minus.get_field(field_id).item()
    except AttributeError as e:
        logger.critical(f"TOPOLOGY CRASH: Block {block.id} missing neighbors for Laplacian of {field_id.name}.")
        raise e

    # 2. Geometry Setup (Rule 4: SSoT from block)
    # Check for zero dimensions before squaring to provide a better error message
    if block.dx <= 0 or block.dy <= 0 or block.dz <= 0:
        logger.critical(f"GEOMETRY CRASH: Block {block.id} has non-positive dimensions: [{block.dx}, {block.dy}, {block.dz}]")
        raise ZeroDivisionError(f"Invalid geometry in block {block.id}")

    dx2, dy2, dz2 = block.dx**2, block.dy**2, block.dz**2

    # 3. Discrete Laplacian Calculation (Standard 7-point stencil)
    term_x = (f_ip - 2.0 * f_c + f_im) / dx2
    term_y = (f_jp - 2.0 * f_c + f_jm) / dy2
    term_z = (f_kp - f_km - 2.0 * f_c + f_kp) / dz2 # Logical check on k-neighbors
    
    # Correction of a potential typo in the original logic to ensure symmetry:
    # (f_kp - 2.0 * f_c + f_km) / dz2
    lap_val = term_x + term_y + ((f_kp - 2.0 * f_c + f_km) / dz2)

    # --- FORENSIC NUMERICAL AUDIT ---
    if not np.isfinite(lap_val):
        logger.error(
            f"NUMERICAL INSTABILITY: Non-finite Laplacian in {block.id} | "
            f"Field: {field_id.name} | Center Val: {f_c:.2e} | "
            f"Terms [X:{term_x:.2e}, Y:{term_y:.2e}, Z:{(f_kp-2.0*f_c+f_km)/dz2:.2e}]"
        )
        raise ArithmeticError(f"Laplacian exploded in block {block.id} for field {field_id.name}")

    return lap_val

def compute_local_laplacian_v_n(block: StencilBlock) -> tuple[float, float, float]:
    """Computes Laplacian for primary velocity components (v^n)."""
    logger.debug(f"OPS [Start]: Computing Laplacian Vector (v^n) for {block.id}")
    try:
        return (
            compute_local_laplacian(block, FI.VX),
            compute_local_laplacian(block, FI.VY),
            compute_local_laplacian(block, FI.VZ)
        )
    except Exception as e:
        logger.error(f"OPS [Failure]: Laplacian vector computation failed for {block.id}")
        raise e

def compute_local_laplacian_p_next(block: StencilBlock) -> float:
    """Computes Laplacian for trial pressure p^{n+1}."""
    logger.debug(f"OPS [Start]: Computing Laplacian P_NEXT for {block.id}")
    return compute_local_laplacian(block, FI.P_NEXT)