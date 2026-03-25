# src/step3/predictor.py

import logging

from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock
from src.step3.ops.advection import compute_local_advection_vector
from src.step3.ops.forces import get_local_body_force
from src.step3.ops.gradient import compute_local_gradient_p
from src.step3.ops.laplacian import compute_local_laplacian_v_n
from src.step3.ops.scaling import get_dt_over_rho

# Initialize logger for this module
logger = logging.getLogger("Solver.Predictor")

def compute_local_predictor_step(block: StencilBlock) -> None:
    """
    Computes the intermediate star-velocity field v*.
    
    Formula: v* = v^n + (dt/rho) * (mu * lap(v^n) - rho * (v^n ⋅ ∇)v^n + F - grad(p^n))
    """
    
    # 1. Local Operator calls
    lap_v = compute_local_laplacian_v_n(block)    
    adv_v = compute_local_advection_vector(block) 
    force = get_local_body_force(block)           
    grad_p = compute_local_gradient_p(block, field_id=FI.P) 
    
    # 2. Scaling factor
    dt_over_rho = get_dt_over_rho(block)
    
    # 3. Retrieve current velocities
    v_n = (
        block.center.get_field(FI.VX),
        block.center.get_field(FI.VY),
        block.center.get_field(FI.VZ)
    )
    
    # --- [DIAGNOSTIC: OPERATOR OUTPUT AUDIT] ---
    # We check the first component (index 0) of every operator result.
    # If any of these show a 'shape' property, they are arrays, not scalars.
    operators = {
        "lap_v[0]": lap_v[0],
        "adv_v[0]": adv_v[0],
        "force[0]": force[0],
        "grad_p[0]": grad_p[0],
        "v_n[0]": v_n[0],
        "dt_over_rho": dt_over_rho
    }

    for name, val in operators.items():
        val_type = type(val)
        shape = getattr(val, "shape", "N/A (Scalar)")
        logger.debug(f"AUDIT [{name}]: Type={val_type} | Shape={shape} | Value={val}")

    # 4. Compute and audit v_star components
    star_fields = [FI.VX_STAR, FI.VY_STAR, FI.VZ_STAR]
    
    for i, field_id in enumerate(star_fields):
        # Calculate intermediate value
        # Note: If this fails, the logger above will have already told us which variable is the array.
        try:
            v_star_val = (v_n[i] + dt_over_rho * (
                block.mu * lap_v[i] - block.rho * adv_v[i] + force[i] - grad_p[i]
            )).item()
            
            # Final check before commitment
            if hasattr(v_star_val, "__len__"):
                logger.error(f"CONTAMINATION DETECTED: field {field_id} produced array of shape {v_star_val.shape}")
                # Emergency cast to allow the log/run to proceed for debugging
                v_star_val = float(v_star_val.item())

            if i == 0:
                logger.info(f"DEBUG [Predictor]: Block {block.id} | VX_STAR: {v_star_val:.4e}")

            # Commit to the Trial (Star) buffer
            block.center.set_field(field_id, v_star_val)
            
        except Exception as e:
            logger.critical(f"MATH FAILURE in component {i}: {e}")
            raise