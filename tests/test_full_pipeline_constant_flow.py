"""
@file test_full_pipeline_constant_flow.py
@brief Literate-style integration test for the full Navier-Stokes solver pipeline under constant flow via Python bindings.

Comprehensive Testing Objectives & Rationale:
  - End-to-End Pipeline Validation: Executes the complete C++ solver pipeline via Python bindings, 
    ensuring seamless interoperability and numerical stability across advection, diffusion, pressure gradient 
    projection, and Rhie-Chow interpolation.
  - Blow-Up Prevention & Finiteness Audit: Verifies that primary field arrays (u, v, w, p) remain 
    strictly finite and bounded, preventing any numerical divergence or overflow across time-stepping stages.
"""

import math

import navier_stokes_solver as nss


def test_full_pipeline_constant_flow_python():
    # ============================================================================
    # SECTION 1 — Grid Setup & Domain Initialization
    # ============================================================================
    # We define the spatial bounding box and discrete grid dimensions (8x8x4) 
    # for the computational fluid domain.
    x_min, x_max = 0.0, 4.0
    y_min, y_max = 0.0, 4.0
    z_min, z_max = 0.0, 2.0

    dims = nss.GridDimensions()
    dims.nx = 8
    dims.ny = 8
    dims.nz = 4

    dims.dx = (x_max - x_min) / dims.nx
    dims.dy = (y_max - y_min) / dims.ny
    dims.dz = (z_max - z_min) / dims.nz
    dims.validate()

    total_cells = dims.nx * dims.ny * dims.nz

    # ============================================================================
    # SECTION 2 — Field Allocation & Collocated Mask Setup
    # ============================================================================
    # We initialize primary velocity (u, v, w) and pressure (p) vectors to zero, 
    # alongside a 3D collocated geometry mask (-1 for walls, 0 for solids, 1 for active fluid cells).
    u = [0.0] * total_cells
    v = [0.0] * total_cells
    w = [0.0] * total_cells
    p = [0.0] * total_cells

    # Construct the 4-layer 8x8 mask profile matching the original test configuration
    single_layer_mask = [
        0,  0,  0,  0,  0,  0,  0,  0,
        0, -1, -1, -1, -1, -1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1, -1, -1, -1, -1, -1,  0,
        0,  0,  0,  0,  0,  0,  0,  0
    ]
    mask = single_layer_mask * dims.nz
    assert len(mask) == total_cells

    # ============================================================================
    # SECTION 3 — Boundary Condition Configuration
    # ============================================================================
    # We define inflow at z_min, outflow at z_max, and no-slip walls along lateral boundaries.
    bc_list = []

    # Inflow boundary at z_min
    bc_inflow = nss.BoundaryCondition()
    bc_inflow.location = "z_min"
    bc_inflow.type = "inflow"
    bc_inflow.values.has_w = True; bc_inflow.values.w = 1.0
    bc_inflow.values.has_u = True; bc_inflow.values.u = 0.0
    bc_inflow.values.has_v = True; bc_inflow.values.v = 0.0
    bc_inflow.values.has_p = True; bc_inflow.values.p = 0.0
    bc_list.append(bc_inflow)

    # Outflow boundary at z_max
    bc_outflow = nss.BoundaryCondition()
    bc_outflow.location = "z_max"
    bc_outflow.type = "outflow"
    bc_outflow.values.has_w = True; bc_outflow.values.w = 1.0
    bc_outflow.values.has_u = True; bc_outflow.values.u = 0.0
    bc_outflow.values.has_v = True; bc_outflow.values.v = 0.0
    bc_outflow.values.has_p = True; bc_outflow.values.p = 0.0
    bc_list.append(bc_outflow)

    # No-slip wall boundaries
    bc_wall = nss.BoundaryCondition()
    bc_wall.location = "wall"
    bc_wall.type = "no-slip"
    bc_wall.values.has_u = True; bc_wall.values.u = 0.0
    bc_wall.values.has_v = True; bc_wall.values.v = 0.0
    bc_wall.values.has_w = True; bc_wall.values.w = 0.0
    bc_wall.values.has_p = True; bc_wall.values.p = 0.0
    bc_list.append(bc_wall)

    # ============================================================================
    # SECTION 4 — Solver Execution via Orchestrator C++ Wrapper
    # ============================================================================
    # We configure solver parameters, time step dt = 0.1, kinematic viscosity mu = 0.01,
    # and invoke the fully integrated C++ orchestrator step.
    config = nss.SolverConfig()
    config.max_poisson_iterations = 2000
    config.poisson_tolerance = 1e-8
    config.density = 1.0

    dt = 0.1
    mu = 0.01
    gravity = [0.0, 0.0, 0.0]
    fx = [0.0] * total_cells
    fy = [0.0] * total_cells
    fz = [0.0] * total_cells

    orchestrator = nss.NavierStokesOrchestrator(dims, config)
    
    # Execute the solver step without mocking
    orchestrator.step(dt, mu, gravity, fx, fy, fz, mask, bc_list, u, v, w, p)

    # ============================================================================
    # SECTION 5 — Numerical Stability & Blow-Up Prevention Audit
    # ============================================================================
    # We verify that no field variables (u, v, w, p) blow up, ensuring all values 
    # across the grid remain strictly finite and within physically plausible bounds.
    for idx in range(total_cells):
        assert math.isfinite(u[idx]), f"Non-finite u velocity detected at index {idx}"
        assert math.isfinite(v[idx]), f"Non-finite v velocity detected at index {idx}"
        assert math.isfinite(w[idx]), f"Non-finite w velocity detected at index {idx}"
        assert math.isfinite(p[idx]), f"Non-finite pressure detected at index {idx}"

        # Solid and boundary cells (mask != 1) must strictly satisfy zero velocity constraints
        if mask[idx] != 1:
            assert abs(u[idx]) < 1e-12
            assert abs(v[idx]) < 1e-12
            assert abs(w[idx]) < 1e-12

    # ============================================================================
    # SECTION 6 — Streamwise Invariant & Tiered Spatial Tolerances Verification
    # ============================================================================
    # We inspect internal active fluid cells (mask == 1) to verify that transverse 
    # velocities remain zero and streamwise velocity propagation remains uniform.
    for k in range(dims.nz):
        for j in range(dims.ny):
            for i in range(dims.nx):
                idx = i + dims.nx * (j + dims.ny * k)

                if mask[idx] == 1:
                    # Identify if cell resides within boundary-adjacent buffer layers
                    is_near_boundary = (i < 2 or i >= dims.nx - 2 or
                                        j < 2 or j >= dims.ny - 2 or
                                        k < 2 or k >= dims.nz - 2)

                    transverse_tol = 0.02 if is_near_boundary else 1e-12
                    streamwise_tol = 0.05 if is_near_boundary else 1e-2

                    # Transverse velocities remain zero
                    assert abs(u[idx]) < transverse_tol
                    assert abs(v[idx]) < transverse_tol

                    # Streamwise velocity maintains steady inflow profile (w = 1.0)
                    assert abs(w[idx] - 1.0) < streamwise_tol
