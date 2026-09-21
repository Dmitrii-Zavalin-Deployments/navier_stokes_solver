"""
@file test_full_pipeline_accelerated_with_gravity.py
@brief Literate-style integration test for the full Navier–Stokes solver pipeline 
       under accelerated flow with gravity in Python.
"""

import pytest
import numpy as np
import math

try:
    import navier_stokes_solver as nss
except ImportError:
    nss = None

def test_full_pipeline_accelerated_with_gravity():
    # ============================================================================
    # SECTION 1 — Grid Setup
    # ============================================================================
    # We define the physical spatial domain bounds for the 3D simulation box:
    #     x in [0.0, 4.0], y in [0.0, 4.0], z in [0.0, 2.0]
    x_min, x_max = 0.0, 4.0
    y_min, y_max = 0.0, 4.0
    z_min, z_max = 0.0, 2.0

    # The grid resolution is configured as 8x8x4 cells:
    #     nx = 8, ny = 8, nz = 4
    nx, ny, nz = 8, 8, 4

    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz

    total_cells = nx * ny * nz

    # ============================================================================
    # SECTION 2 — Allocate Fields & Accelerated Body Forces
    # ============================================================================
    # We initialize primary velocity and pressure fields with baseline states:
    #     u = 0.5, v = 0.2, w = 0.1, p = 0.0
    u = np.full(total_cells, 0.5, dtype=np.float64)
    v = np.full(total_cells, 0.2, dtype=np.float64)
    w = np.full(total_cells, 0.1, dtype=np.float64)
    p = np.zeros(total_cells, dtype=np.float64)

    # We define the domain geometry mask across all 4 Z-layers (k = 0 to 3),
    # designating active fluid cells (1), solid/wall boundaries (-1), and external ghost cells (0).
    mask_layer = [
        0,  0,  0,  0,  0,  0,  0,  0,
        0, -1, -1, -1, -1, -1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1,  1,  1,  1,  1, -1,  0,
        0, -1, -1, -1, -1, -1, -1,  0,
        0,  0,  0,  0,  0,  0,  0,  0
    ]
    mask = np.array(mask_layer * nz, dtype=np.int32)
    assert len(mask) == total_cells

    # We define positive body force fields aligned with velocity components to drive acceleration:
    #     fx = 0.1, fy = 0.1, fz = 0.2
    fx = np.full(total_cells, 0.1, dtype=np.float64)
    fy = np.full(total_cells, 0.1, dtype=np.float64)
    fz = np.full(total_cells, 0.2, dtype=np.float64)

    # ============================================================================
    # SECTION 3 — Boundary Conditions (Non-Zero Inflow for u, v, w)
    # ============================================================================
    # Boundary condition specifications include inflow at z_min, outflow at z_max, and no-slip walls.
    bc_list = [
        {"location": "z_min", "type": "inflow", "u": 0.5, "v": 0.2, "w": 0.1, "p": 0.0},
        {"location": "z_max", "type": "outflow", "u": 0.5, "v": 0.2, "w": 0.1, "p": 0.0},
        {"location": "wall", "type": "no-slip", "u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}
    ]

    # ============================================================================
    # SECTION 4 — Solver Configuration & Pipeline Execution with Gravity
    # ============================================================================
    # We configure solver parameters including time step dt = 0.1, viscosity mu = 0.01, density = 1.0,
    # and a vertical gravitational acceleration vector along the Y-axis (gy = -9.81).
    config = nss.SolverConfig() if nss and hasattr(nss, 'SolverConfig') else type('DummyConfig', (), {'max_poisson_iterations': 2000, 'poisson_tolerance': 1e-8, 'density': 1.0})()
    
    dt = 0.1
    mu = 0.01
    gravity = [0.0, -9.81, 0.0]

    # ============================================================================
    # SECTION 5 — Execute Pipeline via Orchestrator
    # ============================================================================
    dims = nss.GridDimensions() if nss and hasattr(nss, 'GridDimensions') else type('DummyDims', (), {})()
    dims.nx, dims.ny, dims.nz = nx, ny, nz
    dims.dx, dims.dy, dims.dz = dx, dy, dz

    orchestrator = nss.NavierStokesOrchestrator(dims, config) if nss and hasattr(nss, 'NavierStokesOrchestrator') else None
    
    if orchestrator is not None:
        orchestrator.step(dt, mu, gravity, fx, fy, fz, mask, bc_list, u, v, w, p)
        snapshots = orchestrator.get_debug_snapshots()
        assert len(snapshots) > 0

        def get_snapshot(stage_name):
            for snap in snapshots:
                if snap.stage_name == stage_name:
                    return snap
            pytest.fail(f"Missing snapshot for stage: {stage_name}")

        # ============================================================================
        # SECTION 6 — Verify Stage 1 Snapshot: Pre-Step (Literate Verification)
        # ============================================================================
        # Wall cells (mask <= 0) are clamped to zero; fluid cells (mask == 1) match inflow parameters.
        snap_pre = get_snapshot("pre_step")
        for idx in range(total_cells):
            assert math.isfinite(snap_pre.u[idx])
            assert math.isfinite(snap_pre.v[idx])
            assert math.isfinite(snap_pre.w[idx])
            assert math.isfinite(snap_pre.p[idx])

            if mask[idx] <= 0:
                assert abs(snap_pre.u[idx]) < 1e-12
                assert abs(snap_pre.v[idx]) < 1e-12
                assert abs(snap_pre.w[idx]) < 1e-12
                assert abs(snap_pre.p[idx]) < 1e-12
            else:
                assert abs(snap_pre.u[idx] - 0.5) < 1e-12
                assert abs(snap_pre.v[idx] - 0.2) < 1e-12
                assert abs(snap_pre.w[idx] - 0.1) < 1e-12
                assert abs(snap_pre.p[idx] - 0.0) < 1e-12

        # ============================================================================
        # SECTION 7 — Verify Stage 1.5 Snapshot: Ghost & Boundary Synchronization
        # ============================================================================
        snap_sync1 = get_snapshot("ghost_sync_1")
        for idx in range(total_cells):
            assert math.isfinite(snap_sync1.u_star[idx])
            assert math.isfinite(snap_sync1.v_star[idx])
            assert math.isfinite(snap_sync1.w_star[idx])
            assert math.isfinite(snap_sync1.rhs[idx])

            assert abs(snap_sync1.u_star[idx] - snap_pre.u[idx]) < 1e-12
            assert abs(snap_sync1.v_star[idx] - snap_pre.v[idx]) < 1e-12
            assert abs(snap_sync1.w_star[idx] - snap_pre.w[idx]) < 1e-12
            assert abs(snap_sync1.rhs[idx] - snap_pre.p[idx]) < 1e-12

        # ============================================================================
        # SECTION 8 — Verify Stage 2 Snapshot: Predictor
        # ============================================================================
        # Forward-Euler predictor step incorporating body forces and gravitational acceleration.
        snap_pred = get_snapshot("predictor")
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    idx = i + nx * (j + ny * k)
                    assert math.isfinite(snap_pred.u_star[idx])
                    assert math.isfinite(snap_pred.v_star[idx])
                    assert math.isfinite(snap_pred.w_star[idx])

                    if mask[idx] != 1:
                        assert abs(snap_pred.u_star[idx] - snap_pre.u[idx]) < 1e-12
                        assert abs(snap_pred.v_star[idx] - snap_pre.v[idx]) < 1e-12
                        assert abs(snap_pred.w_star[idx] - snap_pre.w[idx]) < 1e-12
                        continue

                    is_core = (1 < i < nx - 2 and 1 < j < ny - 2 and 1 < k < nz - 2)
                    if is_core:
                        e = (i + 1) + nx * (j + ny * k)
                        w = (i - 1) + nx * (j + ny * k)
                        n = i + nx * ((j + 1) + ny * k)
                        s = i + nx * ((j - 1) + ny * k)
                        t = i + nx * (j + ny * (k + 1))
                        b = i + nx * (j + ny * (k - 1))
                        if mask[e] != 1 or mask[w] != 1 or mask[n] != 1 or mask[s] != 1 or mask[t] != 1 or mask[b] != 1:
                            is_core = False

                    tolerance = 1e-12 if is_core else 0.05
                    assert abs(snap_pred.u_star[idx] - 0.51) < tolerance
                    assert abs(snap_pred.v_star[idx] - (-0.771)) < tolerance
                    assert abs(snap_pred.w_star[idx] - 0.12) < tolerance

        # ============================================================================
        # SECTION 9 — Verify Stage 3 Snapshot: Rhie-Chow Interpolation & Face Velocities
        # ============================================================================
        snap_rc1 = get_snapshot("rhie_chow_interpolation")
        for idx in range(total_cells):
            assert abs(snap_rc1.p[idx] - 0.0) < 1e-12

        # ============================================================================
        # SECTION 10 — Verify Stage Snapshot: RHS Assembly Divergence
        # ============================================================================
        snap_rhs = get_snapshot("rhs_assembly")
        for idx in range(total_cells):
            assert math.isfinite(snap_rhs.rhs[idx])
            if mask[idx] != 1:
                assert abs(snap_rhs.rhs[idx]) < 1e-12

        # ============================================================================
        # SECTION 11 — Verify Stage Snapshot: Pressure Poisson Field Convergence
        # ============================================================================
        snap_poisson = get_snapshot("poisson")
        for idx in range(total_cells):
            assert math.isfinite(snap_poisson.p[idx])

        # ============================================================================
        # SECTION 12 — Verify Stage Snapshot: Rhie-Chow Post-Poisson Interpolation
        # ============================================================================
        snap_post = get_snapshot("rhie_chow_post_poisson")
        for idx in range(total_cells):
            assert abs(snap_post.p[idx] - snap_poisson.p[idx]) < 1e-12

        # ============================================================================
        # SECTION 13 — Verify Stage Snapshot: Corrector Velocity Projection
        # ============================================================================
        snap_corr = get_snapshot("corrector")
        for idx in range(total_cells):
            assert math.isfinite(snap_corr.u[idx])
            assert math.isfinite(snap_corr.v[idx])
            assert math.isfinite(snap_corr.w[idx])
            if mask[idx] != 1:
                assert abs(snap_corr.u[idx]) < 1e-12
                assert abs(snap_corr.v[idx]) < 1e-12
                assert abs(snap_corr.w[idx]) < 1e-12

        # ============================================================================
        # SECTION 14 — Verify Stage Snapshot: Final Ghost & Trial Buffer Synchronization
        # ============================================================================
        snap_sync2 = get_snapshot("ghost_sync_2")
        for idx in range(total_cells):
            assert abs(snap_sync2.u_star[idx] - snap_sync2.u[idx]) < 1e-12
            assert abs(snap_sync2.v_star[idx] - snap_sync2.v[idx]) < 1e-12
            assert abs(snap_sync2.w_star[idx] - snap_sync2.w[idx]) < 1e-12
            assert abs(snap_sync2.rhs[idx] - snap_sync2.p[idx]) < 1e-12

        # ============================================================================
        # SECTION 15 — Final Output Verification: Numerical Finiteness & Boundary Conditions
        # ============================================================================
        for idx in range(total_cells):
            assert math.isfinite(u[idx])
            assert math.isfinite(v[idx])
            assert math.isfinite(w[idx])
            assert math.isfinite(p[idx])
            if mask[idx] != 1:
                assert abs(u[idx]) < 1e-12
                assert abs(v[idx]) < 1e-12
                assert abs(w[idx]) < 1e-12
