/**
 * @file test_corrector_coverage.cpp
 * @brief Literate C++ Test Suite targeting 100% test coverage for corrector.cpp error paths.
 *
 * Each test is written as a short narrative: comments explain the intent and
 * numerical setup; code performs the actual verification.
 */

#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <limits>
#include "corrector.hpp"

// =========================================================================
// SECTION 1: Grid Dimension & Geometry Validation Tests
// =========================================================================
// WHAT: Verify that the corrector projection intercepts invalid grid dimensions
//       where nx, ny, or nz are strictly less than 3.
// WHY:  Central difference stencils and boundary-conforming gradients require at
//       least a 3x3x3 stencil to evaluate interior nodes safely.

TEST(CorrectorErrorTest, InvalidGeometryDimensions) {
    // We choose nx = 2 < 3 to violate the minimum geometry requirement.
    int nx = 2;
    int ny = 3;
    int nz = 3;
    
    std::vector<double> u(nx * ny * nz, 0.0);
    std::vector<double> v(nx * ny * nz, 0.0);
    std::vector<double> w(nx * ny * nz, 0.0);
    std::vector<double> u_star(nx * ny * nz, 0.0);
    std::vector<double> v_star(nx * ny * nz, 0.0);
    std::vector<double> w_star(nx * ny * nz, 0.0);
    std::vector<double> p(nx * ny * nz, 0.0);
    std::vector<int>    mask(nx * ny * nz, 1);

    // The solver must reject this geometry with std::invalid_argument.
    EXPECT_THROW({
        navier_stokes_solver::solve_corrector_parallel(
            u, v, w, u_star, v_star, w_star, p, mask,
            nx, ny, nz, 0.1, 0.1, 0.1, 0.01, 1000.0
        );
    }, std::invalid_argument);
}

// =========================================================================
// SECTION 2: Grid Spacing Validation Tests
// =========================================================================
// WHAT: Verify that non-positive grid spacing (dx, dy, dz) triggers an explicit
//       invalid_argument exception.
// WHY:  Dividing by zero or negative spacing leads to non-physical divergence.

TEST(CorrectorErrorTest, InvalidGridSpacing) {
    int nx = 3;
    int ny = 3;
    int nz = 3;
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);
    std::vector<int>    mask(total_cells, 1);

    // dx = 0.0 violates the strictly-positive spacing contract.
    EXPECT_THROW({
        navier_stokes_solver::solve_corrector_parallel(
            u, v, w, u_star, v_star, w_star, p, mask,
            nx, ny, nz, 0.0, 0.1, 0.1, 0.01, 1000.0
        );
    }, std::invalid_argument);
}

// =========================================================================
// SECTION 3: Physics Parameter Validation Tests
// =========================================================================
/*
 * WHAT: Verify that non-positive time-step (dt) or density (rho) triggers an
 *       explicit invalid_argument exception.
 * WHY:  Physical parameters must remain strictly positive to avoid undefined
 *       dynamics in the projection step.
 */

TEST(CorrectorErrorTest, InvalidPhysicsParameters) {
    int nx = 3;
    int ny = 3;
    int nz = 3;
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);
    std::vector<int>    mask(total_cells, 1);

    // dt = 0.0 is non-physical and must be rejected.
    EXPECT_THROW({
        navier_stokes_solver::solve_corrector_parallel(
            u, v, w, u_star, v_star, w_star, p, mask,
            nx, ny, nz, 0.1, 0.1, 0.1, 0.0, 1000.0
        );
    }, std::invalid_argument);

    // rho < 0.0 is also non-physical and must be rejected.
    EXPECT_THROW({
        navier_stokes_solver::solve_corrector_parallel(
            u, v, w, u_star, v_star, w_star, p, mask,
            nx, ny, nz, 0.1, 0.1, 0.1, 0.01, -500.0
        );
    }, std::invalid_argument);
}

// =========================================================================
// SECTION 4: Vector Size Contract Violation Tests
// =========================================================================
/*
 * WHAT: Verify that buffer size mismatches are caught immediately.
 * WHY:  Prevents out-of-bounds access in the parallel loops.
 */

TEST(CorrectorErrorTest, VectorSizeMismatch) {
    int nx = 3;
    int ny = 3;
    int nz = 3;
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    
    // u is intentionally smaller than total_cells to violate the contract.
    std::vector<double> u(total_cells - 1, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);
    std::vector<int>    mask(total_cells, 1);

    EXPECT_THROW({
        navier_stokes_solver::solve_corrector_parallel(
            u, v, w, u_star, v_star, w_star, p, mask,
            nx, ny, nz, 0.1, 0.1, 0.1, 0.01, 1000.0
        );
    }, std::invalid_argument);
}

// =========================================================================
// SECTION 5: Non-Finite Velocity Projection & Loop Error Handling
// =========================================================================
/*
 * WHAT: Verify that finite inputs which overflow to Infinity during the
 *       projection step trigger the forensic numerical audit.
 * WHY:  Ensures the runtime error block at the end of the solver is exercised.
 */

TEST(CorrectorErrorTest, NonFiniteVelocityExplosion) {
    int nx = 3;
    int ny = 3;
    int nz = 3;
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);

    std::vector<double> p(total_cells, 0.0);
    std::vector<int>    mask(total_cells, 1);

    // For nx = ny = nz = 3, interior cell (1,1,1) and its west/east neighbors
    // (assuming row-major get_flat_index):
    //   west  index ≈ 12
    //   east  index ≈ 14
    p[12] = -1.0e150;  // west neighbor
    p[14] =  1.0e150;  // east neighbor

    // dx is tiny but positive; the pressure difference divided by dx can overflow.
    double dx  = 1.0e-160;
    double dy  = 0.1;
    double dz  = 0.1;
    double dt  = 0.01;
    double rho = 1.0;

    EXPECT_THROW({
        navier_stokes_solver::solve_corrector_parallel(
            u, v, w, u_star, v_star, w_star, p, mask,
            nx, ny, nz, dx, dy, dz, dt, rho
        );
    }, std::runtime_error);
}

// =========================================================================
// SECTION 6: Input Finiteness Audit (u_star / p non-finite)
// =========================================================================
/*
 * WHAT: Exercise the early input finiteness audit that scans u_star, v_star,
 *       w_star, and p for NaN/Inf before any projection work.
 * WHY:  This branch guards against corrupted input fields.
 */

TEST(CorrectorErrorTest, NonFiniteInputAudit) {
    int nx = 3;
    int ny = 3;
    int nz = 3;
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);
    std::vector<int>    mask(total_cells, 1);

    // We inject a NaN into u_star[0] so that the finiteness audit fails.
    u_star[0] = std::numeric_limits<double>::quiet_NaN();

    double dx  = 0.1;
    double dy  = 0.1;
    double dz  = 0.1;
    double dt  = 0.01;
    double rho = 1.0;

    EXPECT_THROW({
        navier_stokes_solver::solve_corrector_parallel(
            u, v, w, u_star, v_star, w_star, p, mask,
            nx, ny, nz, dx, dy, dz, dt, rho
        );
    }, std::runtime_error);
}

// =========================================================================
// SECTION 7: Solid Velocity Clamping Pass (mask == 0)
// =========================================================================
/*
 * WHAT: Verify that solid cells (mask == 0) are clamped to zero velocity in
 *       the post-processing pass.
 * WHY:  Ensures boundary conditions are enforced even after projection.
 */

TEST(CorrectorErrorTest, SolidVelocityClamping) {
    int nx = 3;
    int ny = 3;
    int nz = 3;
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    // We start with non-zero velocities everywhere.
    std::vector<double> u(total_cells, 1.0);
    std::vector<double> v(total_cells, 2.0);
    std::vector<double> w(total_cells, 3.0);
    std::vector<double> u_star(total_cells, 1.0);
    std::vector<double> v_star(total_cells, 2.0);
    std::vector<double> w_star(total_cells, 3.0);
    std::vector<double> p(total_cells, 0.0);
    std::vector<int>    mask(total_cells, 1);

    // We mark a few cells as solid:
    //   mask[idx] = 0  ⇒  u[idx], v[idx], w[idx] must be clamped to 0.0.
    mask[0]  = 0;
    mask[5]  = 0;
    mask[10] = 0;

    double dx  = 0.1;
    double dy  = 0.1;
    double dz  = 0.1;
    double dt  = 0.01;
    double rho = 1.0;

    // The solver should complete without throwing and enforce clamping.
    navier_stokes_solver::solve_corrector_parallel(
        u, v, w, u_star, v_star, w_star, p, mask,
        nx, ny, nz, dx, dy, dz, dt, rho
    );

    // We now assert that all marked solid cells have zero velocity.
    EXPECT_DOUBLE_EQ(u[0], 0.0);
    EXPECT_DOUBLE_EQ(v[0], 0.0);
    EXPECT_DOUBLE_EQ(w[0], 0.0);

    EXPECT_DOUBLE_EQ(u[5], 0.0);
    EXPECT_DOUBLE_EQ(v[5], 0.0);
    EXPECT_DOUBLE_EQ(w[5], 0.0);

    EXPECT_DOUBLE_EQ(u[10], 0.0);
    EXPECT_DOUBLE_EQ(v[10], 0.0);
    EXPECT_DOUBLE_EQ(w[10], 0.0);
}
