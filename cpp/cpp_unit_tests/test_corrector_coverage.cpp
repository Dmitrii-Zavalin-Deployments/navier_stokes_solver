/**
 * @file test_corrector_coverage.cpp
 * @brief Literate C++ Test Suite targeting 100% test coverage for corrector.cpp error paths.
 */

#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>
#include <cmath>
#include "corrector.hpp"

// =========================================================================
// SECTION 1: Grid Dimension & Geometry Validation Tests
// =========================================================================
// WHAT: We verify that the corrector projection intercepts invalid grid dimensions 
//       where nx, ny, or nz are strictly less than 3, throwing an appropriate geometry error.
// WHY:  Central difference stencils and boundary-conforming gradients require at least 
//       a 3x3x3 stencil to evaluate interior nodes safely.

TEST(CorrectorErrorTest, InvalidGeometryDimensions) {
    // We set up invalid dimensions where nx is less than the minimum required 3 cells.
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
    std::vector<int> mask(nx * ny * nz, 1);

    // Expecting an invalid_argument exception due to grid dimensions below 3x3x3.
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
// WHAT: We verify that non-positive grid spacing (dx, dy, dz) triggers an explicit 
//       invalid argument exception in the corrector module.
// WHY:  Dividing by zero or negative grid spacing leads to arithmetic exceptions 
//       and non-physical simulation divergence.

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
    std::vector<int> mask(total_cells, 1);

    // Expecting an invalid_argument exception when dx is non-positive.
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
// WHAT: We verify that non-positive time-step (dt) or density (rho) triggers an 
//       explicit invalid argument exception.
// WHY:  Ensures physical property bounds are strictly enforced before projection.

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
    std::vector<int> mask(total_cells, 1);

    // Expecting an invalid_argument exception when dt <= 0.0.
    EXPECT_THROW({
        navier_stokes_solver::solve_corrector_parallel(
            u, v, w, u_star, v_star, w_star, p, mask,
            nx, ny, nz, 0.1, 0.1, 0.1, 0.0, 1000.0
        );
    }, std::invalid_argument);

    // Expecting an invalid_argument exception when rho <= 0.0.
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
// WHAT: We test that size mismatches between expected total cells and buffer vectors 
//       are caught immediately upon entering the corrector solver routine.
// WHY:  Prevents buffer overflows and out-of-bounds memory access during parallel loops.

TEST(CorrectorErrorTest, VectorSizeMismatch) {
    int nx = 3;
    int ny = 3;
    int nz = 3;
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    
    // Intentionally mismatch vector size (e.g., u is smaller than total_cells)
    std::vector<double> u(total_cells - 1, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);
    std::vector<int> mask(total_cells, 1);

    EXPECT_THROW({
        navier_stokes_solver::solve_corrector_parallel(
            u, v, w, u_star, v_star, w_star, p, mask,
            nx, ny, nz, 0.1, 0.1, 0.1, 0.01, 1000.0
        );
    }, std::invalid_argument);
}

// =========================================================================
// SECTION 5: Non-Finite Velocity Projection & Explosion Handling
// =========================================================================
// WHAT: We test that injecting non-finite pressure or velocity values into the fields 
//       triggers the forensic numerical audit and throws a runtime error.
// WHY:  Ensures that runaway numerical instabilities or floating-point explosions 
//       are safely intercepted rather than propagating silently.

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
    std::vector<int> mask(total_cells, 1);

    // Inject NaN into the pressure field to force a non-finite velocity evaluation
    p[0] = NAN;

    EXPECT_THROW({
        navier_stokes_solver::solve_corrector_parallel(
            u, v, w, u_star, v_star, w_star, p, mask,
            nx, ny, nz, 0.1, 0.1, 0.1, 0.01, 1000.0
        );
    }, std::runtime_error);
}
