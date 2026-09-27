/**
 * @file test_pressure_poisson_solver_core.cpp
 * @brief Literate C++ Test Suite for solve_poisson_red_black_parallel defensive gates and math failure paths.
 *
 * We verify:
 *  - Geometry and spacing validation (nx, ny, nz, dx, dy, dz).
 *  - Iteration/tolerance validation.
 *  - Vector size contract checks.
 *  - Pure Neumann configuration detection (zero Dirichlet boundaries).
 *  - Non-finite pressure detection in the Red-Black passes.
 *  - Boundary pressure application for each face location.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "pressure_poisson_solver.hpp"
#include "grid_math.hpp"

using namespace navier_stokes_solver;

static BoundaryCondition make_pressure_bc(const std::string& location, double p_val) {
    BoundaryCondition bc;
    bc.type = "pressure";
    bc.location = location;
    bc.scalar_p = p_val;
    bc.values.has_p = true;
    bc.values.p = p_val;
    return bc;
}

TEST(PressurePoissonSolverCoreTest, GeometryAndSpacingValidation) {
    std::vector<double> p(27, 0.0);
    std::vector<double> rhs(27, 0.0);
    std::vector<int> mask(27, 1);
    std::vector<BoundaryCondition> bc_list{ make_pressure_bc("x_min", 0.0) };

    // nx < 3 should throw geometry error.
    EXPECT_THROW(
        solve_poisson_red_black_parallel(p, rhs, mask, bc_list,
                                         2, 3, 3, 1.0, 1.0, 1.0,
                                         10, 1e-3, 1.0),
        std::invalid_argument
    );

    // Non-positive spacing should throw.
    EXPECT_THROW(
        solve_poisson_red_black_parallel(p, rhs, mask, bc_list,
                                         3, 3, 3, 0.0, 1.0, 1.0,
                                         10, 1e-3, 1.0),
        std::invalid_argument
    );

    // Invalid max_iters or tol should throw.
    EXPECT_THROW(
        solve_poisson_red_black_parallel(p, rhs, mask, bc_list,
                                         3, 3, 3, 1.0, 1.0, 1.0,
                                         0, -1e-3, 1.0),
        std::invalid_argument
    );
}

TEST(PressurePoissonSolverCoreTest, VectorSizeContractViolation) {
    std::vector<double> p(27, 0.0);
    std::vector<double> rhs(26, 0.0);  // mismatch
    std::vector<int> mask(27, 1);
    std::vector<BoundaryCondition> bc_list{ make_pressure_bc("x_min", 0.0) };

    EXPECT_THROW(
        solve_poisson_red_black_parallel(p, rhs, mask, bc_list,
                                         3, 3, 3, 1.0, 1.0, 1.0,
                                         10, 1e-3, 1.0),
        std::invalid_argument
    );
}

TEST(PressurePoissonSolverCoreTest, PureNeumannConfigurationThrowsRuntime) {
    std::vector<double> p(27, 0.0);
    std::vector<double> rhs(27, 0.0);
    std::vector<int> mask(27, 1);
    std::vector<BoundaryCondition> bc_list;  // no pressure/outflow boundaries

    EXPECT_THROW(
        solve_poisson_red_black_parallel(p, rhs, mask, bc_list,
                                         3, 3, 3, 1.0, 1.0, 1.0,
                                         10, 1e-3, 1.0),
        std::runtime_error
    );
}

TEST(PressurePoissonSolverCoreTest, NonFinitePressureExplosionInRedPass) {
    std::vector<double> p(27, 0.0);
    std::vector<double> rhs(27, 0.0);
    std::vector<int> mask(27, 1);
    std::vector<BoundaryCondition> bc_list{ make_pressure_bc("x_min", 0.0) };

    int nx = 3, ny = 3, nz = 3;
    double dx = 1.0, dy = 1.0, dz = 1.0;
    int max_iters = 1;
    double tol = 0.0;
    double density = 1.0;

    // Force a non-finite update by setting rhs to a huge value at a red cell.
    // Red cells satisfy (i + j + k) % 2 == 0; choose (1,1,1).
    int idx_center = get_flat_index(1, 1, 1, nx, ny);
    rhs[static_cast<size_t>(idx_center)] = std::numeric_limits<double>::infinity();

    EXPECT_THROW(
        solve_poisson_red_black_parallel(p, rhs, mask, bc_list,
                                         nx, ny, nz, dx, dy, dz,
                                         max_iters, tol, density),
        std::runtime_error
    );
}

TEST(PressurePoissonSolverCoreTest, BoundaryPressureApplicationFaces) {
    int nx = 3, ny = 3, nz = 3;
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    std::vector<double> p(total_cells, 0.0);
    std::vector<double> rhs(total_cells, 0.0);
    std::vector<int> mask(total_cells, 1);

    // We add Dirichlet pressure boundaries on all faces with distinct values.
    std::vector<BoundaryCondition> bc_list;
    bc_list.push_back(make_pressure_bc("x_min", 1.0));
    bc_list.push_back(make_pressure_bc("x_max", 2.0));
    bc_list.push_back(make_pressure_bc("y_min", 3.0));
    bc_list.push_back(make_pressure_bc("y_max", 4.0));
    bc_list.push_back(make_pressure_bc("z_min", 5.0));
    bc_list.push_back(make_pressure_bc("z_max", 6.0));

    double dx = 1.0, dy = 1.0, dz = 1.0;
    int max_iters = 1;
    double tol = 0.0;
    double density = 1.0;

    // Run a single iteration; boundary application happens inside the loop.
    solve_poisson_red_black_parallel(p, rhs, mask, bc_list,
                                     nx, ny, nz, dx, dy, dz,
                                     max_iters, tol, density);

    // Check that each face has been set to its corresponding pressure value.
    // x_min: i = 0
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            int idx = get_flat_index(0, j, k, nx, ny);
            ASSERT_GE(idx, 0);
            EXPECT_DOUBLE_EQ(p[static_cast<size_t>(idx)], 1.0);
        }
    }
    // x_max: i = nx - 1
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            int idx = get_flat_index(nx - 1, j, k, nx, ny);
            ASSERT_GE(idx, 0);
            EXPECT_DOUBLE_EQ(p[static_cast<size_t>(idx)], 2.0);
        }
    }
    // y_min: j = 0
    for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
            int idx = get_flat_index(i, 0, k, nx, ny);
            ASSERT_GE(idx, 0);
            EXPECT_DOUBLE_EQ(p[static_cast<size_t>(idx)], 3.0);
        }
    }
    // y_max: j = ny - 1
    for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
            int idx = get_flat_index(i, ny - 1, k, nx, ny);
            ASSERT_GE(idx, 0);
            EXPECT_DOUBLE_EQ(p[static_cast<size_t>(idx)], 4.0);
        }
    }
    // z_min: k = 0
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = get_flat_index(i, j, 0, nx, ny);
            ASSERT_GE(idx, 0);
            EXPECT_DOUBLE_EQ(p[static_cast<size_t>(idx)], 5.0);
        }
    }
    // z_max: k = nz - 1
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = get_flat_index(i, j, nz - 1, nx, ny);
            ASSERT_GE(idx, 0);
            EXPECT_DOUBLE_EQ(p[static_cast<size_t>(idx)], 6.0);
        }
    }
}

