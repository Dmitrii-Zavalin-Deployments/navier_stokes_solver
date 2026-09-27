/**
 * @file test_pressure_poisson_apply_neumann.cpp
 * @brief Literate C++ Test Suite for apply_neumann_pressure boundary handling.
 *
 * We verify that Neumann pressure application:
 *  - Returns early for degenerate grids (nx <= 0, etc.).
 *  - Throws on non-positive grid spacing (dx, dy, dz).
 *  - Correctly averages neighbor pressures on each face when Dirichlet anchors are absent.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "pressure_poisson_solver.hpp"

using namespace navier_stokes_solver;

// We define a simple 2x2x2 grid to exercise boundary logic.
TEST(ApplyNeumannPressureTest, DegenerateGridReturnsEarly) {
    // p and p_tmp are arbitrary; function should return without modification.
    std::vector<double> p(1, 42.0);
    std::vector<double> p_tmp;

    DirichletFaces dirichlet{};
    int nx = 0, ny = 2, nz = 2;
    double dx = 1.0, dy = 1.0, dz = 1.0;
    double density = 1.0;

    // For nx <= 0, function returns immediately and p remains unchanged.
    apply_neumann_pressure(p, p_tmp, "x_min", dirichlet, nx, ny, nz, dx, dy, dz, density);
    EXPECT_DOUBLE_EQ(p[0], 42.0);
}

TEST(ApplyNeumannPressureTest, NonPositiveSpacingThrows) {
    std::vector<double> p(8, 0.0);
    std::vector<double> p_tmp;
    DirichletFaces dirichlet{};
    int nx = 2, ny = 2, nz = 2;
    double dx = 0.0;  // invalid
    double dy = 1.0;
    double dz = 1.0;
    double density = 1.0;

    // Non-positive spacing must trigger the geometry error.
    EXPECT_THROW(
        apply_neumann_pressure(p, p_tmp, "x_min", dirichlet, nx, ny, nz, dx, dy, dz, density),
        std::invalid_argument
    );
}

TEST(ApplyNeumannPressureTest, FaceAveragingWithoutDirichletAnchors) {
    // 2x2x2 grid: we set interior neighbor pressures and expect boundary cells to be averaged.
    int nx = 2, ny = 2, nz = 2;
    std::vector<double> p(nx * ny * nz, 0.0);
    std::vector<double> p_tmp;

    // We assign a simple pattern so averaging is easy to check.
    // Indexing via get_flat_index(i,j,k,nx,ny) inside the solver.
    p[0] = 0.0;   // (0,0,0) x_min corner
    p[1] = 10.0;  // (1,0,0) neighbor in +x
    p[2] = 20.0;  // (0,1,0)
    p[3] = 30.0;  // (1,1,0)
    p[4] = 40.0;  // (0,0,1)
    p[5] = 50.0;  // (1,0,1)
    p[6] = 60.0;  // (0,1,1)
    p[7] = 70.0;  // (1,1,1)

    DirichletFaces dirichlet{};
    double dx = 1.0, dy = 1.0, dz = 1.0;
    double density = 1.0;

    // Apply Neumann on x_min face; with no Dirichlet anchors, boundary cells at i=0
    // should be set to their neighbor at i=1.
    apply_neumann_pressure(p, p_tmp, "x_min", dirichlet, nx, ny, nz, dx, dy, dz, density);

    // After Neumann, cells at i=0 should equal their i=1 neighbors.
    // (0,0,0) -> neighbor (1,0,0) = 10
    EXPECT_DOUBLE_EQ(p[0], 10.0);
    // (0,1,0) -> neighbor (1,1,0) = 30
    EXPECT_DOUBLE_EQ(p[2], 30.0);
    // (0,0,1) -> neighbor (1,0,1) = 50
    EXPECT_DOUBLE_EQ(p[4], 50.0);
    // (0,1,1) -> neighbor (1,1,1) = 70
    EXPECT_DOUBLE_EQ(p[6], 70.0);
}

