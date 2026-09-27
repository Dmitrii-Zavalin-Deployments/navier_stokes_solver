/**
 * @file test_pressure_poisson_apply_solid_neumann.cpp
 * @brief Literate C++ Test Suite for apply_solid_neumann_pressure_parallel.
 *
 * We verify that solid Neumann pressure:
 *  - Returns early for degenerate grids or non-positive spacing.
 *  - Leaves non-solid cells untouched.
 *  - Averages neighboring fluid pressures into solid cells (mask == 0).
 */

#include <gtest/gtest.h>
#include <vector>
#include "pressure_poisson_solver.hpp"

using namespace navier_stokes_solver;

TEST(ApplySolidNeumannPressureTest, DegenerateGridReturnsEarly) {
    std::vector<double> p(1, 5.0);
    std::vector<double> p_tmp;
    std::vector<int> mask(1, 0);

    int nx = 0, ny = 1, nz = 1;
    double dx = 1.0, dy = 1.0, dz = 1.0;

    apply_solid_neumann_pressure_parallel(p, p_tmp, mask, nx, ny, nz, dx, dy, dz);
    EXPECT_DOUBLE_EQ(p[0], 5.0);
}

TEST(ApplySolidNeumannPressureTest, NonPositiveSpacingReturnsEarly) {
    std::vector<double> p(1, 5.0);
    std::vector<double> p_tmp;
    std::vector<int> mask(1, 0);

    int nx = 1, ny = 1, nz = 1;
    double dx = 0.0, dy = 1.0, dz = 1.0;

    apply_solid_neumann_pressure_parallel(p, p_tmp, mask, nx, ny, nz, dx, dy, dz);
    // No change expected.
    EXPECT_DOUBLE_EQ(p[0], 5.0);
}

TEST(ApplySolidNeumannPressureTest, SolidCellAveragesNeighborFluidPressure) {
    int nx = 2, ny = 1, nz = 1;
    std::vector<double> p(nx * ny * nz, 0.0);
    std::vector<double> p_tmp;
    std::vector<int> mask(nx * ny * nz, 1);

    // Mark cell (0,0,0) as solid; (1,0,0) as fluid with pressure 10.
    mask[0] = 0;
    mask[1] = 1;
    p[0] = 0.0;
    p[1] = 10.0;

    double dx = 1.0, dy = 1.0, dz = 1.0;

    apply_solid_neumann_pressure_parallel(p, p_tmp, mask, nx, ny, nz, dx, dy, dz);

    // Solid cell should now equal its fluid neighbor’s pressure.
    EXPECT_DOUBLE_EQ(p[0], 10.0);
    EXPECT_DOUBLE_EQ(p[1], 10.0);
}

