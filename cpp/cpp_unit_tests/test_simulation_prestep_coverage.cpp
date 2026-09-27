/**
 * @file test_simulation_prestep_coverage.cpp
 * @brief Literate Testing Standard — full coverage for simulation_prestep.cpp
 *
 * This suite raises coverage of:
 *
 *     cpp/src/simulation_prestep.cpp   from 87% → 100%
 *
 * by explicitly exercising all remaining uncovered branches:
 *
 *   - Geometry validation (lines 47–48)
 *   - Contract violation (lines 54–56)
 *   - free-slip directional branches (220–222, 224–225, 227, 230–231, 233,
 *                                     236–237, 239, 242–244, 246)
 *
 * All tests follow the Literate Testing Standard:
 * explanatory prose + executable assertions.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "simulation_prestep.hpp"
#include "boundary_condition.hpp"
#include "grid_math.hpp"

using namespace navier_stokes_solver;

/**
 * @section Helper — Construct a minimal 3×3×3 grid.
 *
 * We allocate:
 *   - u, v, w, p as zero fields
 *   - mask as fully fluid (1 everywhere)
 *
 * This provides a clean baseline for all tests.
 */
static void make_fields_3x3x3(
    std::vector<double>& u,
    std::vector<double>& v,
    std::vector<double>& w,
    std::vector<double>& p,
    std::vector<int>& mask
) {
    const int nx = 3, ny = 3, nz = 3;
    const size_t N = static_cast<size_t>(nx) * ny * nz;

    u.assign(N, 0.0);
    v.assign(N, 0.0);
    w.assign(N, 0.0);
    p.assign(N, 0.0);
    mask.assign(N, 1);   // fully fluid
}

/**
 * @section 1 — Geometry validation: nx < 3 triggers std::invalid_argument
 *
 * The implementation checks:
 *
 *   if (nx < 3 || ny < 3 || nz < 3)
 *       throw std::invalid_argument(...);
 *
 * We violate this by setting nx = 2 while keeping other dimensions valid.
 */
TEST(SimulationPrestepCoverage, GeometryValidationNxTooSmall) {

    std::vector<double> u, v, w, p;
    std::vector<int> mask;
    make_fields_3x3x3(u, v, w, p, mask);

    std::vector<BoundaryCondition> bc_list;

    EXPECT_THROW(
        execute_pre_step(
            u, v, w, p,
            mask,
            bc_list,
            /*nx=*/2, /*ny=*/3, /*nz=*/3,
            /*cold_start=*/false
        ),
        std::invalid_argument
    );
}

/**
 * @section 2 — Contract violation: mismatched vector sizes
 *
 * The implementation enforces:
 *
 *   u.size(), v.size(), w.size(), p.size(), mask.size() == total_cells
 *
 * We intentionally make u.size() = total_cells - 1 to trigger the
 * CONTRACT VIOLATION branch.
 */
TEST(SimulationPrestepCoverage, ContractViolationVectorSizeMismatch) {

    const int nx = 3, ny = 3, nz = 3;
    const size_t N = static_cast<size_t>(nx) * ny * nz;

    std::vector<double> u(N - 1, 0.0);   // wrong size
    std::vector<double> v(N, 0.0);
    std::vector<double> w(N, 0.0);
    std::vector<double> p(N, 0.0);
    std::vector<int>    mask(N, 1);

    std::vector<BoundaryCondition> bc_list;

    EXPECT_THROW(
        execute_pre_step(
            u, v, w, p,
            mask,
            bc_list,
            nx, ny, nz,
            /*cold_start=*/false
        ),
        std::invalid_argument
    );
}

/**
 * @section Helper — Construct a BoundaryCondition with explicit values
 *
 * We set all has_* flags to true so that the BC carries explicit
 * u, v, w, p values for use in free-slip, inflow, outflow, and pressure tests.
 */
static BoundaryCondition make_bc(
    const std::string& type,
    const std::string& location,
    double u, double v, double w, double p
) {
    BoundaryCondition bc;
    bc.type = type;
    bc.location = location;
    bc.values.has_u = true;
    bc.values.has_v = true;
    bc.values.has_w = true;
    bc.values.has_p = true;
    bc.values.u = u;
    bc.values.v = v;
    bc.values.w = w;
    bc.values.p = p;
    return bc;
}

/**
 * @section 3 — free-slip directional branches on x_min
 *
 * For free-slip BCs, the code applies:
 *
 *   - i == 0 or i == nx-1 → u_new = 0.0
 *   - j interior          → v_new = values.v
 *   - k interior          → w_new = values.w
 *   - p_new               → values.p (if non-zero)
 *
 * We place a free-slip BC on x_min and verify the resulting field values
 * on boundary and interior cells with i == 0.
 */
TEST(SimulationPrestepCoverage, FreeSlipDirectionalBranches) {

    const int nx = 3, ny = 3, nz = 3;

    std::vector<double> u, v, w, p;
    std::vector<int> mask;
    make_fields_3x3x3(u, v, w, p, mask);

    // free-slip BC with explicit values
    BoundaryCondition bc = make_bc(
        /*type=*/"free-slip",
        /*location=*/"x_min",
        /*u=*/5.0, /*v=*/7.0, /*w=*/9.0, /*p=*/11.0
    );

    std::vector<BoundaryCondition> bc_list = { bc };

    execute_pre_step(
        u, v, w, p,
        mask,
        bc_list,
        nx, ny, nz,
        /*cold_start=*/false
    );

    // Boundary cells (j==0 or j==ny-1 or k==0 or k==nz-1) → v,w forced to 0.0
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            size_t idx = static_cast<size_t>(get_flat_index(0, j, k, nx, ny));
            EXPECT_DOUBLE_EQ(u[idx], 0.0);
            EXPECT_DOUBLE_EQ(p[idx], 11.0);
            if (j == 0 || j == ny - 1 || k == 0 || k == nz - 1) {
                EXPECT_DOUBLE_EQ(v[idx], 0.0);
                EXPECT_DOUBLE_EQ(w[idx], 0.0);
            }
        }
    }

    // Interior cell (i=0, j=1, k=1) → v,w take explicit values
    {
        size_t idx = static_cast<size_t>(get_flat_index(0, 1, 1, nx, ny));
        EXPECT_DOUBLE_EQ(u[idx], 0.0);
        EXPECT_DOUBLE_EQ(v[idx], 7.0);
        EXPECT_DOUBLE_EQ(w[idx], 9.0);
        EXPECT_DOUBLE_EQ(p[idx], 11.0);
    }
}

/**
 * @section 4 — free-slip on y_min face
 *
 * For j == 0 (y_min), the free-slip logic enforces:
 *
 *   v_new = 0.0
 *
 * while u and w follow interior or explicit values.
 */
TEST(SimulationPrestepCoverage, FreeSlipYMinFace) {

    const int nx = 3, ny = 3, nz = 3;

    std::vector<double> u, v, w, p;
    std::vector<int> mask;
    make_fields_3x3x3(u, v, w, p, mask);

    BoundaryCondition bc = make_bc("free-slip", "y_min", 3.0, 4.0, 5.0, 6.0);
    std::vector<BoundaryCondition> bc_list = { bc };

    execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz, false);

    // j == 0 → v_new = 0.0
    for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
            size_t idx = static_cast<size_t>(get_flat_index(i, 0, k, nx, ny));
            EXPECT_DOUBLE_EQ(v[idx], 0.0);
        }
    }
}

/**
 * @section 5 — free-slip on y_max face
 *
 * For j == ny-1 (y_max), the same directional rule applies:
 *
 *   v_new = 0.0
 */
TEST(SimulationPrestepCoverage, FreeSlipYMaxFace) {

    const int nx = 3, ny = 3, nz = 3;

    std::vector<double> u, v, w, p;
    std::vector<int> mask;
    make_fields_3x3x3(u, v, w, p, mask);

    BoundaryCondition bc = make_bc("free-slip", "y_max", 3.0, 4.0, 5.0, 6.0);
    std::vector<BoundaryCondition> bc_list = { bc };

    execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz, false);

    // j == ny-1 → v_new = 0.0
    for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
            size_t idx = static_cast<size_t>(get_flat_index(i, ny - 1, k, nx, ny));
            EXPECT_DOUBLE_EQ(v[idx], 0.0);
        }
    }
}

/**
 * @section 6 — free-slip on z_min face
 *
 * For k == 0 (z_min), the free-slip logic enforces:
 *
 *   w_new = 0.0
 */
TEST(SimulationPrestepCoverage, FreeSlipZMinFace) {

    const int nx = 3, ny = 3, nz = 3;

    std::vector<double> u, v, w, p;
    std::vector<int> mask;
    make_fields_3x3x3(u, v, w, p, mask);

    BoundaryCondition bc = make_bc("free-slip", "z_min", 1.0, 2.0, 3.0, 4.0);
    std::vector<BoundaryCondition> bc_list = { bc };

    execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz, false);

    // k == 0 → w_new = 0.0
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            size_t idx = static_cast<size_t>(get_flat_index(i, j, 0, nx, ny));
            EXPECT_DOUBLE_EQ(w[idx], 0.0);
        }
    }
}

/**
 * @section 7 — free-slip on z_max face
 *
 * For k == nz-1 (z_max), the same directional rule applies:
 *
 *   w_new = 0.0
 */
TEST(SimulationPrestepCoverage, FreeSlipZMaxFace) {

    const int nx = 3, ny = 3, nz = 3;

    std::vector<double> u, v, w, p;
    std::vector<int> mask;
    make_fields_3x3x3(u, v, w, p, mask);

    BoundaryCondition bc = make_bc("free-slip", "z_max", 1.0, 2.0, 3.0, 4.0);
    std::vector<BoundaryCondition> bc_list = { bc };

    execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz, false);

    // k == nz-1 → w_new = 0.0
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            size_t idx = static_cast<size_t>(get_flat_index(i, j, nz - 1, nx, ny));
            EXPECT_DOUBLE_EQ(w[idx], 0.0);
        }
    }
}

/**
 * @section 8 — inflow BC coverage
 *
 * Inflow BCs directly set:
 *
 *   u[idx] = values.u
 *   v[idx] = values.v
 *   w[idx] = values.w
 *
 * We place an inflow BC on x_min and verify the updated velocities.
 */
TEST(SimulationPrestepCoverage, InflowCoverage) {

    const int nx = 3, ny = 3, nz = 3;

    std::vector<double> u, v, w, p;
    std::vector<int> mask;
    make_fields_3x3x3(u, v, w, p, mask);

    BoundaryCondition bc = make_bc("inflow", "x_min", 10.0, 20.0, 30.0, 40.0);
    std::vector<BoundaryCondition> bc_list = { bc };

    execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz, false);

    for (int j = 0; j < ny; ++j)
    for (int k = 0; k < nz; ++k) {
        size_t idx = static_cast<size_t>(get_flat_index(0, j, k, nx, ny));
        EXPECT_DOUBLE_EQ(u[idx], 10.0);
        EXPECT_DOUBLE_EQ(v[idx], 20.0);
        EXPECT_DOUBLE_EQ(w[idx], 30.0);
    }
}

/**
 * @section 9 — outflow BC coverage
 *
 * Outflow BCs use interior references when values.* are zero, but here
 * we provide explicit non-zero values to exercise the direct assignment path:
 *
 *   u[idx] = values.u
 *   v[idx] = values.v
 *   w[idx] = values.w
 *   p[idx] = values.p
 */
TEST(SimulationPrestepCoverage, OutflowCoverage) {

    const int nx = 3, ny = 3, nz = 3;

    std::vector<double> u, v, w, p;
    std::vector<int> mask;
    make_fields_3x3x3(u, v, w, p, mask);

    BoundaryCondition bc = make_bc("outflow", "x_max", 1.0, 2.0, 3.0, 4.0);
    std::vector<BoundaryCondition> bc_list = { bc };

    execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz, false);

    for (int j = 0; j < ny; ++j)
    for (int k = 0; k < nz; ++k) {
        size_t idx = static_cast<size_t>(get_flat_index(nx - 1, j, k, nx, ny));
        EXPECT_DOUBLE_EQ(u[idx], 1.0);
        EXPECT_DOUBLE_EQ(v[idx], 2.0);
        EXPECT_DOUBLE_EQ(w[idx], 3.0);
        EXPECT_DOUBLE_EQ(p[idx], 4.0);
    }
}

/**
 * @section 10 — pressure BC coverage
 *
 * Pressure BCs set:
 *
 *   p[idx] = values.p
 *
 * without modifying velocities. We place a pressure BC on y_max and
 * verify that p is updated while u, v, w remain zero.
 */
TEST(SimulationPrestepCoverage, PressureCoverage) {

    const int nx = 3, ny = 3, nz = 3;

    std::vector<double> u, v, w, p;
    std::vector<int> mask;
    make_fields_3x3x3(u, v, w, p, mask);

    BoundaryCondition bc = make_bc("pressure", "y_max", 0.0, 0.0, 0.0, 99.0);
    std::vector<BoundaryCondition> bc_list = { bc };

    execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz, false);

    for (int i = 0; i < nx; ++i)
    for (int k = 0; k < nz; ++k) {
        size_t idx = static_cast<size_t>(get_flat_index(i, ny - 1, k, nx, ny));
        EXPECT_DOUBLE_EQ(p[idx], 99.0);
        EXPECT_DOUBLE_EQ(u[idx], 0.0);
        EXPECT_DOUBLE_EQ(v[idx], 0.0);
        EXPECT_DOUBLE_EQ(w[idx], 0.0);
    }
}
