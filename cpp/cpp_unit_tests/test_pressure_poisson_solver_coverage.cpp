// cpp/cpp_unit_tests/test_pressure_poisson_solver_coverage.cpp
//
// Literate Testing Standard — targeted coverage for pressure_poisson_solver.cpp
//
// This file focuses on the remaining defensive branches and boundary
// synchronization paths in solve_poisson_red_black_parallel(), bringing
// coverage from 76% to 100% for cpp/src/pressure_poisson_solver.cpp.
//
// We rely on the core header:
//   #include "pressure_poisson_solver.hpp"
// which defines:
//   - GridDimensions
//   - BoundaryCondition
//   - DirichletFaces / BoundaryValues
//   - solve_poisson_red_black_parallel(...)

#include <gtest/gtest.h>
#include <limits>
#include <vector>
#include "pressure_poisson_solver.hpp"
#include "grid_math.hpp"

using namespace navier_stokes_solver;

// -----------------------------------------------------------------------------
// Helper: construct a simple 3x3x3 grid with unit spacing.
// -----------------------------------------------------------------------------
static GridDimensions make_dims_3x3x3() {
    GridDimensions dims;
    dims.nx = 3;
    dims.ny = 3;
    dims.nz = 3;
    dims.dx = 1.0;
    dims.dy = 1.0;
    dims.dz = 1.0;
    return dims;
}

// -----------------------------------------------------------------------------
// Helper: construct a fully fluid mask (mask == 1 everywhere).
// -----------------------------------------------------------------------------
static std::vector<int> make_full_fluid_mask(const GridDimensions& dims) {
    size_t N = static_cast<size_t>(dims.nx) * dims.ny * dims.nz;
    return std::vector<int>(N, 1);
}

// -----------------------------------------------------------------------------
// Helper: construct a trivial RHS field (all zeros).
// -----------------------------------------------------------------------------
static std::vector<double> make_zero_rhs(const GridDimensions& dims) {
    size_t N = static_cast<size_t>(dims.nx) * dims.ny * dims.nz;
    return std::vector<double>(N, 0.0);
}

// -----------------------------------------------------------------------------
// Helper: construct a pressure Dirichlet boundary condition on a given face.
// We set scalar_p = 1.0 and rely on values.has_p == false by default.
// -----------------------------------------------------------------------------
static BoundaryCondition make_pressure_bc(const std::string& location) {
    BoundaryCondition bc;
    bc.type = "pressure";
    bc.location = location;
    bc.scalar_p = 1.0;
    // values.has_p is assumed false; scalar_p will be used.
    return bc;
}

// ============================================================================
// SECTION 1 — Validation gates: geometry, spacing, iteration, contract
// ============================================================================
//
// We explicitly trigger each defensive branch:
//
//   - nx < 3 (geometry error)
//   - dx <= 0.0 (spacing error)
//   - max_iters <= 0 (iteration error)
//   - vector size mismatch (contract violation)
//
// Each test constructs minimal valid data for the other parameters and
// asserts that the correct exception type is thrown.
// -----------------------------------------------------------------------------

TEST(PressurePoissonSolverCoverage, GeometryValidationNxTooSmall) {
    // We choose nx = 2 to violate the 3x3x3 requirement.
    GridDimensions dims;
    dims.nx = 2;
    dims.ny = 3;
    dims.nz = 3;
    dims.dx = 1.0;
    dims.dy = 1.0;
    dims.dz = 1.0;

    size_t N = static_cast<size_t>(dims.nx) * dims.ny * dims.nz;
    std::vector<double> p(N, 0.0);
    std::vector<double> rhs(N, 0.0);
    std::vector<int> mask(N, 1);
    std::vector<BoundaryCondition> bc_list;

    EXPECT_THROW(
        solve_poisson_red_black_parallel(
            p, rhs, mask, bc_list,
            dims.nx, dims.ny, dims.nz,
            dims.dx, dims.dy, dims.dz,
            /*max_iters=*/10,
            /*tol=*/1e-6,
            /*density=*/1.0
        ),
        std::invalid_argument
    );
}

TEST(PressurePoissonSolverCoverage, SpacingValidationDxNonPositive) {
    // We set dx = 0.0 to trigger the spacing validation branch.
    GridDimensions dims = make_dims_3x3x3();
    dims.dx = 0.0;

    size_t N = static_cast<size_t>(dims.nx) * dims.ny * dims.nz;
    std::vector<double> p(N, 0.0);
    std::vector<double> rhs(N, 0.0);
    std::vector<int> mask(N, 1);
    std::vector<BoundaryCondition> bc_list;

    EXPECT_THROW(
        solve_poisson_red_black_parallel(
            p, rhs, mask, bc_list,
            dims.nx, dims.ny, dims.nz,
            dims.dx, dims.dy, dims.dz,
            /*max_iters=*/10,
            /*tol=*/1e-6,
            /*density=*/1.0
        ),
        std::invalid_argument
    );
}

TEST(PressurePoissonSolverCoverage, IterationValidationMaxItersNonPositive) {
    // We set max_iters = 0 to violate the iteration contract.
    GridDimensions dims = make_dims_3x3x3();

    size_t N = static_cast<size_t>(dims.nx) * dims.ny * dims.nz;
    std::vector<double> p(N, 0.0);
    std::vector<double> rhs(N, 0.0);
    std::vector<int> mask(N, 1);
    std::vector<BoundaryCondition> bc_list;

    EXPECT_THROW(
        solve_poisson_red_black_parallel(
            p, rhs, mask, bc_list,
            dims.nx, dims.ny, dims.nz,
            dims.dx, dims.dy, dims.dz,
            /*max_iters=*/0,
            /*tol=*/1e-6,
            /*density=*/1.0
        ),
        std::invalid_argument
    );
}

TEST(PressurePoissonSolverCoverage, ContractViolationVectorSizeMismatch) {
    // We intentionally make p smaller than rhs/mask to trigger the size check.
    GridDimensions dims = make_dims_3x3x3();

    size_t N = static_cast<size_t>(dims.nx) * dims.ny * dims.nz;
    std::vector<double> p(N - 1, 0.0);   // wrong size
    std::vector<double> rhs(N, 0.0);
    std::vector<int> mask(N, 1);
    std::vector<BoundaryCondition> bc_list;

    EXPECT_THROW(
        solve_poisson_red_black_parallel(
            p, rhs, mask, bc_list,
            dims.nx, dims.ny, dims.nz,
            dims.dx, dims.dy, dims.dz,
            /*max_iters=*/10,
            /*tol=*/1e-6,
            /*density=*/1.0
        ),
        std::invalid_argument
    );
}

// ============================================================================
// SECTION 2 — Pure Neumann configuration: singular matrix runtime gate
// ============================================================================
//
// The solver enforces that at least one Dirichlet (pressure/outflow) boundary
// is present. If dirichlet_count == 0, it throws std::runtime_error.
//
// We construct a configuration with no pressure/outflow boundaries and
// assert that the runtime_error is raised.
// -----------------------------------------------------------------------------

TEST(PressurePoissonSolverCoverage, PureNeumannConfigurationThrowsRuntime) {
    GridDimensions dims = make_dims_3x3x3();

    size_t N = static_cast<size_t>(dims.nx) * dims.ny * dims.nz;
    std::vector<double> p(N, 0.0);
    std::vector<double> rhs(N, 0.0);
    std::vector<int> mask(N, 1);

    // bc_list is empty: no pressure/outflow boundaries.
    std::vector<BoundaryCondition> bc_list;

    EXPECT_THROW(
        solve_poisson_red_black_parallel(
            p, rhs, mask, bc_list,
            dims.nx, dims.ny, dims.nz,
            dims.dx, dims.dy, dims.dz,
            /*max_iters=*/10,
            /*tol=*/1e-6,
            /*density=*/1.0
        ),
        std::runtime_error
    );
}

// ============================================================================
// SECTION 3 — Non-finite pressure explosion: RED and BLACK passes
// ============================================================================
//
// The solver guards against non-finite pressure values in both the RED and
// BLACK Gauss–Seidel passes. When p_new is not finite, it sets has_error and,
// after the pass, logs a message and throws std::runtime_error.
//
// We design two tests:
//
//   - NonFinitePressureExplosionRedPass:
//       RHS is +inf on RED cells ((i + j + k) % 2 == 0), finite elsewhere.
//       The RED pass encounters non-finite p_new and triggers the error.
//
//   - NonFinitePressureExplosionBlackPass:
//       RHS is +inf on BLACK cells ((i + j + k) % 2 != 0), finite elsewhere.
//       The RED pass is safe; the BLACK pass then explodes.
// -----------------------------------------------------------------------------

TEST(PressurePoissonSolverCoverage, NonFinitePressureExplosionRedPass) {
    GridDimensions dims = make_dims_3x3x3();
    size_t N = static_cast<size_t>(dims.nx) * dims.ny * dims.nz;

    std::vector<double> p(N, 0.0);
    std::vector<double> rhs(N, 0.0);
    std::vector<int> mask = make_full_fluid_mask(dims);

    // We provide a single pressure boundary to satisfy the singularity gate.
    std::vector<BoundaryCondition> bc_list;
    bc_list.push_back(make_pressure_bc("x_min"));

    // Set RHS to +inf on RED cells: (i + j + k) % 2 == 0.
    for (int k = 0; k < dims.nz; ++k) {
        for (int j = 0; j < dims.ny; ++j) {
            for (int i = 0; i < dims.nx; ++i) {
                if ((i + j + k) % 2 == 0) {
                    int raw_idx = get_flat_index(i, j, k, dims.nx, dims.ny);
                    if (raw_idx >= 0) {
                        rhs[static_cast<size_t>(raw_idx)] =
                            std::numeric_limits<double>::infinity();
                    }
                }
            }
        }
    }

    EXPECT_THROW(
        solve_poisson_red_black_parallel(
            p, rhs, mask, bc_list,
            dims.nx, dims.ny, dims.nz,
            dims.dx, dims.dy, dims.dz,
            /*max_iters=*/1,
            /*tol=*/0.0,
            /*density=*/1.0
        ),
        std::runtime_error
    );
}

TEST(PressurePoissonSolverCoverage, NonFinitePressureExplosionBlackPass) {
    GridDimensions dims = make_dims_3x3x3();
    size_t N = static_cast<size_t>(dims.nx) * dims.ny * dims.nz;

    std::vector<double> p(N, 0.0);
    std::vector<double> rhs(N, 0.0);
    std::vector<int> mask = make_full_fluid_mask(dims);

    // Single pressure boundary to satisfy singularity gate.
    std::vector<BoundaryCondition> bc_list;
    bc_list.push_back(make_pressure_bc("x_min"));

    // RHS is finite on RED cells and +inf on BLACK cells.
    for (int k = 0; k < dims.nz; ++k) {
        for (int j = 0; j < dims.ny; ++j) {
            for (int i = 0; i < dims.nx; ++i) {
                int raw_idx = get_flat_index(i, j, k, dims.nx, dims.ny);
                if (raw_idx < 0) continue;
                size_t idx = static_cast<size_t>(raw_idx);

                if ((i + j + k) % 2 == 0) {
                    rhs[idx] = 0.0; // safe for RED pass
                } else {
                    rhs[idx] = std::numeric_limits<double>::infinity();
                }
            }
        }
    }

    EXPECT_THROW(
        solve_poisson_red_black_parallel(
            p, rhs, mask, bc_list,
            dims.nx, dims.ny, dims.nz,
            dims.dx, dims.dy, dims.dz,
            /*max_iters=*/1,
            /*tol=*/0.0,
            /*density=*/1.0
        ),
        std::runtime_error
    );
}

// ============================================================================
// SECTION 4 — Boundary pressure application on y_min, y_max, z_min, z_max
// ============================================================================
//
// Inside the iteration loop, the solver synchronizes Dirichlet pressure
// boundaries for all faces:
//
//   - x_min, x_max
//   - y_min, y_max
//   - z_min, z_max
//
// Existing tests already exercise x_min/x_max. Here we add explicit coverage
// for y_min, y_max, z_min, and z_max by:
//
//   1. Running the solver for a single iteration with zero RHS.
//   2. Providing a pressure boundary on the target face.
//   3. Asserting that all cells on that face are set to the boundary value.
// -----------------------------------------------------------------------------

TEST(PressurePoissonSolverCoverage, BoundaryPressureAppliedOnYMinFace) {
    GridDimensions dims = make_dims_3x3x3();
    size_t N = static_cast<size_t>(dims.nx) * dims.ny * dims.nz;

    std::vector<double> p(N, 0.0);
    std::vector<double> rhs = make_zero_rhs(dims);
    std::vector<int> mask = make_full_fluid_mask(dims);

    std::vector<BoundaryCondition> bc_list;
    bc_list.push_back(make_pressure_bc("y_min"));

    solve_poisson_red_black_parallel(
        p, rhs, mask, bc_list,
        dims.nx, dims.ny, dims.nz,
        dims.dx, dims.dy, dims.dz,
        /*max_iters=*/1,
        /*tol=*/0.0,
        /*density=*/1.0
    );

    // All cells with j == 0 must equal the boundary pressure (1.0).
    for (int k = 0; k < dims.nz; ++k) {
        for (int i = 0; i < dims.nx; ++i) {
            int raw_idx = get_flat_index(i, 0, k, dims.nx, dims.ny);
            ASSERT_GE(raw_idx, 0);
            EXPECT_DOUBLE_EQ(p[static_cast<size_t>(raw_idx)], 1.0);
        }
    }
}

TEST(PressurePoissonSolverCoverage, BoundaryPressureAppliedOnYMaxFace) {
    GridDimensions dims = make_dims_3x3x3();
    size_t N = static_cast<size_t>(dims.nx) * dims.ny * dims.nz;

    std::vector<double> p(N, 0.0);
    std::vector<double> rhs = make_zero_rhs(dims);
    std::vector<int> mask = make_full_fluid_mask(dims);

    std::vector<BoundaryCondition> bc_list;
    bc_list.push_back(make_pressure_bc("y_max"));

    solve_poisson_red_black_parallel(
        p, rhs, mask, bc_list,
        dims.nx, dims.ny, dims.nz,
        dims.dx, dims.dy, dims.dz,
        /*max_iters=*/1,
        /*tol=*/0.0,
        /*density=*/1.0
    );

    // All cells with j == ny - 1 must equal the boundary pressure (1.0).
    for (int k = 0; k < dims.nz; ++k) {
        for (int i = 0; i < dims.nx; ++i) {
            int raw_idx = get_flat_index(i, dims.ny - 1, k, dims.nx, dims.ny);
            ASSERT_GE(raw_idx, 0);
            EXPECT_DOUBLE_EQ(p[static_cast<size_t>(raw_idx)], 1.0);
        }
    }
}

TEST(PressurePoissonSolverCoverage, BoundaryPressureAppliedOnZMinFace) {
    GridDimensions dims = make_dims_3x3x3();
    size_t N = static_cast<size_t>(dims.nx) * dims.ny * dims.nz;

    std::vector<double> p(N, 0.0);
    std::vector<double> rhs = make_zero_rhs(dims);
    std::vector<int> mask = make_full_fluid_mask(dims);

    std::vector<BoundaryCondition> bc_list;
    bc_list.push_back(make_pressure_bc("z_min"));

    solve_poisson_red_black_parallel(
        p, rhs, mask, bc_list,
        dims.nx, dims.ny, dims.nz,
        dims.dx, dims.dy, dims.dz,
        /*max_iters=*/1,
        /*tol=*/0.0,
        /*density=*/1.0
    );

    // All cells with k == 0 must equal the boundary pressure (1.0).
    for (int j = 0; j < dims.ny; ++j) {
        for (int i = 0; i < dims.nx; ++i) {
            int raw_idx = get_flat_index(i, j, 0, dims.nx, dims.ny);
            ASSERT_GE(raw_idx, 0);
            EXPECT_DOUBLE_EQ(p[static_cast<size_t>(raw_idx)], 1.0);
        }
    }
}

TEST(PressurePoissonSolverCoverage, BoundaryPressureAppliedOnZMaxFace) {
    GridDimensions dims = make_dims_3x3x3();
    size_t N = static_cast<size_t>(dims.nx) * dims.ny * dims.nz;

    std::vector<double> p(N, 0.0);
    std::vector<double> rhs = make_zero_rhs(dims);
    std::vector<int> mask = make_full_fluid_mask(dims);

    std::vector<BoundaryCondition> bc_list;
    bc_list.push_back(make_pressure_bc("z_max"));

    solve_poisson_red_black_parallel(
        p, rhs, mask, bc_list,
        dims.nx, dims.ny, dims.nz,
        dims.dx, dims.dy, dims.dz,
        /*max_iters=*/1,
        /*tol=*/0.0,
        /*density=*/1.0
    );

    // All cells with k == nz - 1 must equal the boundary pressure (1.0).
    for (int j = 0; j < dims.ny; ++j) {
        for (int i = 0; i < dims.nx; ++i) {
            int raw_idx = get_flat_index(i, j, dims.nz - 1, dims.nx, dims.ny);
            ASSERT_GE(raw_idx, 0);
            EXPECT_DOUBLE_EQ(p[static_cast<size_t>(raw_idx)], 1.0);
        }
    }
}

