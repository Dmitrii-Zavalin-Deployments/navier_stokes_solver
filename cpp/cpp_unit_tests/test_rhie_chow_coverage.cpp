
/**
 * @file test_rhie_chow_coverage.cpp
 * @brief Literate Testing Standard — targeted coverage for rhie_chow.cpp
 *
 * This suite raises coverage of:
 *
 *     cpp/src/rhie_chow.cpp   lines 49–51
 *
 * to 100% by explicitly exercising the CONTRACT VIOLATION branch:
 *
 *     if (!mask.empty() && mask.size() != total_cells) {
 *         throw std::invalid_argument("CONTRACT VIOLATION: Mask vector size mismatch...");
 *     }
 *
 * All other logic (X/Y/Z face interpolation) is already covered by integration tests.
 */

#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>
#include "rhie_chow.hpp"
#include "grid_math.hpp"

using namespace navier_stokes_solver;

/**
 * @section Helper — Construct a minimal GridConfig for a 3×3×3 domain.
 *
 * Rhie–Chow interpolation requires:
 *   - nx, ny, nz
 *   - dx, dy, dz
 *
 * We use unit spacing for simplicity.
 */
static GridConfig make_grid_3x3x3() {
    GridConfig cfg;
    cfg.nx = 3;
    cfg.ny = 3;
    cfg.nz = 3;
    cfg.dx = 1.0;
    cfg.dy = 1.0;
    cfg.dz = 1.0;
    return cfg;
}

/**
 * @section Helper — Allocate a vector of size N filled with zeros.
 */
static std::vector<double> make_zero_field(size_t N) {
    return std::vector<double>(N, 0.0);
}

/**
 * @section Helper — Allocate a mask of size N filled with 1 (fluid).
 */
static std::vector<int> make_fluid_mask(size_t N) {
    return std::vector<int>(N, 1);
}

/**
 * @section 1 — CONTRACT VIOLATION: Mask size mismatch
 *
 * The Rhie–Chow interpolator checks:
 *
 *     if (!mask.empty() && mask.size() != total_cells)
 *         throw std::invalid_argument(...)
 *
 * To reach this branch, we intentionally provide:
 *
 *     mask.size() = total_cells - 1
 *
 * while all other fields have correct size.
 *
 * This is the only remaining uncovered branch in rhie_chow.cpp.
 */
TEST(RhieChowCoverage, MaskSizeMismatchThrowsInvalidArgument) {

    // --- Grid configuration: 3×3×3 domain
    GridConfig cfg = make_grid_3x3x3();
    const size_t total_cells =
        static_cast<size_t>(cfg.nx) * cfg.ny * cfg.nz;

    // --- Allocate correct-sized fields
    std::vector<double> u   = make_zero_field(total_cells);
    std::vector<double> v   = make_zero_field(total_cells);
    std::vector<double> w   = make_zero_field(total_cells);
    std::vector<double> p   = make_zero_field(total_cells);
    std::vector<double> a_p = make_zero_field(total_cells);

    // --- Allocate face fields (correct sizes)
    std::vector<double> u_face((cfg.nx - 1) * cfg.ny * cfg.nz, 0.0);
    std::vector<double> v_face(cfg.nx * (cfg.ny - 1) * cfg.nz, 0.0);
    std::vector<double> w_face(cfg.nx * cfg.ny * (cfg.nz - 1), 0.0);

    // --- INTENTIONALLY WRONG mask size: total_cells - 1
    std::vector<int> bad_mask = make_fluid_mask(total_cells - 1);

    // --- Instantiate interpolator
    RhieChowInterpolator interp;

    // --- EXPECT: CONTRACT VIOLATION → std::invalid_argument
    EXPECT_THROW(
        interp.interpolateFaceVelocities(
            u, v, w,
            p,
            a_p,
            bad_mask,   // WRONG SIZE triggers lines 49–51
            cfg,
            u_face,
            v_face,
            w_face
        ),
        std::invalid_argument
    );
}

/**
 * @section 2 — Sanity test: correct mask size does NOT throw
 *
 * This ensures the test suite is well‑formed and the interpolator
 * behaves normally when mask.size() == total_cells.
 */
TEST(RhieChowCoverage, CorrectMaskSizeDoesNotThrow) {

    GridConfig cfg = make_grid_3x3x3();
    const size_t total_cells =
        static_cast<size_t>(cfg.nx) * cfg.ny * cfg.nz;

    std::vector<double> u   = make_zero_field(total_cells);
    std::vector<double> v   = make_zero_field(total_cells);
    std::vector<double> w   = make_zero_field(total_cells);
    std::vector<double> p   = make_zero_field(total_cells);
    std::vector<double> a_p = make_zero_field(total_cells);

    std::vector<double> u_face((cfg.nx - 1) * cfg.ny * cfg.nz, 0.0);
    std::vector<double> v_face(cfg.nx * (cfg.ny - 1) * cfg.nz, 0.0);
    std::vector<double> w_face(cfg.nx * cfg.ny * (cfg.nz - 1), 0.0);

    std::vector<int> good_mask = make_fluid_mask(total_cells);

    RhieChowInterpolator interp;

    EXPECT_NO_THROW(
        interp.interpolateFaceVelocities(
            u, v, w,
            p,
            a_p,
            good_mask,
            cfg,
            u_face,
            v_face,
            w_face
        )
    );
}

