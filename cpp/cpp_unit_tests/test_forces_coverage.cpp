
/**
 * @file test_forces_coverage.cpp
 * @brief Literate Test Suite achieving 100% coverage for forces.cpp.
 *
 * This suite follows the Literate Testing Standard:
 * narrative explanations appear as commented prose,
 * while numerical checks and assertions appear as executable code.
 */

#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <stdexcept>
#include <cmath>
#include <limits>
#include "forces.hpp"

// ============================================================================
// SECTION 1 — Contract Violation: Wrong Vector Size
// ============================================================================
//
// WHAT:
//   validate_and_get_forces() requires exactly 3 components (Fx, Fy, Fz).
//
// WHY:
//   The solver must reject malformed force vectors early to prevent
//   undefined behavior in downstream physics kernels.
//
// NARRATIVE:
//   We intentionally pass a vector of size 2. The function must print a
//   contract violation message and throw std::invalid_argument.
//

TEST(ForcesCoverageTest, InvalidVectorSize) {
    // We define a malformed force vector with only 2 components.
    std::vector<double> forces = {1.0, 2.0};

    // The expected behavior is an immediate contract violation.
    EXPECT_THROW({
        navier_stokes_solver::validate_and_get_forces(forces);
    }, std::invalid_argument);
}

// ============================================================================
// SECTION 2 — Numerical Audit: Non‑Finite Force Component
// ============================================================================
//
// WHAT:
//   validate_and_get_forces() scans all 3 components for finiteness.
//
// WHY:
//   Non‑finite forces (NaN, Inf) would corrupt the momentum update step.
//
// NARRATIVE:
//   We inject a NaN into Fx. The solver must detect this and throw
//   std::runtime_error.
//

TEST(ForcesCoverageTest, NonFiniteForceComponent) {
    // We define a force vector where Fx is NaN.
    std::vector<double> forces = {
        std::numeric_limits<double>::quiet_NaN(),  // Fx
        1.0,                                       // Fy
        2.0                                        // Fz
    };

    // The numerical audit must reject this input.
    EXPECT_THROW({
        navier_stokes_solver::validate_and_get_forces(forces);
    }, std::runtime_error);
}

// ============================================================================
// SECTION 3 — Valid Force Vector: Successful Retrieval
// ============================================================================
//
// WHAT:
//   When all three components are finite and the vector size is correct,
//   validate_and_get_forces() must return the exact 3‑component array.
//
// WHY:
//   This is the main success path of the function.
//
// NARRATIVE:
//   We define a physically reasonable force vector (Fx, Fy, Fz).
//   The solver must return {Fx, Fy, Fz} unchanged.
//

TEST(ForcesCoverageTest, ValidForceVector) {
    // We define a valid force vector.
    std::vector<double> forces = {10.0, -5.0, 3.14};

    // We retrieve the validated forces.
    std::array<double,3> out =
        navier_stokes_solver::validate_and_get_forces(forces);

    // We assert exact equality.
    EXPECT_DOUBLE_EQ(out[0], 10.0);
    EXPECT_DOUBLE_EQ(out[1], -5.0);
    EXPECT_DOUBLE_EQ(out[2], 3.14);
}

// ============================================================================
// SECTION 4 — Extreme Finite Values (Stress Test)
// ============================================================================
//
// WHAT:
//   Ensure the function accepts extremely large but finite values.
//
// WHY:
//   CFD simulations often involve large body forces (e.g., buoyancy terms).
//
// NARRATIVE:
//   We use ±1e308 (largest finite IEEE‑754 double). These must pass the
//   finiteness audit and be returned unchanged.
//

TEST(ForcesCoverageTest, ExtremeFiniteValues) {
    // We define extremely large but finite forces.
    std::vector<double> forces = {
        1.0e308,   // Fx
        -1.0e308,  // Fy
        5.0e307    // Fz
    };

    // The solver must accept these values.
    std::array<double,3> out =
        navier_stokes_solver::validate_and_get_forces(forces);

    EXPECT_DOUBLE_EQ(out[0], 1.0e308);
    EXPECT_DOUBLE_EQ(out[1], -1.0e308);
    EXPECT_DOUBLE_EQ(out[2], 5.0e307);
}

// ============================================================================
// SECTION 5 — Negative Forces (Valid Case)
// ============================================================================
//
// WHAT:
//   Negative forces are physically valid (e.g., drag, gravity).
//
// WHY:
//   The solver must not reject negative values.
//
// NARRATIVE:
//   We define a force vector with all negative components.
//   The solver must return them unchanged.
//

TEST(ForcesCoverageTest, NegativeForcesValid) {
    std::vector<double> forces = {-9.81, -0.5, -100.0};

    std::array<double,3> out =
        navier_stokes_solver::validate_and_get_forces(forces);

    EXPECT_DOUBLE_EQ(out[0], -9.81);
    EXPECT_DOUBLE_EQ(out[1], -0.5);
    EXPECT_DOUBLE_EQ(out[2], -100.0);
}

