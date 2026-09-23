/**
 * @file predictor.cpp
 * @brief Implementation of Step 1 Predictor Trial Velocity Computation with 3D Gravity Integration and heavy execution tracing.
 */

#include "predictor.hpp"
#include "advection.hpp"
#include "laplacian.hpp"
#include "grid_math.hpp"
#include <stdexcept>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace navier_stokes_solver {

void validate_inputs(
    const GridDimensions& dims,
    const FluidProperties& fluid,
    double dt,
    const double* u, const double* v, const double* w,
    const double* fx, const double* fy, const double* fz,
    const std::vector<double>& gravity,
    const std::vector& p,
    const std::vector<int>& mask,
    const double* u_star, const double* v_star, const double* w_star
) {
    std::cout << "[PREDICTOR_TRACE] Entering validate_inputs...\n";
    if (!u || !v || !w || !fx || !fy || !fz || !u_star || !v_star || !w_star) {
        std::cout << "[PREDICTOR_TRACE_ERROR] Null pointer supplied to predictor module.\n";
        throw std::invalid_argument("CONTRACT VIOLATION: Null pointer supplied to predictor module.");
    }
    if (gravity.size() != 3) {
        std::cout << "[PREDICTOR_TRACE_ERROR] gravity vector size mismatch: " << gravity.size() << "\n";
        throw std::invalid_argument("CONTRACT VIOLATION: gravity vector must contain exactly 3 components [gx, gy, gz].");
    }
    const size_t total_cells = static_cast<size_t>(dims.nx) * dims.ny * dims.nz;
    if (mask.size() != total_cells) {
        std::cout << "[PREDICTOR_TRACE_ERROR] Mask size mismatch: " << mask.size() << " vs expected " << total_cells << "\n";
        throw std::invalid_argument("CONTRACT VIOLATION: Mask vector size does not match grid dimensions.");
    }
    if (p.size() != total_cells) {
        std::cout << "[PREDICTOR_TRACE_ERROR] Pressure size mismatch: " << p.size() << " vs expected " << total_cells << "\n";
        throw std::invalid_argument("CONTRACT VIOLATION: Pressure vector size does not match grid dimensions.");
    }
    if (dims.nx < 3 || dims.ny < 3 || dims.nz < 3) {
        std::cout << "[PREDICTOR_TRACE_ERROR] Grid dimensions too small: " << dims.nx << "x" << dims.ny << "x" << dims.nz << "\n";
        throw std::invalid_argument("GEOMETRY ERROR: Grid dimensions must be at least 3x3x3 for central stencils.");
    }
    if (dims.dx <= 0.0 || dims.dy <= 0.0 || dims.dz <= 0.0) {
        std::cout << "[PREDICTOR_TRACE_ERROR] Non-positive grid spacing (dx, dy, dz).\n";
        throw std::invalid_argument("GEOMETRY ERROR: Grid spacing (dx, dy, dz) must be strictly positive.");
    }
    if (dt <= 0.0) {
        std::cout << "[PREDICTOR_TRACE_ERROR] Non-positive time step dt: " << dt << "\n";
        throw std::invalid_argument("TEMPORAL ERROR: Time step dt must be strictly positive.");
    }
    if (fluid.nu < 0.0) {
        std::cout << "[PREDICTOR_TRACE_ERROR] Negative viscosity nu: " << fluid.nu << "\n";
        throw std::invalid_argument("PHYSICS ERROR: Kinematic viscosity nu cannot be negative.");
    }
    if (fluid.density <= 0.0) {
        std::cout << "[PREDICTOR_TRACE_ERROR] Non-positive fluid density: " << fluid.density << "\n";
        throw std::invalid_argument("PHYSICS ERROR: Fluid density must be strictly positive.");
    }
    std::cout << "[PREDICTOR_TRACE] validate_inputs passed successfully.\n";
}

void compute_trial_velocities(
    const GridDimensions& dims,
    const FluidProperties& fluid,
    double dt,
    const double* u, const double* v, const double* w,
    const double* fx, const double* fy, const double* fz,
    const std::vector<double>& gravity,
    const std::vector& p,
    const std::vector<int>& mask,
    double* u_star, double* v_star, double* w_star
) {
    std::cout << "[PREDICTOR_TRACE] Entering compute_trial_velocities...\n";
    validate_inputs(dims, fluid, dt, u, v, w, fx, fy, fz, gravity, p, mask, u_star, v_star, w_star);

    const size_t nx = dims.nx;
    const size_t ny = dims.ny;
    const size_t nz = dims.nz;
    const size_t total_cells = nx * ny * nz;

    #ifdef _OPENMP
    int active_threads = omp_get_max_threads();
    #else
    int active_threads = 1;
    #endif

    std::cout << "[THREAD_TRACE] File: predictor.cpp | Operations (Cells): " << total_cells 
              << " | Grid: " << dims.nx << "x" << dims.ny << "x" << dims.nz 
              << " | Active Threads: " << active_threads << "\n";

    // Helper lambdas for tracking max absolute values across fields
    auto max_abs = [](const double* ptr, size_t size) {
        double m = 0.0;
        for (size_t i = 0; i < size; ++i) {
            m = std::max(m, std::abs(ptr[i]));
        }
        return m;
    };

    auto vec_max_abs = [](const std::vector<double>& vec) {
        double m = 0.0;
        for (double val : vec) {
            m = std::max(m, std::abs(val));
        }
        return m;
    };

    std::cout << "[PRESTEP_TRACE] Initial input max abs -> u: " << max_abs(u, total_cells)
              << ", v: " << max_abs(v, total_cells)
              << ", w: " << max_abs(w, total_cells)
              << ", fx: " << max_abs(fx, total_cells)
              << ", fy: " << max_abs(fy, total_cells)
              << ", fz: " << max_abs(fz, total_cells) << "\n";

    const int Nx_int = static_cast<int>(nx);
    const int Ny_int = static_cast<int>(ny);
    const int Nz_int = static_cast<int>(nz);

    // 1. Copy current state to star fields as baseline.
    // This automatically preserves all pre-step Dirichlet boundary values (mask == -1) 
    // and solid states (mask == 0) without corruption or uninitialized garbage.
    std::copy(u, u + total_cells, u_star);
    std::copy(v, v + total_cells, v_star);
    std::copy(w, w + total_cells, w_star);

    std::cout << "[PREDICTOR_TRACE] Step 1 Complete: Copied baseline to star fields. Max abs -> u_star: " 
              << max_abs(u_star, total_cells) << ", v_star: " << max_abs(v_star, total_cells) 
              << ", w_star: " << max_abs(w_star, total_cells) << "\n";

    // 2. Allocate temporary field buffers for advection and Laplacian terms
    std::vector<double> adv_u(total_cells, 0.0);
    std::vector<double> adv_v(total_cells, 0.0);
    std::vector<double> adv_w(total_cells, 0.0);

    std::vector<double> lap_u(total_cells, 0.0);
    std::vector<double> lap_v(total_cells, 0.0);
    std::vector<double> lap_w(total_cells, 0.0);

    std::cout << "[PREDICTOR_TRACE] Step 2 Complete: Allocated temporary buffers for advection and Laplacian.\n";

    // 3. Compute domain-wide advection fields using repository operators
    compute_advection(u, v, w, u, adv_u.data(), Nx_int, Ny_int, Nz_int, dims.dx, dims.dy, dims.dz);
    compute_advection(u, v, w, v, adv_v.data(), Nx_int, Ny_int, Nz_int, dims.dx, dims.dy, dims.dz);
    compute_advection(u, v, w, w, adv_w.data(), Nx_int, Ny_int, Nz_int, dims.dx, dims.dy, dims.dz);

    std::cout << "[PREDICTOR_TRACE] Step 3 Complete: Advection computed. Max abs -> adv_u: " 
              << vec_max_abs(adv_u) << ", adv_v: " << vec_max_abs(adv_v) 
              << ", adv_w: " << vec_max_abs(adv_w) << "\n";

    // 4. Compute domain-wide Laplacian fields using repository operators
    compute_laplacian(u, lap_u.data(), Nx_int, Ny_int, Nz_int, dims.dx, dims.dy, dims.dz);
    compute_laplacian(v, lap_v.data(), Nx_int, Ny_int, Nz_int, dims.dx, dims.dy, dims.dz);
    compute_laplacian(w, lap_w.data(), Nx_int, Ny_int, Nz_int, dims.dx, dims.dy, dims.dz);

    std::cout << "[PREDICTOR_TRACE] Step 4 Complete: Laplacian computed. Max abs -> lap_u: " 
              << vec_max_abs(lap_u) << ", lap_v: " << vec_max_abs(lap_v) 
              << ", lap_w: " << vec_max_abs(lap_w) << "\n";

    // 5. Parallel Temporal Integration (Forward-Euler Predictor Step)
    // Executed STRICTLY on active fluid cells (mask == 1) to respect physical constraints.
    bool has_non_finite = false;
    const double gx = gravity[0];
    const double gy = gravity[1];
    const double gz = gravity[2];

    std::cout << "[PREDICTOR_TRACE] Step 5 Starting: Temporal integration with gravity [" 
              << gx << ", " << gy << ", " << gz << "] and dt = " << dt << "\n";

    #pragma omp parallel for collapse(3) schedule(static) if(total_cells > 1000) reduction(||:has_non_finite)
    for (int i = 0; i < Nx_int; ++i) {
        for (int j = 0; j < Ny_int; ++j) {
            for (int k = 0; k < Nz_int; ++k) {
                const size_t idx = static_cast<size_t>(get_flat_index(i, j, k, Nx_int, Ny_int));

                if (mask[idx] != 1) continue; // Skip non-fluid cells (boundaries and solids)

                // Mask-aware neighbor lookups for robust pressure gradient computation
                const int w_idx = get_flat_index(i - 1, j, k, Nx_int, Ny_int);
                const int e_idx = get_flat_index(i + 1, j, k, Nx_int, Ny_int);
                const int s_idx = get_flat_index(i, j - 1, k, Nx_int, Ny_int);
                const int n_idx = get_flat_index(i, j + 1, k, Nx_int, Ny_int);
                const int d_idx = get_flat_index(i, j, k - 1, Nx_int, Ny_int);
                const int u_idx = get_flat_index(i, j, k + 1, Nx_int, Ny_int);

                double dp_dx = 0.0;
                bool has_west = (w_idx >= 0 && mask[static_cast(w_idx)] == 1);
                bool has_east = (e_idx >= 0 && mask[static_cast(e_idx)] == 1);
                if (has_west && has_east) {
                    dp_dx = (p[static_cast(e_idx)] - p[static_cast(w_idx)]) / (2.0 * dims.dx);
                } else if (has_east) {
                    dp_dx = (p[static_cast(e_idx)] - p[idx]) / dims.dx;
                } else if (has_west) {
                    dp_dx = (p[idx] - p[static_cast(w_idx)]) / dims.dx;
                }

                double dp_dy = 0.0;
                bool has_south = (s_idx >= 0 && mask[static_cast(s_idx)] == 1);
                bool has_north = (n_idx >= 0 && mask[static_cast(n_idx)] == 1);
                if (has_south && has_north) {
                    dp_dy = (p[static_cast(n_idx)] - p[static_cast(s_idx)]) / (2.0 * dims.dy);
                } else if (has_north) {
                    dp_dy = (p[static_cast(n_idx)] - p[idx]) / dims.dy;
                } else if (has_south) {
                    dp_dy = (p[idx] - p[static_cast(s_idx)]) / dims.dy;
                }

                double dp_dz = 0.0;
                bool has_down = (d_idx >= 0 && mask[static_cast(d_idx)] == 1);
                bool has_up = (u_idx >= 0 && mask[static_cast(u_idx)] == 1);
                if (has_down && has_up) {
                    dp_dz = (p[static_cast(u_idx)] - p[static_cast(d_idx)]) / (2.0 * dims.dz);
                } else if (has_up) {
                    dp_dz = (p[static_cast(u_idx)] - p[idx]) / dims.dz;
                } else if (has_down) {
                    dp_dz = (p[idx] - p[static_cast(d_idx)]) / dims.dz;
                }

                double u_t = u[idx] + dt * (-adv_u[idx] + fluid.nu * lap_u[idx] + fx[idx] / fluid.density + gx - (1.0 / fluid.density) * dp_dx);
                double v_t = v[idx] + dt * (-adv_v[idx] + fluid.nu * lap_v[idx] + fy[idx] / fluid.density + gy - (1.0 / fluid.density) * dp_dy);
                double w_t = w[idx] + dt * (-adv_w[idx] + fluid.nu * lap_w[idx] + fz[idx] / fluid.density + gz - (1.0 / fluid.density) * dp_dz);

                if (!std::isfinite(u_t) || !std::isfinite(v_t) || !std::isfinite(w_t)) {
                    has_non_finite = true;
                }

                u_star[idx] = u_t;
                v_star[idx] = v_t;
                w_star[idx] = w_t;
            }
        }
    }

    if (has_non_finite) {
        std::cout << "[PREDICTOR_TRACE_ERROR] Math failure: Non-finite trial velocity calculated in predictor.\n";
        throw std::runtime_error("MATH FAILURE: Non-finite trial velocity calculated in predictor.");
    }

    std::cout << "[PREDICTOR_TRACE] Step 5 Complete: Temporal integration finished successfully. Final star max abs -> u_star: " 
              << max_abs(u_star, total_cells) << ", v_star: " << max_abs(v_star, total_cells) 
              << ", w_star: " << max_abs(w_star, total_cells) << "\n";
}

} // namespace navier_stokes_solver
