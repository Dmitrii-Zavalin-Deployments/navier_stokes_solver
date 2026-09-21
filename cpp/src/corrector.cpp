/**
 * @file corrector.cpp
 * @brief Implementation of Step 4 Corrector Velocity Projection for collocated grids with
 *        robust boundary-conforming pressure gradients and stabilized interior updates.
 */

#include "corrector.hpp"
#include "grid_math.hpp"
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace navier_stokes_solver {

void solve_corrector_parallel(
    std::vector<double>& u,
    std::vector<double>& v,
    std::vector<double>& w,
    const std::vector<double>& u_star,
    const std::vector<double>& v_star,
    const std::vector<double>& w_star,
    const std::vector<double>& p,
    const std::vector<int>& mask,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double dt, double rho
) {
    if (nx < 3 || ny < 3 || nz < 3) {
        throw std::invalid_argument("GEOMETRY ERROR: Grid dimensions must be at least 3x3x3 for corrector projection.");
    }
    if (dx <= 0.0 || dy <= 0.0 || dz <= 0.0) {
        throw std::invalid_argument("GEOMETRY ERROR: Grid spacing must be strictly positive.");
    }
    if (dt <= 0.0 || rho <= 0.0) {
        throw std::invalid_argument("PHYSICS ERROR: Time step dt and density rho must be strictly positive.");
    }

    const size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    if (u.size() != total_cells || v.size() != total_cells || w.size() != total_cells ||
        u_star.size() != total_cells || v_star.size() != total_cells || w_star.size() != total_cells ||
        p.size() != total_cells || mask.size() != total_cells) {
        throw std::invalid_argument("CONTRACT VIOLATION: Vector size mismatch in corrector module.");
    }

    #ifdef _OPENMP
    int active_threads = omp_get_max_threads();
    #else
    int active_threads = 1;
    #endif

    std::cout << "[THREAD_TRACE] File: corrector.cpp | Operations (Cells): " << total_cells 
              << " | Grid: " << nx << "x" << ny << "x" << nz 
              << " | Active Threads: " << active_threads << "\n";

    const double coeff = dt / rho;
    const double idx_2inv = 0.5 / dx;
    const double idy_2inv = 0.5 / dy;
    const double idz_2inv = 0.5 / dz;
    const double id_inv  = 1.0 / dx;
    const double idy_inv = 1.0 / dy;
    const double idz_inv = 1.0 / dz;

    bool has_error = false;
    int err_i = 0, err_j = 0, err_k = 0;
    double err_u = 0.0, err_v = 0.0, err_w = 0.0;

    // Execute corrector step strictly on active interior fluid cells (mask == 1)
    #pragma omp parallel for collapse(3) schedule(static) if(total_cells > 1000)
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                
                const size_t idx_cell = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
                
                // Skip solid cells (mask == 0) and wall boundaries (mask == -1)
                if (mask[idx_cell] != 1) continue;

                // Neighbor indices
                const size_t idx_west  = static_cast<size_t>(get_flat_index(i - 1, j, k, nx, ny));
                const size_t idx_east  = static_cast<size_t>(get_flat_index(i + 1, j, k, nx, ny));
                const size_t idx_south = static_cast<size_t>(get_flat_index(i, j - 1, k, nx, ny));
                const size_t idx_north = static_cast<size_t>(get_flat_index(i, j + 1, k, nx, ny));
                const size_t idx_down  = static_cast<size_t>(get_flat_index(i, j, k - 1, nx, ny));
                const size_t idx_up    = static_cast<size_t>(get_flat_index(i, j, k + 1, nx, ny));

                const double p_center = p[idx_cell];
                const double p_west  = p[idx_west];
                const double p_east  = p[idx_east];
                const double p_south = p[idx_south];
                const double p_north = p[idx_north];
                const double p_down  = p[idx_down];
                const double p_up    = p[idx_up];

                // --- HEAVY TRACE: BLOCK 1 - Pressure & Neighbors (Sampled at [1,1,1]) ---
                if (i == 1 && j == 1 && k == 1) {
                    #pragma omp critical
                    {
                        std::cout << "[CORRECTOR_TRACE] Cell [" << i << "," << j << "," << k << "] - Block 1 (Pressure & Neighbors):\n"
                                  << "  p_center: " << p_center << ", p_west: " << p_west << ", p_east: " << p_east << "\n"
                                  << "  p_south: " << p_south << ", p_north: " << p_north << ", p_down: " << p_down << ", p_up: " << p_up << "\n";
                    }
                }

                // --- ROBUST MASK-AWARE PRESSURE GRADIENT EVALUATION ---
                double dp_dx = 0.0;
                if (mask[idx_east] == 1 && mask[idx_west] == 1) {
                    dp_dx = (p_east - p_west) * idx_2inv; // 2nd-order interior central difference
                } else if ((mask[idx_east] == 0 || mask[idx_east] == -1) && mask[idx_west] == 1) {
                    dp_dx = (p_east - p_center) * id_inv; // Boundary-conforming gradient (east wall)
                } else if (mask[idx_east] == 1 && (mask[idx_west] == 0 || mask[idx_west] == -1)) {
                    dp_dx = (p_center - p_west) * id_inv; // Boundary-conforming gradient (west wall)
                } else {
                    dp_dx = 0.0;
                }

                double dp_dy = 0.0;
                if (mask[idx_north] == 1 && mask[idx_south] == 1) {
                    dp_dy = (p_north - p_south) * idy_2inv; // 2nd-order interior central difference
                } else if ((mask[idx_north] == 0 || mask[idx_north] == -1) && mask[idx_south] == 1) {
                    dp_dy = (p_north - p_center) * idy_inv; // Boundary-conforming gradient (north wall)
                } else if (mask[idx_north] == 1 && (mask[idx_south] == 0 || mask[idx_south] == -1)) {
                    dp_dy = (p_center - p_south) * idy_inv; // Boundary-conforming gradient (south wall)
                } else {
                    dp_dy = 0.0;
                }

                double dp_dz = 0.0;
                if (mask[idx_up] == 1 && mask[idx_down] == 1) {
                    dp_dz = (p_up - p_down) * idz_2inv; // 2nd-order interior central difference
                } else if ((mask[idx_up] == 0 || mask[idx_up] == -1) && mask[idx_down] == 1) {
                    dp_dz = (p_up - p_center) * idz_inv; // Boundary-conforming gradient (up wall)
                } else if (mask[idx_up] == 1 && (mask[idx_down] == 0 || mask[idx_down] == -1)) {
                    dp_dz = (p_center - p_down) * idz_inv; // Boundary-conforming gradient (down wall)
                } else {
                    dp_dz = 0.0;
                }

                // --- HEAVY TRACE: BLOCK 2 - Pressure Gradients ---
                if (i == 1 && j == 1 && k == 1) {
                    #pragma omp critical
                    {
                        std::cout << "[CORRECTOR_TRACE] Cell [" << i << "," << j << "," << k << "] - Block 2 (Gradients):\n"
                                  << "  dp_dx: " << dp_dx << ", dp_dy: " << dp_dy << ", dp_dz: " << dp_dz << "\n";
                    }
                }

                // Project trial velocity onto divergence-free subspace
                double new_u = u_star[idx_cell] - coeff * dp_dx;
                double new_v = v_star[idx_cell] - coeff * dp_dy;
                double new_w = w_star[idx_cell] - coeff * dp_dz;

                // --- HEAVY TRACE: BLOCK 3 - Velocity Projection ---
                if (i == 1 && j == 1 && k == 1) {
                    #pragma omp critical
                    {
                        std::cout << "[CORRECTOR_TRACE] Cell [" << i << "," << j << "," << k << "] - Block 3 (Velocity Projection):\n"
                                  << "  u_star: " << u_star[idx_cell] << " -> new_u: " << new_u << "\n"
                                  << "  v_star: " << v_star[idx_cell] << " -> new_v: " << new_v << "\n"
                                  << "  w_star: " << w_star[idx_cell] << " -> new_w: " << new_w << "\n";
                    }
                }

                // --- FORENSIC NUMERICAL AUDIT ---
                if (!std::isfinite(new_u) || !std::isfinite(new_v) || !std::isfinite(new_w)) {
                    #pragma omp critical
                    {
                        if (!has_error) {
                            has_error = true;
                            err_i = i;
                            err_j = j;
                            err_k = k;
                            err_u = new_u;
                            err_v = new_v;
                            err_w = new_w;
                        }
                    }
                }

                u[idx_cell] = new_u;
                v[idx_cell] = new_v;
                w[idx_cell] = new_w;
            }
        }
    }

    // --- SOLID VELOCITY CLAMPING PASS ---
    #pragma omp parallel for schedule(static) if(total_cells > 1000)
    for (int64_t idx = 0; idx < static_cast<int64_t>(total_cells); ++idx) {
        if (mask[idx] == 0) {
            u[idx] = 0.0;
            v[idx] = 0.0;
            w[idx] = 0.0;
        }
    }

    // --- HEAVY TRACE: BLOCK 4 - Final Field Max Absolute Values ---
    double max_u = 0.0, max_v = 0.0, max_w = 0.0;
    for (size_t idx = 0; idx < total_cells; ++idx) {
        if (std::abs(u[idx]) > max_u) max_u = std::abs(u[idx]);
        if (std::abs(v[idx]) > max_v) max_v = std::abs(v[idx]);
        if (std::abs(w[idx]) > max_w) max_w = std::abs(w[idx]);
    }
    std::cout << "[CORRECTOR_TRACE] Step Complete. Final field max absolute values -> u: " 
              << max_u << ", v: " << max_v << ", w: " << max_w << "\n";

    if (has_error) {
        std::cerr << "MATH FAILURE [corrector.cpp]: Non-finite velocity projected at grid index [" 
                  << err_i << ", " << err_j << ", " << err_k << "] | "
                  << "Vel: [" << err_u << ", " << err_v << ", " << err_w << "]\n";
        throw std::runtime_error("Corrector projection exploded. Velocity field is non-finite.");
    }
}

} // namespace navier_stokes_solver
