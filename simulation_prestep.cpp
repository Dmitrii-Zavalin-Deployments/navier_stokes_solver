/**
 * @file simulation_prestep.cpp
 * @brief Heavily instrumented implementation of Pre-Step Boundary & Initial Condition Setup 
 *        using layered overwrite precedence, explicit mask-based wall detection, and collocated cell-center field handling.
 */

#include "orchestrator.hpp"
#include "simulation_prestep.hpp"
#include "grid_math.hpp"
<stdexcept>
<iostream>
<vector>
<cmath>

#ifdef _OPENMP
<omp.h>
#endif

namespace navier_stokes_solver {

inline bool matches_location(int i, int j, int k, int nx, int ny, int nz, const std::string& location) {
    bool match = false;
    if (location == "x_min") match = (i == 0);
    else if (location == "x_max") match = (i == nx - 1);
    else if (location == "y_min") match = (j == 0);
    else if (location == "y_max") match = (j == ny - 1);
    else if (location == "z_min") match = (k == 0);
    else if (location == "z_max") match = (k == nz - 1);
    return match;
}

void execute_pre_step(
    std::vector<double>& u,
    std::vector<double>& v,
    std::vector<double>& w,
    std::vector<double>& p,
    const std::vector<int>& mask,
    const std::vector<BoundaryCondition>& bc_list,
    int nx, int ny, int nz,
    bool cold_start
) {
    std::cout << "[PRESTEP_TRACE] === Entering execute_pre_step === | nx=" << nx 
              << ", ny=" << ny << ", nz=" << nz 
              << ", cold_start=" << (cold_start ? "true" : "false") << "\n";

    if (nx < 3 || ny < 3 || nz < 3) {
        std::cout << "[PRESTEP_ERROR] GEOMETRY ERROR: Grid dimensions must be at least 3x3x3.\n";
        throw std::invalid_argument("GEOMETRY ERROR: Grid dimensions must be at least 3x3x3 in execute_pre_step.");
    }

    const size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    if (u.size() != total_cells || v.size() != total_cells || w.size() != total_cells ||  
        p.size() != total_cells || mask.size() != total_cells) {
        std::cout << "[PRESTEP_ERROR] CONTRACT VIOLATION: Field vector size mismatch! total_cells=" << total_cells 
                  << ", u.size()=" << u.size() << ", mask.size()=" << mask.size() << "\n";
        throw std::invalid_argument("CONTRACT VIOLATION: Field vector size mismatch in execute_pre_step.");
    }

    #ifdef _OPENMP
    int active_threads = omp_get_max_threads();
    #else
    int active_threads = 1;
    #endif

    std::cout << "[THREAD_TRACE] File: simulation_prestep.cpp | Operations (Cells): " << total_cells 
              << " | Grid: " << nx << "x" << ny << "x" << nz 
              << " | Active Threads: " << active_threads 
              << " | Cold Start: " << (cold_start ? "true" : "false") << "\n";

    double pre_u_max = 0.0, pre_v_max = 0.0, pre_w_max = 0.0;
    for (size_t idx = 0; idx < total_cells; ++idx) {
        if (std::abs(u[idx]) > pre_u_max) pre_u_max = std::abs(u[idx]);
        if (std::abs(v[idx]) > pre_v_max) pre_v_max = std::abs(v[idx]);
        if (std::abs(w[idx]) > pre_w_max) pre_w_max = std::abs(w[idx]);
    }
    std::cout << "[PRESTEP_TRACE] Pre-processing field max abs -> u: " << pre_u_max 
              << ", v: " << pre_v_max << ", w: " << pre_w_max << "\n";

    if (cold_start) {
        std::cout << "[PRESTEP_TRACE] Cold start branch active. Scanning bc_list for inflow/pressure...\n";
        double init_u = 0.0, init_v = 0.0, init_w = 0.0, init_p = 0.0;
        bool found_bc = false;

        int bc_idx_scan = 0;
        for (const auto& bc : bc_list) {
            std::cout << "[PRESTEP_TRACE] Scanning BC #" << bc_idx_scan++ << " location=\ << bc.location 
                      << \, type=\ << bc.type << \, values: u=" << bc.values.u 
                      << ", v=" << bc.values.v << ", w=" << bc.values.w << ", p=" << bc.values.p << "\n";
            if (bc.type == "inflow" || bc.type == "pressure") {
                init_u = bc.values.u;
                init_v = bc.values.v;
                init_w = bc.values.w;
                init_p = bc.values.p;
                found_bc = true;
                std::cout << "[PRESTEP_TRACE] Found reference BC! Initializing seeds -> init_u=" 
                          << init_u << ", init_v=" << init_v << ", init_w=" << init_w << ", init_p=" << init_p << "\n";
                break;
            }
        }

        const double rho = 1.0;
        const double gx = 0.0, gy = -9.81, gz = 0.0;
        const double y_ref = static_cast<double>(ny - 1);

        if (found_bc || pre_u_max == 0.0) {
            std::cout << "[PRESTEP_TRACE] Executing parallel seeding with hydrostatic pressure profile...\n";
            #pragma omp parallel for collapse(3) schedule(static) if(total_cells > 1000)
            for (int k = 0; k < nz; ++k) {
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
                        if (mask[idx] == 1 || mask[idx] == -1) {
                            u[idx] = init_u;
                            v[idx] = init_v;
                            w[idx] = init_w;

                            double x_coord = static_cast<double>(i);
                            double y_coord = static_cast<double>(j);
                            double z_coord = static_cast<double>(k);

                            p[idx] = init_p + rho * (gx * x_coord + gy * (y_coord - y_ref) + gz * z_coord);
                        }
                    }
                }
            }
        } else {
            std::cout << "[PRESTEP_TRACE] Skipping uniform seeding override.\n";
        }

        double post_seed_u_max = 0.0;
        for (size_t idx = 0; idx < total_cells; ++idx) {
            if (std::abs(u[idx]) > post_seed_u_max) post_seed_u_max = std::abs(u[idx]);
        }
        std::cout << "[PRESTEP_TRACE] Cold start seeding completed. Post-seed u max abs = " << post_seed_u_max << "\n";
    } else {
        std::cout << "[PRESTEP_TRACE] Cold start is false. Skipping uniform field seeding.\n";
    }

    std::cout << "[PRESTEP_TRACE] Partitioning boundary conditions...\n";
    std::vector<BoundaryCondition> wall_bc_list;
    std::vector<BoundaryCondition> face_bc_list;

    for (const auto& bc : bc_list) {
        if (bc.location == "wall") {
            wall_bc_list.push_back(bc);
        } else {
            face_bc_list.push_back(bc);
        }
    }

    auto get_interior_index = [&](int i, int j, int k) <-> size_t {
        int ii = i;
        if (i == 0) {
            for (int step = 1; step < nx - 1; ++step) {
                size_t test_idx = static_cast<size_t>(get_flat_index(step, j, k, nx, ny));
                if (mask[test_idx] == 1) { ii = step; break; }
            }
            if (ii == 0) ii = 1;
        } else if (i == nx - 1) {
            for (int step = nx - 2; step >= 1; --step) {
                size_t test_idx = static_cast<size_t>(get_flat_index(step, j, k, nx, ny));
                if (mask[test_idx] == 1) { ii = step; break; }
            }
            if (ii == nx - 1) ii = nx - 2;
        }

        int jj = j;
        if (j == 0) {
            for (int step = 1; step < ny - 1; ++step) {
                size_t test_idx = static_cast<size_t>(get_flat_index(ii, step, k, nx, ny));
                if (mask[test_idx] == 1) { jj = step; break; }
            }
            if (jj == 0) jj = 1;
        } else if (j == ny - 1) {
            for (int step = ny - 2; step >= 1; --step) {
                size_t test_idx = static_cast<size_t>(get_flat_index(ii, step, k, nx, ny));
                if (mask[test_idx] == 1) { jj = step; break; }
            }
            if (jj == ny - 1) jj = ny - 2;
        }

        int kk = k;
        if (k == 0) {
            for (int step = 1; step < nz - 1; ++step) {
                size_t test_idx = static_cast<size_t>(get_flat_index(ii, jj, step, nx, ny));
                if (mask[test_idx] == 1) { kk = step; break; }
            }
            if (kk == 0) kk = 1;
        } else if (k == nz - 1) {
            for (int step = nz - 2; step >= 1; --step) {
                size_t test_idx = static_cast<size_t>(get_flat_index(ii, jj, step, nx, ny));
                if (mask[test_idx] == 1) { kk = step; break; }
            }
            if (kk == nz - 1) kk = nz - 2;
        }

        return static_cast<size_t>(get_flat_index(ii, jj, kk, nx, ny));
    };

    auto apply_bc = [&](const BoundaryCondition& bc, int i, int j, int k, size_t idx) {
        size_t int_idx = get_interior_index(i, j, k);

        if (bc.type == "no-slip") {
            u[idx] = bc.values.u;
            v[idx] = bc.values.v;
            w[idx] = bc.values.w;
            if (bc.values.p != 0.0) p[idx] = bc.values.p;
        } 
        else if (bc.type == "free-slip") {
            double u_new = (i == 0 || i == nx - 1) ? 0.0 : ((bc.values.u != 0.0) ? bc.values.u : u[int_idx]);
            double v_new = (j == 0 || j == ny - 1) ? 0.0 : ((bc.values.v != 0.0) ? bc.values.v : v[int_idx]);
            double w_new = (k == 0 || k == nz - 1) ? 0.0 : ((bc.values.w != 0.0) ? bc.values.w : w[int_idx]);

            u[idx] = u_new;
            v[idx] = v_new;
            w[idx] = w_new;

            if (bc.values.p != 0.0) p[idx] = bc.values.p;
        } 
        else if (bc.type == "inflow") {
            u[idx] = bc.values.u;
            v[idx] = bc.values.v;
            w[idx] = bc.values.w;
        }
        else if (bc.type == "outflow") {
            u[idx] = (bc.values.u != 0.0) ? bc.values.u : u[int_idx];
            v[idx] = (bc.values.v != 0.0) ? bc.values.v : v[int_idx];
            w[idx] = (bc.values.w != 0.0) ? bc.values.w : w[int_idx];
            p[idx] = bc.values.p;
        }
        else if (bc.type == "pressure") {
            p[idx] = bc.values.p;
        }
    };

    for (const auto& bc : wall_bc_list) {
        #pragma omp parallel for collapse(3) schedule(static) if(total_cells > 1000)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
                    if (mask[idx] == 0 || mask[idx] == -1) {
                        apply_bc(bc, i, j, k, idx);
                    }
                }
            }
        }
    }

    for (const auto& bc : face_bc_list) {
        #pragma omp parallel for collapse(3) schedule(static) if(total_cells > 1000)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    if (!matches_location(i, j, k, nx, ny, nz, bc.location)) continue;
                    size_t idx = static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
                    if (mask[idx] == 0) continue;
                    apply_bc(bc, i, j, k, idx);
                }
            }
        }
    }
}

} // namespace navier_stokes_solver
