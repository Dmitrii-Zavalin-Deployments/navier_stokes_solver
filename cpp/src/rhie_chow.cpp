/**
 * @file rhie_chow.cpp
 * @brief Implementation of Rhie-Chow collocated grid velocity interpolation with
 *        robust boundary-conforming cell-centered pressure gradients and zero-flux boundary conditioning.
 */

#include "rhie_chow.hpp"
#include "grid_math.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace navier_stokes_solver {

void RhieChowInterpolator::interpolateFaceVelocities(
    const std::vector<double>& u,
    const std::vector<double>& v,
    const std::vector<double>& w,
    const std::vector<double>& p,
    const std::vector<double>& a_p,
    const std::vector<int>& mask,
    const GridConfig& config,
    std::vector<double>& u_face,
    std::vector<double>& v_face,
    std::vector<double>& w_face
) {
    int nx = config.nx;
    int ny = config.ny;
    int nz = config.nz;
    double dx = config.dx;
    double dy = config.dy;
    double dz = config.dz;

    // Helper lambda for 3D flat indexing using repository standard get_flat_index
    auto get_idx = [nx, ny](int i, int j, int k) {
        return static_cast<size_t>(get_flat_index(i, j, k, nx, ny));
    };

    const size_t total_cells = static_cast<size_t>(nx) * ny * nz;
    if (!mask.empty() && mask.size() != total_cells) {
        throw std::invalid_argument("CONTRACT VIOLATION: Mask vector size mismatch in RhieChowInterpolator.");
    }

    const double id_inv  = 1.0 / dx;
    const double idy_inv = 1.0 / dy;
    const double idz_inv = 1.0 / dz;
    const double idx_2inv = 0.5 / dx;
    const double idy_2inv = 0.5 / dy;
    const double idz_2inv = 0.5 / dz;

    // --- 1. X-Face Velocities ---
    #pragma omp parallel for collapse(3) schedule(static) if(nx * ny * nz > 1000)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx - 1; ++i) {
                size_t idx_P = get_idx(i, j, k);
                size_t idx_E = get_idx(i + 1, j, k);
                size_t face_idx = static_cast<size_t>(i + (nx - 1) * (j + ny * k));

                // Enforce zero normal flux across solid (mask == 0) and wall (mask == -1) boundaries
                if (!mask.empty() && (mask[idx_P] != 1 || mask[idx_E] != 1)) {
                    u_face[face_idx] = 0.0;
                    continue;
                }

                // Linear interpolation of velocity to face
                double u_lin = 0.5 * (u[idx_P] + u[idx_E]);

                // Averaged momentum coefficient at face
                double ap_face = 0.5 * (a_p[idx_P] + a_p[idx_E]);
                double d_face = (ap_face > 0.0) ? (1.0 / ap_face) : 0.0;

                // Sharp pressure gradient at face
                double dp_dx_sharp = (p[idx_E] - p[idx_P]) * id_inv;

                // --- ROBUST MASK-AWARE CELL-CENTERED GRADIENT AT P ---
                double dp_dx_P = 0.0;
                bool has_west = (i > 0);
                bool has_east_P = (i < nx - 1);
                size_t idx_west = has_west ? get_idx(i - 1, j, k) : 0;
                
                bool west_fluid = has_west && (mask.empty() || mask[idx_west] == 1);
                bool east_fluid_P = has_east_P && (mask.empty() || mask[idx_E] == 1);

                if (west_fluid && east_fluid_P) {
                    dp_dx_P = (p[idx_E] - p[idx_west]) * idx_2inv;
                } else if (!west_fluid && east_fluid_P) {
                    dp_dx_P = (p[idx_E] - p[idx_P]) * id_inv;
                } else if (west_fluid && !east_fluid_P) {
                    dp_dx_P = (p[idx_P] - p[idx_west]) * id_inv;
                } else {
                    dp_dx_P = 0.0;
                }

                // --- ROBUST MASK-AWARE CELL-CENTERED GRADIENT AT E ---
                double dp_dx_E = 0.0;
                bool has_west_E = true; // index i (idx_P)
                bool has_east_EE = (i + 2 < nx);
                size_t idx_east_EE = has_east_EE ? get_idx(i + 2, j, k) : 0;

                bool west_fluid_E = mask.empty() || mask[idx_P] == 1;
                bool east_fluid_EE = has_east_EE && (mask.empty() || mask[idx_east_EE] == 1);

                if (west_fluid_E && east_fluid_EE) {
                    dp_dx_E = (p[idx_east_EE] - p[idx_P]) * idx_2inv;
                } else if (!west_fluid_E && east_fluid_EE) {
                    dp_dx_E = (p[idx_east_EE] - p[idx_E]) * id_inv;
                } else if (west_fluid_E && !east_fluid_EE) {
                    dp_dx_E = (p[idx_E] - p[idx_P]) * id_inv;
                } else {
                    dp_dx_E = 0.0;
                }

                double dp_dx_avg = 0.5 * (dp_dx_P + dp_dx_E);

                // Rhie-Chow correction formulation
                u_face[face_idx] = u_lin - d_face * (dp_dx_sharp - dp_dx_avg);
            }
        }
    }

    // --- 2. Y-Face Velocities ---
    #pragma omp parallel for collapse(3) schedule(static) if(nx * ny * nz > 1000)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny - 1; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx_P = get_idx(i, j, k);
                size_t idx_N = get_idx(i, j + 1, k);
                size_t face_idx = static_cast<size_t>(i + nx * (j + (ny - 1) * k));

                // Enforce zero normal flux across solid/wall boundaries
                if (!mask.empty() && (mask[idx_P] != 1 || mask[idx_N] != 1)) {
                    v_face[face_idx] = 0.0;
                    continue;
                }

                double v_lin = 0.5 * (v[idx_P] + v[idx_N]);
                double ap_face = 0.5 * (a_p[idx_P] + a_p[idx_N]);
                double d_face = (ap_face > 0.0) ? (1.0 / ap_face) : 0.0;

                double dp_dy_sharp = (p[idx_N] - p[idx_P]) * idy_inv;

                // --- ROBUST MASK-AWARE CELL-CENTERED GRADIENT AT P ---
                double dp_dy_P = 0.0;
                bool has_south = (j > 0);
                bool has_north_P = (j < ny - 1);
                size_t idx_south = has_south ? get_idx(i, j - 1, k) : 0;

                bool south_fluid = has_south && (mask.empty() || mask[idx_south] == 1);
                bool north_fluid_P = has_north_P && (mask.empty() || mask[idx_N] == 1);

                if (south_fluid && north_fluid_P) {
                    dp_dy_P = (p[idx_N] - p[idx_south]) * idy_2inv;
                } else if (!south_fluid && north_fluid_P) {
                    dp_dy_P = (p[idx_N] - p[idx_P]) * idy_inv;
                } else if (south_fluid && !north_fluid_P) {
                    dp_dy_P = (p[idx_P] - p[idx_south]) * idy_inv;
                } else {
                    dp_dy_P = 0.0;
                }

                // --- ROBUST MASK-AWARE CELL-CENTERED GRADIENT AT N ---
                double dp_dy_N = 0.0;
                bool has_south_N = true; // index j (idx_P)
                bool has_north_NN = (j + 2 < ny);
                size_t idx_north_NN = has_north_NN ? get_idx(i, j + 2, k) : 0;

                bool south_fluid_N = mask.empty() || mask[idx_P] == 1;
                bool north_fluid_NN = has_north_NN && (mask.empty() || mask[idx_north_NN] == 1);

                if (south_fluid_N && north_fluid_NN) {
                    dp_dy_N = (p[idx_north_NN] - p[idx_P]) * idy_2inv;
                } else if (!south_fluid_N && north_fluid_NN) {
                    dp_dy_N = (p[idx_north_NN] - p[idx_N]) * idy_inv;
                } else if (south_fluid_N && !north_fluid_NN) {
                    dp_dy_N = (p[idx_N] - p[idx_P]) * idy_inv;
                } else {
                    dp_dy_N = 0.0;
                }

                double dp_dy_avg = 0.5 * (dp_dy_P + dp_dy_N);

                v_face[face_idx] = v_lin - d_face * (dp_dy_sharp - dp_dy_avg);
            }
        }
    }

    // --- 3. Z-Face Velocities ---
    #pragma omp parallel for collapse(3) schedule(static) if(nx * ny * nz > 1000)
    for (int k = 0; k < nz - 1; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx_P = get_idx(i, j, k);
                size_t idx_T = get_idx(i, j, k + 1);
                size_t face_idx = static_cast<size_t>(i + nx * (j + ny * k));

                // Enforce zero normal flux across solid/wall boundaries
                if (!mask.empty() && (mask[idx_P] != 1 || mask[idx_T] != 1)) {
                    w_face[face_idx] = 0.0;
                    continue;
                }

                double w_lin = 0.5 * (w[idx_P] + w[idx_T]);
                double ap_face = 0.5 * (a_p[idx_P] + a_p[idx_T]);
                double d_face = (ap_face > 0.0) ? (1.0 / ap_face) : 0.0;

                double dp_dz_sharp = (p[idx_T] - p[idx_P]) * idz_inv;

                // --- ROBUST MASK-AWARE CELL-CENTERED GRADIENT AT P ---
                double dp_dz_P = 0.0;
                bool has_down = (k > 0);
                bool has_up_P = (k < nz - 1);
                size_t idx_down = has_down ? get_idx(i, j, k - 1) : 0;

                bool down_fluid = has_down && (mask.empty() || mask[idx_down] == 1);
                bool up_fluid_P = has_up_P && (mask.empty() || mask[idx_T] == 1);

                if (down_fluid && up_fluid_P) {
                    dp_dz_P = (p[idx_T] - p[idx_down]) * idz_2inv;
                } else if (!down_fluid && up_fluid_P) {
                    dp_dz_P = (p[idx_T] - p[idx_P]) * idz_inv;
                } else if (down_fluid && !up_fluid_P) {
                    dp_dz_P = (p[idx_P] - p[idx_down]) * idz_inv;
                } else {
                    dp_dz_P = 0.0;
                }

                // --- ROBUST MASK-AWARE CELL-CENTERED GRADIENT AT T ---
                double dp_dz_T = 0.0;
                bool has_down_T = true; // index k (idx_P)
                bool has_up_TT = (k + 2 < nz);
                size_t idx_up_TT = has_up_TT ? get_idx(i, j, k + 2) : 0;

                bool down_fluid_T = mask.empty() || mask[idx_P] == 1;
                bool up_fluid_TT = has_up_TT && (mask.empty() || mask[idx_up_TT] == 1);

                if (down_fluid_T && up_fluid_TT) {
                    dp_dz_T = (p[idx_up_TT] - p[idx_P]) * idz_2inv;
                } else if (!down_fluid_T && up_fluid_TT) {
                    dp_dz_T = (p[idx_up_TT] - p[idx_T]) * idz_inv;
                } else if (down_fluid_T && !up_fluid_TT) {
                    dp_dz_T = (p[idx_T] - p[idx_P]) * idz_inv;
                } else {
                    dp_dz_T = 0.0;
                }

                double dp_dz_avg = 0.5 * (dp_dz_P + dp_dz_T);

                w_face[face_idx] = w_lin - d_face * (dp_dz_sharp - dp_dz_avg);
            }
        }
    }
}

} // namespace navier_stokes_solver
