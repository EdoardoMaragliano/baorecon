# utils.py

"""Misc numerical helpers (density<->overdensity conversions, smoothing-radius
relations, box splitting and periodic distance)."""

import numpy as np


def rho_to_delta(rho, mean_density):
    """Overdensity field from a density field: ``delta = rho/mean - 1``."""
    return rho / mean_density - 1.0


def delta_to_rho(delta, mean_density):
    """Density field from an overdensity field: ``rho = (delta + 1) * mean``."""
    return (delta + 1.0) * mean_density


def effective_smoothing_radius(cell_size, gaussian_kernel_radius):
    """Effective Fourier-space smoothing radius for a Gaussian kernel on a grid."""
    return np.sqrt(gaussian_kernel_radius ** 2 - (0.64 * cell_size) ** 2)


def gaussian_smoothing_radius_from_effective(cell_size, effective_smoothing_radius):
    """Configuration-space Gaussian smoothing radius from the effective radius."""
    return np.sqrt(effective_smoothing_radius ** 2 + (0.64 * cell_size) ** 2)


def split_box_into_eight(density_on_grid):
    """Split a density grid into eight equal sub-boxes."""
    N = density_on_grid.shape[0]
    new_edge = N // 2
    return [density_on_grid[i:new_edge + i, j:new_edge + j, k:new_edge + k]
            for i in (0, new_edge)
            for j in (0, new_edge)
            for k in (0, new_edge)]


def periodic_distance(pos1, pos2, boxsize):
    """Minimum-image Euclidean distance between two sets of positions."""
    diff = pos2 - pos1 - np.round((pos2 - pos1) / boxsize) * boxsize
    return np.linalg.norm(diff, axis=-1)


def periodic_distance_mesh(i1, i2, size):
    """Minimum-image distance between two grid indices along one axis."""
    return min(abs(i1 - i2), size - abs(i1 - i2))


def euclidean_distance_periodic_mesh(i1, j1, k1, i2, j2, k2, size, cell_size):
    """Periodic Euclidean distance (in physical units) between two mesh cells."""
    di = periodic_distance_mesh(i1, i2, size)
    dj = periodic_distance_mesh(j1, j2, size)
    dk = periodic_distance_mesh(k1, k2, size)
    return (di ** 2 + dj ** 2 + dk ** 2) ** 0.5 * cell_size
