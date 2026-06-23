"""Field operations on the mesh (backend-dispatching).

Public entry points live in :mod:`baorecon.field_ops._interface`; CPU and GPU
kernels live in :mod:`baorecon.field_ops.cpu` / :mod:`baorecon.field_ops.gpu`.
"""

from baorecon.field_ops._interface import (
    CUPY_AVAILABLE,
    _get_array_module,
    divergence,
    divergence_FFT,
    interpolate_vector_field,
    project_vector_field,
    smoothed_field,
)

__all__ = [
    "CUPY_AVAILABLE",
    "_get_array_module",
    "divergence",
    "divergence_FFT",
    "interpolate_vector_field",
    "project_vector_field",
    "smoothed_field",
]
