from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
from zeldareco.mesh.mesh import Mesh
import warnings
from zeldareco.utils.loggers import setup_logger

logger = setup_logger(__name__)


class DisplacementSolver(ABC):
    """
    Backwards-compatible displacement solver base class (copied from the previous `base_solver.py`).

    This class provides the `psi_mesh` property and the legacy constructor signature used
    by the older solvers. It is kept here so callers importing `DisplacementSolver` from
    `zeldareco.displacement_solver.poisson_solver` continue to work after the refactor.
    """

    def __init__(self,
                 delta_on_mesh: np.ndarray,
                 mesh: Mesh,
                 redshift: float = None,
                 RSDspace='RealSpace',
                 f: float = None,
                 bias: float = 1.0,
                 smoothing_radius: float = 15):

        self.mesh = mesh
        self.delta_on_mesh = delta_on_mesh
        if(self.delta_on_mesh is None):
            warnings.warn('The density field is not provided.')

        self.redshift = redshift
        self.f = f
        self.h = 0.7
        self.bias = bias
        # protect against None
        self.beta = (self.f / self.bias) if (self.f is not None and self.bias != 0) else 0.0
        self.smoothing_radius = smoothing_radius

        if(RSDspace not in ['RealSpace', 'RedshiftSpace']):
            raise ValueError(f"Invalid value for 'RSDspace': {RSDspace}. Must be 'RealSpace' or 'RedshiftSpace'.")
        else:
            self.RSDspace = RSDspace

        self._psi_x = None

    def print_info(self):
        self.mesh.print_info()
        logger.info(f'\\t')
        logger.info(f'Reconstruction information:')
        logger.info(f'Redshift is {self.redshift}')
        logger.info(f'Growth rate is {self.f}')
        logger.info(f'Bias is {self.bias}' )
        logger.info(f'Smoothing radius is {self.smoothing_radius}')
        logger.info(f'RSDspace is { self.RSDspace}')

    @property
    def delta_on_mesh(self):
        return self._delta_on_mesh

    @delta_on_mesh.setter
    def delta_on_mesh(self, delta):
        self._delta_on_mesh = delta

    @property
    def psi_mesh(self):
        if self._psi_x is None:
            logger.debug("psi not computed yet. Computing now...")
            self.compute_psi_mesh()
        logger.debug("Returning computed psi.")
        return self._psi_x



class PoissonSolver(ABC):
    """
    Abstract base class for Poisson-like solvers producing a potential and displacement.

    Provides lazy caching for `potential` and `displacement` and is compatible with
    the newer wrappers (`FFTSolver`, `MultigridSolver`).
    """

    def __init__(
        self,
        delta_on_mesh: np.ndarray,
        mesh: Mesh,
        f: float = None,
        bias: float = 1.0,
        RSDspace: str = "RealSpace",
    ) -> None:
        self.delta_on_mesh = np.asarray(delta_on_mesh)
        self.mesh = mesh
        self.f = f
        self.bias = bias
        # protect against None/bad values
        self.beta = (self.f / self.bias) if (self.f is not None and self.bias != 0) else 0.0
        self.RSDspace = RSDspace

        self._potential: Optional[np.ndarray] = None
        self._displacement: Optional[np.ndarray] = None

    @property
    def potential(self) -> Optional[np.ndarray]:
        if self._potential is None:
            logger.debug("Computing potential via _compute_potential()")
            self._compute_potential()
        return self._potential

    @property
    def displacement(self) -> Optional[np.ndarray]:
        if self._displacement is None:
            logger.debug("Computing displacement via _compute_displacement() or from potential")
            self._compute_displacement()
        return self._displacement

    @abstractmethod
    def _compute_displacement(self) -> None:
        logger.debug("No _compute_displacement implementation in base class. This should be overridden by subclasses.")
        raise NotImplementedError

    @abstractmethod
    def _compute_potential(self) -> None:
        logger.debug("No _compute_potential implementation in base class. This should be overridden by subclasses.")
        raise NotImplementedError

