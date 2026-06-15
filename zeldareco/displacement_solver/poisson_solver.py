from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
from zeldareco.mesh.mesh import Mesh
import warnings
from zeldareco.utils.loggers import setup_logger

logger = setup_logger(__name__)


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

