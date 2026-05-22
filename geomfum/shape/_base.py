"""Base shape."""

import abc
import logging

import gsops.backend as gs
import torch

from geomfum.operator import Gradient, Laplacian


class Shape(abc.ABC):
    """Abstract base class for geometric shapes with differential operators.

    Parameters
    ----------
    shape_type : str
        Type of shape (e.g. ``"mesh"``, ``"pointcloud"``).
    """

    def __init__(self, shape_type):
        self.shape_type = shape_type

        self._basis = None
        self.laplacian = Laplacian(self)
        self.gradient = Gradient(self)
        self.landmark_indices = None

    @property
    def device(self):
        """Device of the shape, inferred from vertices."""
        return getattr(self.vertices, "device", torch.device("cpu"))

    def to(self, device):
        """Move all computed shape data to ``device``.

        Geometry is always computed on CPU (scipy constraint). This method
        only moves already-computed results. Lazy properties that haven't
        been computed yet will be computed on CPU and moved to ``device``
        on first access after this call.

        Parameters
        ----------
        device : torch.device or str
            Target device.

        Returns
        -------
        self : Shape
        """
        device = torch.device(device)
        if self.device == device:
            return self
        self.vertices = gs.to_device(self.vertices, device)
        if self._basis is not None:
            self._basis.to(device)
        self.laplacian.to(device)
        self.gradient.to(device)
        if self.landmark_indices is not None:
            self.landmark_indices = gs.to_device(self.landmark_indices, device)
        self._to(device)
        return self

    def _to(self, device):
        """Move subclass-specific cached fields to ``device``."""

    def equip_with_operator(self, name, Operator, allow_overwrite=True, **kwargs):
        """Equip shape with a differential or geometric operator.

        Parameters
        ----------
        name : str
            Attribute name for the operator.
        Operator : class
            Operator class to instantiate.
        allow_overwrite : bool, optional
            Whether to allow overwriting existing attributes (default: True).
        **kwargs
            Additional arguments passed to the Operator constructor.

        Returns
        -------
        self : Shape
            The shape instance for method chaining.
        """
        name_exists = hasattr(self, name)
        if name_exists:
            if allow_overwrite:
                logging.warning(f"Overriding {name}.")
            else:
                raise ValueError(f"{name} already exists")

        operator = Operator(self, **kwargs)
        setattr(self, name, operator)

        return self

    @property
    def basis(self):
        """Function basis associated with the shape.

        Returns
        -------
        basis : Basis
            Basis.
        """
        if self._basis is None:
            return self.laplacian.basis

        return self._basis

    def set_basis(self, basis):
        """Set function basis associated with the shape.

        Parameters
        ----------
        basis : Basis
            Basis.
        """
        self._basis = basis

    def set_landmarks(self, landmark_indices, append=False):
        """Set landmarks points on the shape.

        Parameters
        ----------
        landmark_indices : array-like, shape=[n_landmarks]
            Landmarks.
        append : bool
            Whether to append landmarks to already-existing ones.
        """
        if append:
            self.landmark_indices = gs.stack(self.landmark_indices, landmark_indices)

        else:
            self.landmark_indices = landmark_indices

        return self
