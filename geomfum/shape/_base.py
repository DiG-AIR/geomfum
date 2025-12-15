"""Base shape."""

import abc
import logging

import gsops.backend as gs

from geomfum.operator import Gradient, Laplacian


class Shape(abc.ABC):
    """Abstract base class for geometric shapes with differential operators.

    Parameters
    ----------
    is_mesh : bool
        Whether the shape is a mesh (True) or point cloud (False).
    """

    def __init__(self, is_mesh):
        self.is_mesh = is_mesh

        self._basis = None
        self.laplacian = Laplacian(self)
        self.gradient = Gradient(self)
        self.landmark_indices = None

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
