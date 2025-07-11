import abc

import geomstats.backend as gs

import geomfum.backend as xgs
import geomfum.linalg as la


class Descriptor(abc.ABC):
    pass


class SpectralDescriptor(Descriptor, abc.ABC):
    """Spectral descriptor.

    Parameters
    ----------
    domain : callable or array-like, shape=[n_domain]
        Method to compute domain points (``f(basis, n_domain)``) or
        domain points.
    """

    def __init__(self, domain, scale = True):
        super().__init__()
        self.domain = domain
        self.scale = scale


    @abc.abstractmethod
    def __call__(self, shape,):
        """Compute descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape.
        """
        if not hasattr(shape, "landmark_indices") or shape.landmark_indices is None:
            raise AttributeError(
                f"Shape must have 'landmark_indices' set for {self.__class__.__name__}."
            )
        return self._compute(shape)

    def _scale(self, coefs):
        """Scale coefficients (weights) to unit sum.

        Parameters
        ----------
        coefs : array-like, shape=[n_domain, n_eigen]
            Coefficients to scale.

        Returns
        -------
        coefs : array-like, shape=[n_domain, n_eigen]
            Scaled coefficients.
        """
        if self.scale:
            return la.scale_to_unit_sum(coefs)
        return coefs

    def _compute_descriptor(self, coefs, vecs):
        """Compute descriptors from coefficients and eigenvectors.

        Parameters
        ----------
        coefs : array-like, shape=[n_domain, n_eigen]
            Coefficients.
        vecs : array-like, shape=[n_vertices, n_eigen]
            Eigenvectors.

        Returns
        -------
        descriptors : array-like, shape=[n_domain, n_vertices]
        """
        vecs_term = xgs.square(vecs)
        return gs.einsum("...j,ij->...i", coefs, vecs_term)

    def _compute_landmark_descriptor(self, coefs, vecs, landmarks):
        """Compute descriptors with landmarks.

        Parameters
        ----------
        coefs : array-like, shape=[n_domain, n_eigen]
            Coefficients.
        vecs : array-like, shape=[n_vertices, n_eigen]
            Eigenvectors.
        landmarks : array-like, shape=[n_landmarks]
            Landmark indices.

        Returns
        -------
        descriptors : array-like, shape=[n_landmarks * n_domain, n_vertices]
            Descriptor values.
        """
        weighted_evects = vecs[None, landmarks, :] * coefs[:, None, :]
        descriptor = gs.einsum("tpk,nk->ptn", weighted_evects, vecs)

        if self.scale:
            inv_scaling = coefs.sum(1)
            descriptor = (1 / inv_scaling)[None, :, None] * descriptor

        return gs.reshape(
            descriptor,
            (descriptor.shape[0] * descriptor.shape[1], vecs.shape[0]),
        )

    def _compute_landmark_descriptor(self, coefs, vecs, landmarks):
        """Compute descriptor with landmarks.

        Parameters
        ----------
        coefs : array-like, shape=[n_domain, n_eigen]
            Coefficients.
        vecs : array-like, shape=[n_vertices, n_eigen]
            Eigenvectors.
        landmarks : array-like, shape=[n_landmarks]
            Landmark indices.

        Returns
        -------
        descriptor : array-like, shape=[n_landmarks * n_domain, n_vertices]
            Descriptor values.
        """
        # weighted_evects: (n_domain, n_landmarks, n_eigen)
        weighted_evects = vecs[None, landmarks, :] * coefs[:, None, :]

        # result: (n_landmarks, n_domain, n_vertices)
        descriptor = gs.einsum("tpk,nk->ptn", weighted_evects, vecs)

        if self.scale:
            inv_scaling = coefs.sum(1)  # (n_domain,)
            descriptor = (1 / inv_scaling)[None, :, None] * descriptor

        # reshape to (n_landmarks * n_domain, n_vertices)
        return gs.reshape(
            descriptor,
            (descriptor.shape[0] * descriptor.shape[1], vecs.shape[0]),
        )
    
    def _compute(self, shape):
        """Compute descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape.

        Returns
        -------
        descriptor : array-like, shape=[n_landmarks * n_domain, n_vertices]
            Descriptor values.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    


class DistanceFromLandmarksDescriptor(Descriptor):
    """Distance from landmarks descriptor. A simple descriptor that returns the distance from landmarks as a function on the shape."""

    def __call__(self, shape):
        """Compute descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape.

        Returns
        -------
        descriptor : array-like, shape=[n_landmarks]
            Descriptor values.
        """
        if not hasattr(shape, "landmark_indices"):
            raise AttributeError(
                "shape object does not have 'landmark_indices' attribute"
            )

        if shape.metric is None:
            raise ValueError("shape is not equipped with metric")
        distances_list = shape.metric.dist_from_source(shape.landmark_indices)[0]
        distances = gs.stack(distances_list)
        return distances
