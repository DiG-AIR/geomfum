"""Base Descriptors Classes."""

import abc

import gsops.backend as gs

import geomfum.linalg as la


class Descriptor(abc.ABC):
    """Abstract base class for shape descriptors."""


class SpectralDescriptor(Descriptor, abc.ABC):
    """Spectral descriptor computed from Laplacian eigenfunctions with spectral filters.

    Parameters
    ----------
    spectral_filter : SpectralFilter
        Spectral filter.
    domain : callable or array-like, shape=[n_domain]
        Method to compute domain points (``f(basis, n_domain)``) or
        domain points.
    sigma : float
        Standard deviation for the Gaussian.
    scale : bool
        Whether to scale weights to sum to one.
    landmarks : bool
        Whether to compute landmarks based descriptors.
    k: int, optional
        Number of eigenvalues and eigenvectors to use. If None, basis.use_k is used.
    """

    def __init__(
        self,
        spectral_filter=None,
        domain=None,
        sigma=1,
        scale=True,
        landmarks=False,
        k=None,
    ):
        super().__init__()
        self.domain = domain
        self.sigma = sigma
        self.scale = scale
        self.landmarks = landmarks
        self.spectral_filter = spectral_filter
        self.k = k

    def __call__(self, shape):
        """Compute descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape.
        """
        if self.k is not None:
            if shape.basis.spectrum_size != self.k:
                shape.basis.use_k = self.k
        vals = shape.basis.vals
        vecs = shape.basis.vecs

        domain, sigma = (
            self.domain(shape) if callable(self.domain) else (self.domain, self.sigma)
        )

        coefs = self.spectral_filter(vals, domain, sigma)

        if self.landmarks:
            if not hasattr(shape, "landmark_indices") or shape.landmark_indices is None:
                raise AttributeError(
                    f"Shape must have 'landmark_indices' set for {self.__class__.__name__}."
                )
            return self._compute_landmark_descriptor(
                coefs, vecs, shape.landmark_indices
            )
        else:
            return self._compute_descriptor(coefs, vecs)

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
        vecs_term = gs.square(vecs)
        if self.scale:
            coefs = la.scale_to_unit_sum(coefs)
        return gs.einsum("...j,ij->...i", coefs, vecs_term)

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


class DistanceFromLandmarksDescriptor(Descriptor):
    """Descriptor computing geodesic distances from landmark points."""

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
