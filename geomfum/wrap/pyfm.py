"""pyFM wrapper."""

import gsops.backend as gs
import numpy as np
import pyFM.mesh
import pyFM.mesh.geometry
import pyFM.signatures

from geomfum.descriptor._base import SpectralDescriptor
from geomfum.descriptor.spectral import WksDefaultDomain, hks_default_domain
from geomfum.laplacian import BaseLaplacianFinder
from geomfum.operator import FunctionalOperator, VectorFieldOperator
from geomfum.sample import BaseSampler


class PyfmMeshLaplacianFinder(BaseLaplacianFinder):
    """Algorithm to find the Laplacian of a mesh."""

    def __call__(self, shape):
        """Apply algorithm.

        Parameters
        ----------
        shape : TriangleMesh
            Mesh.

        Returns
        -------
        stiffness_matrix : sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Stiffness matrix.
        mass_matrix : sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Diagonal lumped mass matrix.
        """
        return (
            gs.sparse.from_scipy_csc(
                pyFM.mesh.laplacian.cotangent_weights(shape.vertices, shape.faces)
            ),
            gs.sparse.from_scipy_dia(
                pyFM.mesh.laplacian.dia_area_mat(shape.vertices, shape.faces)
            ),
        )


class PyfmHeatKernelSignature(SpectralDescriptor):
    """Heat kernel signature using pyFM.

    Parameters
    ----------
    scale : bool
        Whether to scale weights to sum to one.
    n_domain : int
        Number of domain points. Ignored if ``domain`` is not None.
    domain : callable or array-like, shape=[n_domain]
        Method to compute time points (``f(shape, n_domain)``) or
        time points.
    """

    def __init__(self, scale=True, n_domain=3, domain=None):
        super().__init__(
            domain=domain
            or (lambda shape: hks_default_domain(shape, n_domain=n_domain)),
        )
        self.scale = scale

    def __call__(self, shape):
        """Compute descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape with basis.

        Returns
        -------
        descr : array-like, shape=[n_domain, n_vertices]
            Descriptor.
        """
        domain, _ = (
            self.domain(shape) if callable(self.domain) else (self.domain, self.sigma)
        )

        return gs.from_numpy(
            pyFM.signatures.HKS(
                shape.basis.vals, shape.basis.vecs, domain, scaled=self.scale
            ).T
        )


class PyfmLandmarkHeatKernelSignature(SpectralDescriptor):
    """Landmark-based Heat kernel signature using pyFM.

    Parameters
    ----------
    scale : bool
        Whether to scale weights to sum to one.
    n_domain : int
        Number of domain points. Ignored if ``domain`` is not None.
    domain : callable or array-like, shape=[n_domain]
        Method to compute domain points (``f(shape)``) or
        domain points.
    """

    def __init__(self, scale=True, n_domain=3, domain=None):
        super().__init__(
            domain=domain
            or (lambda shape: hks_default_domain(shape, n_domain=n_domain)),
        )
        self.scale = scale

    def __call__(self, shape):
        """Compute landmark descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape with basis.

        Returns
        -------
        descr : array-like, shape=[n_domain, n_vertices]
            Descriptor.
        """
        if not hasattr(shape, "landmark_indices") or shape.landmark_indices is None:
            raise AttributeError(
                "Shape must have 'landmark_indices' set for LandmarkHeatKernelSignature."
            )

        domain, _ = (
            self.domain(shape) if callable(self.domain) else (self.domain, self.sigma)
        )

        return gs.from_numpy(
            pyFM.signatures.lm_HKS(
                shape.basis.vals,
                shape.basis.vecs,
                shape.landmark_indices,
                domain,
                scaled=self.scale,
            ).T
        )


class PyfmWaveKernelSignature(SpectralDescriptor):
    """Wave kernel signature using pyFM.

    Parameters
    ----------
    scale : bool
        Whether to scale weights to sum to one.
    sigma : float
        Standard deviation. Ignored if ``domain`` is a callable (other
        than default one).
    n_domain : int
        Number of energy points. Ignored if ``domain`` is not a callable.
    domain : callable or array-like, shape=[n_domain]
        Method to compute domain points (``f(shape)``) or
        domain points.
    """

    def __init__(
        self, scale=True, sigma=None, n_domain=3, domain=None, landmarks=False
    ):
        super().__init__(
            domain=domain or WksDefaultDomain(n_domain=n_domain, sigma=sigma),
            scale=scale,
            landmarks=landmarks,
            sigma=sigma,
        )

    def __call__(self, shape):
        """Compute descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape with basis.

        Returns
        -------
        descr : array-like, shape=[{n_domain, n_landmarks*n_domain}, n_vertices]
            Descriptor.
        """
        if callable(self.domain):
            domain, sigma = self.domain(shape)
        else:
            domain = self.domain
            sigma = self.sigma

        return pyFM.signatures.WKS(
            shape.basis.vals, shape.basis.vecs, domain, sigma, scaled=self.scale
        ).T


class PyfmLandmarkWaveKernelSignature(SpectralDescriptor):
    """Landmark-based Wave kernel signature using pyFM.

    Parameters
    ----------
    scale : bool
        Whether to scale weights to sum to one.
    sigma : float
        Standard deviation. Ignored if ``domain`` is a callable (other
        than default one).
    n_domain : int
        Number of energy points. Ignored if ``domain`` is not a callable.
    domain : callable or array-like, shape=[n_domain]
        Method to compute domain points (``f(shape)``) or
        domain points.
    """

    def __init__(self, scale=True, sigma=None, n_domain=3, domain=None):
        super().__init__(
            domain=domain or WksDefaultDomain(n_domain=n_domain, sigma=sigma),
        )
        self.scale = scale
        self.sigma = sigma

    def __call__(self, shape):
        """Compute landmark descriptor."""
        if not hasattr(shape, "landmark_indices") or shape.landmark_indices is None:
            raise AttributeError(
                "Shape must have 'landmark_indices' set for LandmarkHeatKernelSignature."
            )

        domain, sigma = (
            self.domain(shape) if callable(self.domain) else (self.domain, self.sigma)
        )
        return pyFM.signatures.lm_WKS(
            shape.basis.vals,
            shape.basis.vecs,
            shape.landmark_indices,
            domain,
            sigma,
            scaled=self.scale,
        ).T


class PyfmFaceValuedGradient(FunctionalOperator):
    """Gradient of a function on a mesh.

    Computes the gradient of a function on f using linear
    interpolation between vertices.
    """

    def __call__(self, point):
        """Apply operator.

        Parameters
        ----------
        point : array-like, shape=[..., n_vertices]
            Function value on each vertex.

        Returns
        -------
        gradient : array-like, shape=[..., n_faces]
            Gradient of the function on each face.
        """
        gradient = pyFM.mesh.geometry.grad_f(
            point.T,
            self._shape.vertices,
            self._shape.faces,
            self._shape.face_normals,
            face_areas=self._shape.face_areas,
        )
        if gradient.ndim > 2:
            return gs.moveaxis(gradient, 0, 1)

        return gradient


class PyfmFaceDivergenceOperator(VectorFieldOperator):
    """Divergence of a function on a mesh."""

    def __call__(self, vector):
        """Divergence of a vector field on a mesh.

        Parameters
        ----------
        vector : array-like, shape=[..., n_faces, 3]
            Vector field on the mesh.

        Returns
        -------
        divergence : array-like, shape=[..., n_vertices]
            Divergence of the vector field on each vertex.
        """
        if vector.ndim > 2:
            vector = np.moveaxis(vector, 0, 1)

        div = pyFM.mesh.geometry.div_f(
            vector,
            self._shape.vertices,
            self._shape.faces,
            self._shape.face_normals,
            vert_areas=self._shape.vertex_areas,
        )
        if div.ndim > 1:
            return np.moveaxis(div, 0, 1)

        return div


class PyFmFaceOrientationOperator(VectorFieldOperator):
    r"""Orientation operator associated to a gradient field.

    For a given function :math:`g` on the vertices, this operator linearly computes
    :math:`< \grad(f) x \grad(g)`, n> for each vertex by averaging along the adjacent
    faces.
    In practice, we compute :math:`< n x \grad(f), \grad(g) >` for simpler computation.
    """

    def __call__(self, vector):
        """Apply operator.

        Parameters
        ----------
        vector : array-like, shape=[..., n_faces, 3]
            Gradient field on the mesh.

        Returns
        -------
        operator : sparse.csc_matrix or list[sparse.csc_matrix], shape=[n_vertices, n_vertices]
            Orientation operator.
        """
        return get_orientation_op(
            vector,
            self._shape.vertices,
            self._shape.faces,
            self._shape.face_normals,
            self._shape.vertex_areas,
        )


def get_orientation_op(
    grad_field, vertices, faces, normals, per_vert_area, rotated=False
):
    """
    Compute the linear orientation operator associated to a gradient field grad(f).

    This operator computes g -> < grad(f) x grad(g), n> (given at each vertex) for any function g
    In practice, we compute < n x grad(f), grad(g) > for simpler computation.

    Parameters
    ----------
    grad_field    :
        (n_f,3) gradient field on the mesh
    vertices      :
        (n_v,3) coordinates of vertices
    faces         :
        (n_f,3) indices of vertices for each face
    normals       :
        (n_f,3) normals coordinate for each face
    per_vert_area :
        (n_v,) voronoi area for each vertex
    rotated       : bool
        whether gradient field is already rotated by n x grad(f)

    Returns
    -------
    operator : sparse.csc_matrix or list[sparse.csc_matrix], shape=[n_vertices, n_verticess]
        (n_v,n_v) orientation operator.

    Notes
    -----
    * vectorized version of ``pyFm.geometry.mesh.get_orientation_op``.
    """
    n_vertices = per_vert_area.shape[0]
    per_vert_area = gs.asarray(per_vert_area)

    v1 = vertices[faces[:, 0]]  # (n_f,3)
    v2 = vertices[faces[:, 1]]  # (n_f,3)
    v3 = vertices[faces[:, 2]]  # (n_f,3)

    # Define (normalized) gradient directions for each barycentric coordinate on each face
    # Remove normalization since it will disappear later on after multiplcation
    Jc1 = gs.cross(normals, v3 - v2) / 2
    Jc2 = gs.cross(normals, v1 - v3) / 2
    Jc3 = gs.cross(normals, v2 - v1) / 2

    # Rotate the gradient field
    if rotated:
        rot_field = grad_field
    else:
        rot_field = gs.cross(normals, grad_field)  # (n_f,3)

    I = gs.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    J = gs.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])

    # Compute pairwise dot products between the gradient directions
    # and the gradient field
    Sij = (
        1
        / 3
        * gs.concatenate(
            [
                gs.einsum("ij,...ij->...i", Jc2, rot_field),
                gs.einsum("ij,...ij->...i", Jc3, rot_field),
                gs.einsum("ij,...ij->...i", Jc1, rot_field),
            ],
            axis=-1,
        )
    )

    Sji = (
        1
        / 3
        * gs.concatenate(
            [
                gs.einsum("ij,...ij->...i", Jc1, rot_field),
                gs.einsum("ij,...ij->...i", Jc2, rot_field),
                gs.einsum("ij,...ij->...i", Jc3, rot_field),
            ],
            axis=-1,
        )
    )

    In = gs.concatenate([I, J, I, J])
    Jn = gs.concatenate([J, I, I, J])
    Sn = gs.concatenate([Sij, Sji, -Sij, -Sji], axis=-1)

    inv_area = gs.sparse.dia_matrix(1 / per_vert_area, shape=(n_vertices, n_vertices))

    indices = gs.stack([In, Jn])
    if Sn.ndim == 1:
        W = gs.sparse.csc_matrix(
            indices, Sn, shape=(n_vertices, n_vertices), coalesce=True
        )

        return inv_area @ W

    out = []
    for Sn_ in Sn:
        W = gs.sparse.csc_matrix(
            indices, Sn_, shape=(n_vertices, n_vertices), coalesce=True
        )
        out.append(inv_area @ W)

    return out


class PyfmEuclideanFarthestVertexSampler(BaseSampler):
    """Farthest point Euclidean sampling.

    Parameters
    ----------
    min_n_samples : int
        Minimum number of samples to target.
    """

    def __init__(self, min_n_samples):
        super().__init__()
        self.min_n_samples = min_n_samples

    def sample(self, shape):
        """Sample using farthest point sampling.

        Parameters
        ----------
        shape : TriangleMesh
            Mesh.

        Returns
        -------
        samples : array-like, shape=[n_samples, 3]
            Coordinates of samples.
        """

        def dist_func(i):
            return np.linalg.norm(shape.vertices - shape.vertices[i, None, :], axis=1)

        return pyFM.mesh.geometry.farthest_point_sampling_call(
            dist_func,
            self.min_n_samples,
            n_points=shape.n_vertices,
            verbose=False,
        )
