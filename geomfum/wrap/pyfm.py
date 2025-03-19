"""pyFM wrapper."""

import numpy as np
import pyFM.mesh
import pyFM.signatures
import scipy

from geomfum.descriptor._base import SpectralDescriptor
from geomfum.laplacian import BaseLaplacianFinder
from geomfum.operator import FunctionalOperator, VectorFieldOperator


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
        stiffness_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Stiffness matrix.
        mass_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Diagonal lumped mass matrix.
        """
        return (
            pyFM.mesh.laplacian.cotangent_weights(shape.vertices, shape.faces),
            pyFM.mesh.laplacian.dia_area_mat(shape.vertices, shape.faces),
        )


class PyfmHeatKernelSignature(SpectralDescriptor):
    """Heat kernel signature using pyFM.

    Parameters
    ----------
    scaled : bool
        Whether to scale for each time value.
    n_domain : int
        Number of time points. Ignored if ``domain`` is not a callable.
    domain : callable or array-like, shape=[n_domain]
        Method to compute time points (``f(basis, n_domain)``) or
        time points.
    use_landmarks : bool
        Whether to use landmarks.
    """

    def __init__(self, scaled=True, n_domain=3, domain=None, use_landmarks=False):
        super().__init__(
            n_domain, domain or self.default_domain, use_landmarks=use_landmarks
        )
        self.scaled = scaled

    def default_domain(self, basis, n_domain):
        """Compute default domain.

        Parameters
        ----------
        basis : Eigenbasis.
            Basis.
        n_domain : int
            Number of time points.

        Returns
        -------
        domain : array-like, shape=[n_domain]
            Time points.
        """
        abs_ev = np.sort(np.abs(basis.vals))
        index = 1 if np.isclose(abs_ev[0], 0.0) else 0
        return np.geomspace(
            4 * np.log(10) / abs_ev[-1], 4 * np.log(10) / abs_ev[index], n_domain
        )

    def __call__(self, basis, domain=None):
        """Compute descriptor.

        Parameters
        ----------
        basis : Eigenbasis.
            Basis.
        domain : array-like, shape=[n_domain]
            Time points.

        Returns
        -------
        descr : array-like, shape=[n_domain, n_vertices]
            Descriptor.
        """
        if domain is None:
            domain = (
                self.domain(basis, self.n_domain)
                if callable(self.domain)
                else self.domain
            )

        if self.use_landmarks:
            return pyFM.signatures.lm_HKS(
                basis.vals,
                basis.vecs,
                basis.landmark_indices,
                domain,
                scaled=self.scaled,
            ).T

        return pyFM.signatures.HKS(basis.vals, basis.vecs, domain, scaled=self.scaled).T


class PyfmWaveKernelSignature(SpectralDescriptor):
    """Wave kernel signature using pyFM.

    Parameters
    ----------
    scaled : bool
        Whether to scale for each energy value.
    sigma : float
        Standard deviation.
    n_domain : int
        Number of energy points. Ignored if ``domain`` is not a callable.
    domain : callable or array-like, shape=[n_domain]
        Method to compute energy points (``f(basis, n_domain)``) or
        energy points.
    use_landmarks : bool
        Whether to use landmarks.
    """

    def __init__(
        self, scaled=True, sigma=None, n_domain=3, domain=None, use_landmarks=False
    ):
        super().__init__(
            n_domain, domain or self.default_domain, use_landmarks=use_landmarks
        )

        self.scaled = scaled
        self.sigma = sigma

    def default_sigma(self, e_min, e_max, n_domain):
        """Compute default sigma.

        Parameters
        ----------
        e_min : float
            Minimum energy.
        e_max : float
            Maximum energy.
        n_domain : int
            Number of energy points.

        Returns
        -------
        sigma : float
            Standard deviation.
        """
        return 7 * (e_max - e_min) / n_domain

    def default_domain(self, basis, n_domain):
        """Compute default domain.

        Parameters
        ----------
        basis : Eigenbasis.
            Basis.
        n_domain : int
            Number of energy points to use.

        Returns
        -------
        domain : array-like, shape=[n_domain]
        """
        abs_ev = np.sort(np.abs(basis.vals))
        index = 1 if np.isclose(abs_ev[0], 0.0) else 0

        e_min, e_max = np.log(abs_ev[index]), np.log(abs_ev[-1])

        sigma = (
            self.default_sigma(e_min, e_max, n_domain)
            if self.sigma is None
            else self.sigma
        )

        e_min += 2 * sigma
        e_max -= 2 * sigma

        energy = np.linspace(e_min, e_max, n_domain)

        return energy, sigma

    def __call__(self, basis, domain=None):
        """Compute descriptor.

        Parameters
        ----------
        basis : Eigenbasis.
            Basis.
        domain : array-like, shape=[n_domain]
            Energy points for computation.

        Returns
        -------
        descr : array-like, shape=[{n_domain, n_landmarks*n_domain}, n_vertices]
            Descriptor.
        """
        sigma = None
        if domain is None:
            if callable(self.domain):
                domain, sigma = self.domain(basis, self.n_domain)
            else:
                domain = self.domain

        if sigma is None:
            # TODO: simplify sigma
            # TODO: need to verify this
            sigma = (
                self.default_sigma(np.amin(domain), np.amax(domain), len(domain))
                if self.sigma is None
                else self.sigma
            )

        if self.use_landmarks:
            return pyFM.signatures.lm_WKS(
                basis.vals,
                basis.vecs,
                basis.landmark_indices,
                domain,
                sigma,
                scaled=self.scaled,
            ).T

        return pyFM.signatures.WKS(
            basis.vals, basis.vecs, domain, sigma, scaled=self.scaled
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
            return np.moveaxis(gradient, 0, 1)

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
        operator : scipy.sparse.csc_matrix or list[sparse.csc_matrix], shape=[n_vertices, n_vertices]
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
    --------------------------------
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
    --------------------------
    operator : sparse.csc_matrix or list[sparse.csc_matrix], shape=[n_vertices, n_verticess]
        (n_v,n_v) orientation operator.

    Notes
    -----
    * vectorized version of ``pyFm.geometry.mesh.get_orientation_op``.
    """
    n_vertices = per_vert_area.shape[0]
    per_vert_area = np.asarray(per_vert_area)

    v1 = vertices[faces[:, 0]]  # (n_f,3)
    v2 = vertices[faces[:, 1]]  # (n_f,3)
    v3 = vertices[faces[:, 2]]  # (n_f,3)

    # Define (normalized) gradient directions for each barycentric coordinate on each face
    # Remove normalization since it will disappear later on after multiplcation
    Jc1 = np.cross(normals, v3 - v2) / 2
    Jc2 = np.cross(normals, v1 - v3) / 2
    Jc3 = np.cross(normals, v2 - v1) / 2

    # Rotate the gradient field
    if rotated:
        rot_field = grad_field
    else:
        rot_field = np.cross(normals, grad_field)  # (n_f,3)

    I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    J = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])

    # Compute pairwise dot products between the gradient directions
    # and the gradient field
    Sij = (
        1
        / 3
        * np.concatenate(
            [
                np.einsum("ij,...ij->...i", Jc2, rot_field),
                np.einsum("ij,...ij->...i", Jc3, rot_field),
                np.einsum("ij,...ij->...i", Jc1, rot_field),
            ],
            axis=-1,
        )
    )

    Sji = (
        1
        / 3
        * np.concatenate(
            [
                np.einsum("ij,...ij->...i", Jc1, rot_field),
                np.einsum("ij,...ij->...i", Jc2, rot_field),
                np.einsum("ij,...ij->...i", Jc3, rot_field),
            ],
            axis=-1,
        )
    )

    In = np.concatenate([I, J, I, J])
    Jn = np.concatenate([J, I, I, J])
    Sn = np.concatenate([Sij, Sji, -Sij, -Sji], axis=-1)

    inv_area = scipy.sparse.diags(
        1 / per_vert_area, shape=(n_vertices, n_vertices), format="csc"
    )

    if Sn.ndim == 1:
        W = scipy.sparse.coo_matrix(
            (Sn, (In, Jn)), shape=(n_vertices, n_vertices)
        ).tocsc()

        return inv_area @ W

    out = []
    for Sn_ in Sn:
        W = scipy.sparse.coo_matrix(
            (Sn_, (In, Jn)), shape=(n_vertices, n_vertices)
        ).tocsc()
        out.append(inv_area @ W)

    return out
