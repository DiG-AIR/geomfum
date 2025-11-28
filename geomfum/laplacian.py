"""Laplacian-related algorithms."""

import abc

import gsops.backend as gs

import geomfum.wrap as _wrap  # noqa (for register)
from geomfum._registry import LaplacianFinderRegistry, MeshWhichRegistryMixins
from geomfum.basis import LaplaceEigenBasis
from geomfum.numerics.eig import ScipyEigsh


class BaseLaplacianFinder(abc.ABC):
    """Algorithm to find the Laplacian."""

    @abc.abstractmethod
    def __call__(self, shape):
        """Apply algorithm.

        Parameters
        ----------
        shape : Shape
            Shape.

        Returns
        -------
        stiffness_matrix : array-like, shape=[n_vertices, n_vertices]
            Stiffness matrix.
        mass_matrix : array-like, shape=[n_vertices, n_vertices]
            Mass matrix.
        """


class LaplacianFinder(MeshWhichRegistryMixins, BaseLaplacianFinder):
    """Algorithm to find the Laplacian."""

    _Registry = LaplacianFinderRegistry

    def __call__(self, shape):
        """Apply algorithm. Laplace Beltrami operator with cotangent weights formulation.

        Parameters
        ----------
        shape : TriangleMesh
            Mesh.

        Returns
        -------
        stiffness_matrix : sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Stiffness matrix.
        mass_matrix : scipy.sparse.dia_matrix or sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Diagonal lumped mass matrix.
        """
        face_vertex_coords = shape.face_vertex_coords

        edges21 = face_vertex_coords[:, 2] - face_vertex_coords[:, 1]
        edges02 = face_vertex_coords[:, 0] - face_vertex_coords[:, 2]
        edges10 = face_vertex_coords[:, 1] - face_vertex_coords[:, 0]

        elen21 = gs.linalg.norm(edges21, axis=1)
        elen02 = gs.linalg.norm(edges02, axis=1)
        elen10 = gs.linalg.norm(edges10, axis=1)

        cos_angle12 = gs.einsum("ij,ij->i", -edges02, edges10) / (elen02 * elen10)
        cos_angle20 = gs.einsum("ij,ij->i", edges21, -edges10) / (elen21 * elen10)
        cos_angle01 = gs.einsum("ij,ij->i", -edges21, edges02) / (elen21 * elen02)

        vind012 = gs.concatenate(
            [shape.faces[:, 0], shape.faces[:, 1], shape.faces[:, 2]]
        )
        vind120 = gs.concatenate(
            [shape.faces[:, 1], shape.faces[:, 2], shape.faces[:, 0]]
        )
        cos_angles = gs.concatenate([cos_angle01, cos_angle12, cos_angle20])

        cot_angles = 0.5 * cos_angles / gs.sqrt(1 - cos_angles**2)

        row = gs.concatenate([vind012, vind120, vind012, vind120])
        col = gs.concatenate([vind120, vind012, vind012, vind120])
        data = gs.concatenate([-cot_angles, -cot_angles, cot_angles, cot_angles])

        stiffness_matrix = gs.sparse.csc_matrix(
            gs.stack([row, col]),
            data,
            shape=(shape.n_vertices, shape.n_vertices),
            coalesce=True,
        )

        mass_matrix = gs.sparse.dia_matrix(shape.vertex_areas)
        return stiffness_matrix, mass_matrix


class LaplacianSpectrumFinder:
    """Algorithm to find Laplacian spectrum.

    Parameters
    ----------
    spectrum_size : int
        Spectrum size. Ignored if ``eig_solver`` is not None.
    nonzero : bool
        Remove zero zero eigenvalue.
    fix_sign : bool
        Wheather to have all the first components with positive sign.
    laplacian_finder : BaseLaplacianFinder
        Algorithm to find the Laplacian. Ignored if Laplace and mass matrices
        were already computed.
    eig_solver : EigSolver
        Eigen solver.
    """

    def __init__(
        self,
        spectrum_size=100,
        nonzero=False,
        fix_sign=False,
        laplacian_finder=None,
        eig_solver=None,
    ):
        if eig_solver is None:
            eig_solver = ScipyEigsh(spectrum_size=spectrum_size, sigma=-0.01)

        self.nonzero = nonzero
        self.fix_sign = fix_sign
        self.laplacian_finder = laplacian_finder
        self.eig_solver = eig_solver

    @property
    def spectrum_size(self):
        """Spectrum size.

        Returns
        -------
        spectrum_size : int
            Spectrum size.
        """
        return self.eig_solver.spectrum_size

    @spectrum_size.setter
    def spectrum_size(self, spectrum_size):
        """Set spectrum size.

        Parameters
        ----------
        spectrum_size : int
            Spectrum size.
        """
        self.eig_solver.spectrum_size = spectrum_size

    def __call__(self, shape, as_basis=True, recompute=False):
        """Apply algorithm.

        Parameters
        ----------
        shape : Shape
            Shape.
        as_basis : bool
            Whether return basis or eigenvals/vecs.
        recompute : bool
            Whether to recompute Laplacian if information is cached.

        Returns
        -------
        eigenvals : array-like, shape=[spectrum_size]
            Eigenvalues. (If ``basis is False``.)
        eigenvecs : array-like, shape=[n_vertices, spectrum_size]
            Eigenvectors. (If ``basis is False``.)
        basis : LaplaceEigenBasis
            A basis. (If ``basis is True``.)
        """
        stiffness_matrix, mass_matrix = shape.laplacian.find(
            self.laplacian_finder, recompute=recompute
        )

        eigenvals, eigenvecs = self.eig_solver(stiffness_matrix, M=mass_matrix)

        if self.nonzero:
            eigenvals = eigenvals[1:]
            eigenvecs = eigenvecs[:, 1:]

        if self.fix_sign:
            indices = eigenvecs[0, :] < 0
            eigenvals[indices] *= -1
            eigenvecs[:, indices] *= -1

        if as_basis:
            return LaplaceEigenBasis(shape, eigenvals, eigenvecs)

        return eigenvals, eigenvecs
