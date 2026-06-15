"""pyFM wrapper."""

import gsops.backend as gs
import numpy as np
import pyFM.functional
import pyFM.mesh
import pyFM.mesh.geometry
import pyFM.signatures

from geomfum.descriptor._base import SpectralDescriptor
from geomfum.descriptor.spectral import WksDefaultDomain, hks_default_domain
from geomfum.laplacian import BaseLaplacianFinder
from geomfum.matcher.base import BaseMatcher, CorrespondenceResult
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

    face_i = gs.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    face_j = gs.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])

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

    In = gs.concatenate([face_i, face_j, face_i, face_j])
    Jn = gs.concatenate([face_j, face_i, face_i, face_j])
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


class PyfmFunctionalMapMatcher(BaseMatcher):
    """Reference functional-map matcher wrapping pyFM's ``FunctionalMapping``.

    Registered under the ``"pyfm"`` key of
    :class:`~geomfum.matcher.fmap.FunctionalMapMatcher`. Build it with
    ``FunctionalMapMatcher.from_registry(which="pyfm", ...)``.

    It exposes the canonical functional-map pipeline of Ovsjanikov et al.
    (2012), as implemented in pyFM, through geomfum's :class:`BaseMatcher`
    contract so it can be dropped into the same evaluation framework as the
    geomfum-native matchers.

    The whole pipeline (Laplacian spectrum, WKS/HKS descriptors, energy
    optimization, optional ICP/ZoomOut refinement and the final point-to-point
    conversion) is delegated to pyFM. Nothing here touches geomfum's basis or
    descriptor machinery, so this matcher doubles as an *independent
    cross-check* of geomfum's own ``FunctionalMapMatcher`` (``which="geomfum"``):
    the two should produce comparable maps on the same pair.

    Orientation follows geomfum's convention: ``shape_a`` is pyFM ``mesh1`` and
    ``shape_b`` is pyFM ``mesh2``, so ``fmap12`` is pyFM's ``FM`` (shape
    ``[k_b, k_a]``) and ``p2p21`` is pyFM's ``p2p_21`` (for each vertex of
    ``shape_b``, the matched vertex index in ``shape_a``).

    Parameters
    ----------
    fmap_size : int or tuple of int
        Number of LBO eigenfunctions ``(k_a, k_b)``. A single int uses the same
        size for both shapes.
    n_descr : int
        Number of WKS/HKS descriptor functions computed before subsampling.
    descr_type : str
        Descriptor family, ``"WKS"`` or ``"HKS"``.
    subsample_step : int
        Step used to subsample descriptors (pyFM ``subsample_step``).
    k_process : int, optional
        Number of eigenpairs pyFM precomputes. Defaults to pyFM's own default
        (200). Lower it to speed up small-budget runs; it must be at least as
        large as the final spectrum size used by ZoomOut refinement.
    w_descr, w_lap, w_dcomm, w_orient : float
        Energy weights for descriptor preservation, Laplacian commutativity,
        descriptor-operator commutativity and orientation preservation. Defaults
        mirror pyFM's own defaults.
    orient_reversing : bool
        Use the orientation-reversing term instead of orientation-preserving
        (e.g. for shapes related by a mirror symmetry).
    optinit : str
        Functional-map initialization: ``"zeros"``, ``"identity"`` or
        ``"random"``.
    refine : str or None
        Optional refinement applied after the fit: ``None``, ``"icp"`` or
        ``"zoomout"``.
    refine_kwargs : dict, optional
        Extra keyword args forwarded to pyFM's ``icp_refine`` /
        ``zoomout_refine`` (e.g. ``{"nit": 10, "step": 1}`` for ZoomOut).
    use_landmarks : bool
        If True and both shapes carry ``landmark_indices``, pass them to pyFM as
        matched landmark descriptors.
    verbose : bool
        Forward pyFM verbosity.
    """

    def __init__(
        self,
        fmap_size=30,
        n_descr=100,
        descr_type="WKS",
        subsample_step=5,
        k_process=None,
        w_descr=1e-1,
        w_lap=1e-3,
        w_dcomm=1.0,
        w_orient=0.0,
        orient_reversing=False,
        optinit="zeros",
        refine=None,
        refine_kwargs=None,
        use_landmarks=False,
        verbose=False,
    ):
        if isinstance(fmap_size, int):
            fmap_size = (fmap_size, fmap_size)
        if refine not in (None, "icp", "zoomout"):
            raise ValueError(
                f'refine must be None, "icp" or "zoomout", not {refine!r}'
            )
        self.fmap_size = tuple(fmap_size)
        self.n_descr = n_descr
        self.descr_type = descr_type
        self.subsample_step = subsample_step
        self.k_process = k_process
        self.w_descr = w_descr
        self.w_lap = w_lap
        self.w_dcomm = w_dcomm
        self.w_orient = w_orient
        self.orient_reversing = orient_reversing
        self.optinit = optinit
        self.refine = refine
        self.refine_kwargs = refine_kwargs or {}
        self.use_landmarks = use_landmarks
        self.verbose = verbose

    def _to_trimesh(self, shape):
        """Build a pyFM ``TriMesh`` from a geomfum shape (CPU numpy)."""
        vertices = gs.to_numpy(gs.to_device(shape.vertices, "cpu"))
        faces = gs.to_numpy(gs.to_device(shape.faces, "cpu"))
        return pyFM.mesh.TriMesh(vertices, faces)

    def _landmarks(self, shape_a, shape_b):
        """Stack per-shape landmark indices into pyFM's ``(p, 2)`` format."""
        lmk_a = getattr(shape_a, "landmark_indices", None)
        lmk_b = getattr(shape_b, "landmark_indices", None)
        if not self.use_landmarks or lmk_a is None or lmk_b is None:
            return None
        return np.stack(
            [np.asarray(lmk_a).reshape(-1), np.asarray(lmk_b).reshape(-1)], axis=1
        )

    def __call__(self, shape_a, shape_b):
        """Compute correspondence between two shapes via pyFM.

        Parameters
        ----------
        shape_a : Shape
            First shape (target for ``p2p21``; pyFM ``mesh1``).
        shape_b : Shape
            Second shape (source for ``p2p21``; pyFM ``mesh2``).

        Returns
        -------
        result : CorrespondenceResult
            Contains ``fmap12`` (``[k_b, k_a]``), ``p2p21`` (``[n_vertices_b]``)
            and the subsampled, normalized descriptors ``descr_a`` / ``descr_b``.
        """
        mesh1 = self._to_trimesh(shape_a)
        mesh2 = self._to_trimesh(shape_b)

        model = pyFM.functional.FunctionalMapping(mesh1, mesh2)
        model.preprocess(
            n_ev=self.fmap_size,
            n_descr=self.n_descr,
            descr_type=self.descr_type,
            landmarks=self._landmarks(shape_a, shape_b),
            subsample_step=self.subsample_step,
            k_process=self.k_process,
            verbose=self.verbose,
        )
        model.fit(
            w_descr=self.w_descr,
            w_lap=self.w_lap,
            w_dcomm=self.w_dcomm,
            w_orient=self.w_orient,
            orient_reversing=self.orient_reversing,
            optinit=self.optinit,
            verbose=self.verbose,
        )

        if self.refine == "icp":
            model.icp_refine(verbose=self.verbose, **self.refine_kwargs)
        elif self.refine == "zoomout":
            model.zoomout_refine(verbose=self.verbose, **self.refine_kwargs)

        fmap12 = np.asarray(model.FM)  # (k_b, k_a)
        p2p21 = np.asarray(model.get_p2p())  # (n_b,) maps b -> a

        return CorrespondenceResult(
            fmap12=gs.from_numpy(fmap12),
            p2p21=gs.from_numpy(p2p21),
            descr_a=gs.from_numpy(model.descr1.T),
            descr_b=gs.from_numpy(model.descr2.T),
        )


class PyfmZoomOutMatcher(PyfmFunctionalMapMatcher):
    """pyFM functional map matcher with ZoomOut refinement (Melzi et al. 2019).

    Registered under the ``"pyfm"`` key of
    :class:`~geomfum.matcher.fmap.ZoomOutMatcher`. Build it with
    ``ZoomOutMatcher.from_registry(which="pyfm", ...)``.

    Identical to :class:`PyfmFunctionalMapMatcher` but with ZoomOut spectral
    upsampling applied after the initial fit. ``nit`` and ``step`` control the
    refinement (forwarded to pyFM's ``zoomout_refine``); all other keyword
    arguments are forwarded to :class:`PyfmFunctionalMapMatcher`.

    Parameters
    ----------
    fmap_size : int or tuple of int
        Initial number of LBO eigenfunctions, before refinement.
    nit : int
        Number of ZoomOut iterations.
    step : int
        Spectral upsampling step per iteration.
    """

    def __init__(self, fmap_size=30, nit=10, step=1, **kwargs):
        kwargs.pop("refine", None)
        refine_kwargs = kwargs.pop("refine_kwargs", None) or {"nit": nit, "step": step}
        super().__init__(
            fmap_size=fmap_size,
            refine="zoomout",
            refine_kwargs=refine_kwargs,
            **kwargs,
        )
