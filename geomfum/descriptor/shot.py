"""SHOT (Signatures of Histograms of OrienTations) descriptor.

Shape-only implementation translated from the reference C++ code by
Tombari, Salti and Di Stefano:
    "Unique Signatures of Histograms for Local Surface Description"
    ECCV 2010

Only the shape channel (normal-alignment cosines) is implemented here;
the color channel is omitted since geomfum meshes do not carry per-vertex RGB.

References
----------
.. [1] Tombari, Salti, Di Stefano. "Unique Signatures of Histograms for
   Local Surface Description." ECCV 2010.
"""

import gsops.backend as gs
import numpy as np
from scipy.spatial import cKDTree

from ._base import Descriptor

# Angular constants (radians) matching the C++ macros
_DEG_45 = np.pi / 4  # 0.785...
_DEG_90 = np.pi / 2  # 1.570...
_DEG_135 = 3.0 * np.pi / 4  # 2.356...
_DEG_168 = (
    2.7488935718910690836548129603696  # 168° in radians (azimuth sector boundary)
)

# Standard SHOT spatial decomposition: 8 azimuth × 2 inclination × 2 radial = 32 volumes
_N_SECTORS = 32


def _compute_lrf(center, neighbors, radius):
    """Compute the SHOT local reference frame at *center*.

    Translated from ``getSHOTLocalRF`` in ``shot.cpp``.

    Parameters
    ----------
    center : ndarray, shape=[3]
        Central point.
    neighbors : ndarray, shape=[n, 3]
        Neighbour positions (must not contain *center* itself).
    radius : float
        Support radius used for distance-based weighting.

    Returns
    -------
    x_axis, y_axis, z_axis : ndarray, shape=[3] each
        Orthonormal local frame.  ``z_axis`` approximates the surface normal
        (smallest-eigenvalue direction of the weighted covariance);
        ``x_axis`` is the primary tangent (largest eigenvalue);
        ``y_axis = cross(z_axis, x_axis)``.
    """
    vij = neighbors - center  # (n, 3)
    dists = np.linalg.norm(vij, axis=1)  # (n,)
    weights = radius - dists  # (n,)  positive since dists < radius

    w_sum = weights.sum()
    if w_sum < 1e-30:
        # Degenerate neighbourhood — return identity frame
        return (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )

    cov = (weights[:, None] * vij).T @ vij / w_sum  # (3, 3)

    # eigh returns eigenvalues/vectors in ASCENDING order
    _, eigvecs = np.linalg.eigh(cov)
    x_axis = eigvecs[:, 2].copy()  # largest eigenvalue → primary tangent
    z_axis = eigvecs[:, 0].copy()  # smallest eigenvalue → surface normal

    # Sign disambiguation: orient each axis so that the majority of
    # neighbours lie on its positive side (mirroring the C++ logic).
    plus_x = int(np.sum(vij @ x_axis >= 0))
    if plus_x < len(vij) - plus_x:
        x_axis = -x_axis

    plus_z = int(np.sum(vij @ z_axis >= 0))
    if plus_z < len(vij) - plus_z:
        z_axis = -z_axis

    y_axis = np.cross(z_axis, x_axis)
    return x_axis, y_axis, z_axis


def _shot_single_point(
    center,
    neighbors,
    neighbor_normals,
    sq_distances,
    x_axis,
    y_axis,
    z_axis,
    radius,
    n_bins,
):
    """Compute the SHOT histogram for a single central point (vectorised).

    Translated from ``interpolateSingleChannel`` + the main loop in
    ``computeDesc`` in ``shot.cpp``.  The per-neighbour loop is replaced by
    bulk numpy operations; ``np.add.at`` handles unbuffered histogram voting.

    Parameters
    ----------
    center : ndarray, shape=[3]
    neighbors : ndarray, shape=[n, 3]
    neighbor_normals : ndarray, shape=[n, 3]
        Unit normals at each neighbour.
    sq_distances : ndarray, shape=[n]
        Squared Euclidean distances from *center* to each neighbour.
    x_axis, y_axis, z_axis : ndarray, shape=[3]
        Local reference frame at *center*.
    radius : float
    n_bins : int
        Number of shape bins (default 10 → 11 bins including boundary).

    Returns
    -------
    desc : ndarray, shape=[_N_SECTORS * (n_bins + 1)]
        L2-normalised descriptor.
    """
    desc = np.zeros(_N_SECTORS * (n_bins + 1), dtype=np.float64)

    # --- filter degenerate neighbours ---
    dists = np.sqrt(sq_distances)
    valid = dists > 1e-15
    if not np.any(valid):
        return desc

    dists = dists[valid]
    neighbors = neighbors[valid]
    neighbor_normals = neighbor_normals[valid]

    r_half = radius * 0.5
    r_14 = radius * 0.25
    r_34 = radius * 0.75

    # --- project neighbours into local reference frame ---
    deltas = neighbors - center  # (n, 3)
    x = deltas @ x_axis  # (n,)
    y = deltas @ y_axis
    z = deltas @ z_axis

    x = np.where(np.abs(x) < 1e-30, 0.0, x)
    y = np.where(np.abs(y) < 1e-30, 0.0, y)
    z = np.where(np.abs(z) < 1e-30, 0.0, z)

    # --- spatial sector index (desc_idx, 0–31) ---
    bit4 = ((y > 0) | ((y == 0) & (x < 0))).astype(np.int32)
    bit3_cond = (x > 0) | ((x == 0) & (y > 0))
    bit3 = np.where(bit3_cond, 1 - bit4, bit4).astype(np.int32)

    desc_idx = ((bit4 << 3) + (bit3 << 2)) << 1  # (n,) in {0, 8, 16, 24}

    cond = (x * y > 0) | (x == 0)
    desc_idx += np.where(
        cond,
        np.where(np.abs(x) >= np.abs(y), 0, 4),
        np.where(np.abs(x) > np.abs(y), 4, 0),
    )
    desc_idx += np.where(z > 0, 1, 0)  # inclination bit
    desc_idx += np.where(dists > r_half, 2, 0)  # radial bit

    # --- shape feature: cos between LRF z-axis and neighbour normals ---
    cos_desc = np.clip(neighbor_normals @ z_axis, -1.0, 1.0)
    bin_dist_f = ((1.0 + cos_desc) * n_bins) / 2.0  # (n,), in [0, n_bins]

    step_idx = np.floor(bin_dist_f + 0.5).astype(np.int32)
    volume_idx = desc_idx * (n_bins + 1)

    frac = bin_dist_f - step_idx  # fractional part for bin interpolation
    int_weight = 1.0 - np.abs(frac)

    # Feature-bin interpolation (modular wrapping)
    pos_mask = frac > 0
    np.add.at(
        desc,
        volume_idx[pos_mask] + (step_idx[pos_mask] + 1) % (n_bins + 1),
        frac[pos_mask],
    )
    np.add.at(
        desc,
        volume_idx[~pos_mask] + (step_idx[~pos_mask] + n_bins) % (n_bins + 1),
        -frac[~pos_mask],
    )

    # Radial interpolation (adjacent shell)
    outer = dists > r_half
    inner = ~outer

    rad_d_out = (dists - r_34) / r_half
    far_outer = outer & (dists > r_34)
    near_outer = outer & ~far_outer

    int_weight[far_outer] += 1.0 - rad_d_out[far_outer]
    int_weight[near_outer] += 1.0 + rad_d_out[near_outer]
    np.add.at(
        desc,
        (desc_idx[near_outer] - 2) * (n_bins + 1) + step_idx[near_outer],
        -rad_d_out[near_outer],
    )

    rad_d_in = (dists - r_14) / r_half
    far_inner = inner & (dists < r_14)
    near_inner = inner & ~far_inner

    int_weight[far_inner] += 1.0 + rad_d_in[far_inner]
    int_weight[near_inner] += 1.0 - rad_d_in[near_inner]
    np.add.at(
        desc,
        (desc_idx[near_inner] + 2) * (n_bins + 1) + step_idx[near_inner],
        rad_d_in[near_inner],
    )

    # Inclination interpolation (adjacent vertical zone)
    inc_cos = np.clip(z / dists, -1.0, 1.0)
    inclination = np.arccos(inc_cos)

    upper = (inclination > _DEG_90) | (
        (np.abs(inclination - _DEG_90) < 1e-30) & (z <= 0)
    )
    lower = ~upper

    inc_d_up = (inclination - _DEG_135) / _DEG_90
    inc_d_lo = (inclination - _DEG_45) / _DEG_90

    far_upper = upper & (inclination > _DEG_135)
    near_upper = upper & ~far_upper
    far_lower = lower & (inclination < _DEG_45)
    near_lower = lower & ~far_lower

    int_weight[far_upper] += 1.0 - inc_d_up[far_upper]
    int_weight[near_upper] += 1.0 + inc_d_up[near_upper]
    np.add.at(
        desc,
        (desc_idx[near_upper] + 1) * (n_bins + 1) + step_idx[near_upper],
        -inc_d_up[near_upper],
    )

    int_weight[far_lower] += 1.0 + inc_d_lo[far_lower]
    int_weight[near_lower] += 1.0 - inc_d_lo[near_lower]
    np.add.at(
        desc,
        (desc_idx[near_lower] - 1) * (n_bins + 1) + step_idx[near_lower],
        inc_d_lo[near_lower],
    )

    # Azimuth interpolation (adjacent angular sector, wrapped mod 32)
    has_az = (y != 0.0) | (x != 0.0)
    azimuth = np.arctan2(y, x)
    sel = desc_idx >> 2
    az_dist = (azimuth - (-_DEG_168 + _DEG_45 * sel)) / _DEG_45
    az_dist = np.clip(az_dist, -0.5, 0.5)

    az_pos = has_az & (az_dist > 0)
    az_neg = has_az & ~az_pos

    interp_pos = (desc_idx + 4) % _N_SECTORS
    interp_neg = (desc_idx - 4 + _N_SECTORS) % _N_SECTORS

    int_weight[az_pos] += 1.0 - az_dist[az_pos]
    np.add.at(
        desc,
        interp_pos[az_pos] * (n_bins + 1) + step_idx[az_pos],
        az_dist[az_pos],
    )

    int_weight[az_neg] += 1.0 + az_dist[az_neg]
    np.add.at(
        desc,
        interp_neg[az_neg] * (n_bins + 1) + step_idx[az_neg],
        -az_dist[az_neg],
    )

    # Accumulate main votes
    np.add.at(desc, volume_idx + step_idx, int_weight)

    # L2 normalise
    norm = np.linalg.norm(desc)
    if norm > 1e-15:
        desc /= norm
    return desc


class ShotDescriptor(Descriptor):
    """SHOT descriptor for triangle meshes.

    Computes a 352-dimensional (by default) local shape signature per vertex
    by accumulating normal-alignment statistics of neighbours into a spatially
    structured histogram.  The algorithm is translated directly from the
    reference C++ implementation by Tombari, Salti and Di Stefano (ECCV 2010).

    Parameters
    ----------
    radius : float, optional
        Support sphere radius.  If ``None`` (default), the radius is set
        automatically to ``0.05 × bounding-box diagonal`` of the input mesh,
        which gives a neighbourhood of roughly 50–200 vertices on typical
        normalised human-body meshes (FAUST, SMAL, …).
    n_bins : int
        Number of histogram bins per spatial volume (default 10).
        Descriptor dimensionality = 32 × (n_bins + 1) = 352 for n_bins=10.
    min_neighbors : int
        Minimum number of neighbours required to compute a descriptor for a
        vertex.  Vertices with fewer neighbours receive an all-zero row.

    References
    ----------
    .. [1] Tombari, Salti, Di Stefano. "Unique Signatures of Histograms for
       Local Surface Description." ECCV 2010.
    """

    def __init__(self, radius=None, n_bins=10, min_neighbors=5):
        super().__init__()
        self.radius = radius
        self.n_bins = n_bins
        self.min_neighbors = min_neighbors

    @property
    def descriptor_size(self):
        """Dimensionality of the descriptor vector per vertex."""
        return _N_SECTORS * (self.n_bins + 1)

    def _resolve_radius(self, vertices):
        if self.radius is not None:
            return float(self.radius)
        bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
        return 0.05 * bbox_diag

    def __call__(self, shape):
        """Compute the SHOT descriptor for every vertex of *shape*.

        Parameters
        ----------
        shape : TriangleMesh
            Input shape.  Must expose ``shape.vertices`` (n_vertices, 3) and
            ``shape.vertex_normals`` (n_vertices, 3).

        Returns
        -------
        descriptors : array-like, shape=[descriptor_size, n_vertices]
            L2-normalised per-vertex SHOT descriptors.
        """
        device = getattr(shape.vertices, "device", None)
        vertices = np.asarray(
            gs.to_numpy(gs.to_device(shape.vertices, "cpu")), dtype=np.float64
        )
        normals = np.asarray(
            gs.to_numpy(gs.to_device(shape.vertex_normals, "cpu")), dtype=np.float64
        )
        radius = self._resolve_radius(vertices)
        return gs.to_device(
            gs.array(self._compute_descriptors(vertices, normals, radius)), device
        )

    def _compute_descriptors(self, vertices, normals, radius):
        """Compute per-vertex SHOT histograms on CPU numpy arrays.

        Parameters
        ----------
        vertices : ndarray, shape=[n_vertices, 3]
        normals : ndarray, shape=[n_vertices, 3]
        radius : float

        Returns
        -------
        out : ndarray, shape=[descriptor_size, n_vertices]
        """
        n_vertices = vertices.shape[0]
        n_bins = self.n_bins
        desc_size = self.descriptor_size

        tree = cKDTree(vertices)
        all_neighbors = tree.query_ball_point(vertices, radius)

        out = np.zeros((n_vertices, desc_size), dtype=np.float64)

        for i in range(n_vertices):
            idx = [j for j in all_neighbors[i] if j != i]
            if len(idx) < self.min_neighbors:
                continue

            nb_pts = vertices[idx]
            nb_nrms = normals[idx]
            sq_dists = np.sum((nb_pts - vertices[i]) ** 2, axis=1)

            x_axis, y_axis, z_axis = _compute_lrf(vertices[i], nb_pts, radius)

            out[i] = _shot_single_point(
                vertices[i],
                nb_pts,
                nb_nrms,
                sq_dists,
                x_axis,
                y_axis,
                z_axis,
                radius,
                n_bins,
            )

        return out.T
