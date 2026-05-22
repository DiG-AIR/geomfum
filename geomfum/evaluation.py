"""Correspondence evaluation metrics and manager.

Notation
--------
- ``p2p21``: point-to-point map from shape_b to shape_a.
  ``p2p21[i]`` = vertex in A that vertex i in B maps to.
- ``dist_a``: geodesic distance matrix on shape A.
- ``corr_a``, ``corr_b``: ground-truth correspondence indices into A and B.
- ``mask_a``: binary overlap mask on A (1 = in overlap region).

Inputs dict
-----------
``EvaluationManager.compute`` assembles a flat dict from:

1. ``CorrespondenceResult.to_dict()`` (or any plain dict) —
   keys: ``p2p21``, ``fmap12``, ``soft_perm_ba``, ``overlap_ab``, …
2. Explicit context kwargs:
   ``shape_a``, ``shape_b``, ``corr_a``, ``corr_b``, ``dist_a``, ``mask_a``.

Each metric declares ``required_inputs`` listing the keys it needs.
Metrics whose required inputs are absent are silently skipped.
"""

import abc

import numpy as np

from geomfum.metric._base import VertexEuclideanMetric

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class CorrespondenceMetric(abc.ABC):
    """Base class for correspondence evaluation metrics.

    Follows the same ``required_inputs`` pattern as
    ``geomfum.learning.losses`` loss classes, but operates on numpy arrays
    (no gradients) and receives inputs as a single flat dict.

    Subclasses must declare ``required_inputs`` and implement ``__call__``.
    """

    required_inputs: list = []

    @abc.abstractmethod
    def __call__(self, inputs: dict) -> float:
        """Compute the metric from the flat inputs dict.

        Parameters
        ----------
        inputs : dict
            Flat dict assembled by ``EvaluationManager``.  All keys in
            ``required_inputs`` are guaranteed to be present.

        Returns
        -------
        float
        """


# ---------------------------------------------------------------------------
# Full-shape metrics
# ---------------------------------------------------------------------------


class GeodesicErrorMetric(CorrespondenceMetric):
    """Normalised mean geodesic error of ``p2p21``.

    Error = mean(dist_a[p2p21[corr_b], corr_a]) / diam_a.
    Falls back to identity correspondence when ``corr_a``/``corr_b`` absent.

    Required inputs: ``p2p21``, ``dist_a``.
    Optional inputs: ``corr_a``, ``corr_b``.
    """

    required_inputs = ["p2p21", "dist_a"]

    def __call__(self, inputs):
        """Compute the normalised mean geodesic error."""
        dist_a = np.asarray(inputs["dist_a"])
        p2p21 = np.asarray(inputs["p2p21"])
        corr_a = inputs.get("corr_a")
        corr_b = inputs.get("corr_b")

        if corr_a is None or corr_b is None:
            error = np.mean(dist_a[p2p21, np.arange(len(p2p21))])
        else:
            error = np.mean(dist_a[p2p21[np.asarray(corr_b)], np.asarray(corr_a)])

        diam = dist_a.max()
        return float(error / diam) if diam > 0 else 0.0


class EuclideanErrorMetric(CorrespondenceMetric):
    """Normalised mean Euclidean error of ``p2p21``.

    Error = mean(||v_a[p2p21[corr_b]] - v_a[corr_a]||) / eucl_diam_a.
    Falls back to identity when ``corr_a``/``corr_b`` absent.

    Required inputs: ``p2p21``, ``shape_a``.
    Optional inputs: ``corr_a``, ``corr_b``.
    """

    required_inputs = ["p2p21", "shape_a"]

    def __call__(self, inputs):
        """Compute the normalised mean Euclidean error."""
        p2p21 = np.asarray(inputs["p2p21"])
        shape_a = inputs["shape_a"]
        corr_a = inputs.get("corr_a")
        corr_b = inputs.get("corr_b")

        verts = np.asarray(shape_a.vertices)

        if corr_a is None or corr_b is None:
            pred = verts[p2p21]
            gt = verts[np.arange(len(p2p21))]
        else:
            pred = verts[p2p21[np.asarray(corr_b)]]
            gt = verts[np.asarray(corr_a)]

        error = np.mean(np.linalg.norm(pred - gt, axis=-1))
        diam = VertexEuclideanMetric(shape_a).dist_matrix().max()
        return float(error / diam) if diam > 0 else 0.0


class DirichletEnergyMetric(CorrespondenceMetric):
    """Dirichlet energy of ``p2p21`` (smoothness proxy).

    E = sum_i v_a[p2p21]^T L_b v_a[p2p21] / n_b.

    Required inputs: ``p2p21``, ``shape_a``, ``shape_b``.
    """

    required_inputs = ["p2p21", "shape_a", "shape_b"]

    def __call__(self, inputs):
        """Compute the Dirichlet energy."""
        p2p21 = np.asarray(inputs["p2p21"])
        shape_a = inputs["shape_a"]
        shape_b = inputs["shape_b"]

        mapping = np.asarray(shape_a.vertices[p2p21])  # [n_b, 3]

        if shape_b.laplacian.stiffness_matrix is None:
            L, _ = shape_b.laplacian.find()
        else:
            L = shape_b.laplacian.stiffness_matrix

        if hasattr(L, "tocsr"):
            L = L.tocsr()

        energy = sum(float(mapping[:, i].T @ L @ mapping[:, i]) for i in range(3))
        return energy / shape_b.n_vertices


class CoverageMetric(CorrespondenceMetric):
    """Area-weighted fraction of shape A covered by ``p2p21``.

    Required inputs: ``p2p21``, ``shape_a``.
    """

    required_inputs = ["p2p21", "shape_a"]

    def __call__(self, inputs):
        """Compute the coverage fraction."""
        p2p21 = np.asarray(inputs["p2p21"])
        shape_a = inputs["shape_a"]
        areas = np.asarray(shape_a.vertex_areas)
        unique = np.unique(p2p21)
        return float(areas[unique].sum() / areas.sum())


class CoverageCountMetric(CorrespondenceMetric):
    """Count-based fraction of shape A vertices reached by ``p2p21``.

    Required inputs: ``p2p21``, ``shape_a``.
    """

    required_inputs = ["p2p21", "shape_a"]

    def __call__(self, inputs):
        """Compute the count-based coverage fraction."""
        p2p21 = np.asarray(inputs["p2p21"])
        shape_a = inputs["shape_a"]
        return float(np.unique(p2p21).shape[0] / shape_a.n_vertices)


class SoftGeodesicErrorMetric(CorrespondenceMetric):
    """Expected geodesic error under a soft permutation ``soft_perm_ba``.

    Required inputs: ``soft_perm_ba``, ``dist_a``.
    Optional inputs: ``corr_a``, ``corr_b``.
    """

    required_inputs = ["soft_perm_ba", "dist_a"]

    def __call__(self, inputs):
        """Compute the expected geodesic error under a soft permutation."""
        P = np.asarray(inputs["soft_perm_ba"])  # [n_b, n_a]
        D = np.asarray(inputs["dist_a"])  # [n_a, n_a]
        corr_a = inputs.get("corr_a")
        corr_b = inputs.get("corr_b")

        diam = D.max()

        if corr_a is None or corr_b is None:
            expected = (P * D.T).sum(axis=1)
        else:
            corr_a = np.asarray(corr_a)
            corr_b = np.asarray(corr_b)
            P_rows = P[corr_b, :]  # [n_corr, n_a]
            gt_dists = D[:, corr_a].T  # [n_corr, n_a]
            expected = (P_rows * gt_dists).sum(axis=1)

        return float(np.mean(expected) / diam) if diam > 0 else 0.0


# ---------------------------------------------------------------------------
# Partial-shape metrics
# ---------------------------------------------------------------------------


class PartialGeodesicErrorMetric(CorrespondenceMetric):
    """Geodesic error restricted to the GT overlap region (mask_a).

    Follows the filtered protocol from EchoMatch / SHREC16.

    Required inputs: ``p2p21``, ``dist_a``, ``corr_a``, ``corr_b``, ``mask_a``.
    """

    required_inputs = ["p2p21", "dist_a", "corr_a", "corr_b", "mask_a"]

    def __call__(self, inputs):
        """Compute the normalised mean geodesic error."""
        dist_a = np.asarray(inputs["dist_a"])
        p2p21 = np.asarray(inputs["p2p21"])
        corr_a = np.asarray(inputs["corr_a"])
        corr_b = np.asarray(inputs["corr_b"])
        mask_a = np.asarray(inputs["mask_a"])

        valid = mask_a[corr_a] > 0.5
        if valid.sum() == 0:
            return 0.0
        return float(np.mean(dist_a[corr_a[valid], p2p21[corr_b[valid]]]))


class OverlapIoUMetric(CorrespondenceMetric):
    """IoU between predicted overlap scores and GT overlap mask on A.

    Parameters
    ----------
    threshold : float
        Binarisation threshold for ``overlap_ab``.  Default 0.5.

    Required inputs: ``overlap_ab``, ``mask_a``.
    """

    required_inputs = ["overlap_ab", "mask_a"]

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, inputs):
        """Compute the IoU between predicted and GT overlap."""
        pred = np.asarray(inputs["overlap_ab"]) >= self.threshold
        gt = np.asarray(inputs["mask_a"]) >= 0.5
        intersection = (pred & gt).sum()
        union = (pred | gt).sum()
        return 1.0 if union == 0 else float(intersection / union)


class PCKAucMetric(CorrespondenceMetric):
    """AUC of the PCK curve, evaluated over the GT overlap region.

    Parameters
    ----------
    t_max : float
        Maximum normalised geodesic threshold.  Default 0.20.
    n_steps : int
        Number of threshold steps.  Default 100.

    Required inputs: ``p2p21``, ``dist_a``, ``corr_a``, ``corr_b``, ``mask_a``.
    """

    required_inputs = ["p2p21", "dist_a", "corr_a", "corr_b", "mask_a"]

    def __init__(self, t_max: float = 0.20, n_steps: int = 100):
        self.t_max = t_max
        self.n_steps = n_steps

    def __call__(self, inputs):
        """Compute the AUC of the PCK curve over the GT overlap region."""
        dist_a = np.asarray(inputs["dist_a"])
        p2p21 = np.asarray(inputs["p2p21"])
        corr_a = np.asarray(inputs["corr_a"])
        corr_b = np.asarray(inputs["corr_b"])
        mask_a = np.asarray(inputs["mask_a"])

        valid = mask_a[corr_a] > 0.5
        if valid.sum() == 0:
            return 0.0

        geo_err = dist_a[corr_a[valid], p2p21[corr_b[valid]]]
        diam = dist_a.max()
        geo_err_norm = geo_err / diam if diam > 0 else geo_err

        thresholds = np.linspace(0.0, self.t_max, self.n_steps)
        pck = np.array([(geo_err_norm <= t).mean() for t in thresholds])
        return float(np.trapezoid(pck, thresholds) / self.t_max)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class EvaluationManager:
    """Manages a set of correspondence evaluation metrics.

    Analogous to ``geomfum.learning.losses.LossManager`` but for
    post-hoc evaluation: no gradients, no weighting, graceful skipping.

    Assembles a flat inputs dict from a ``CorrespondenceResult`` (or plain
    dict) plus explicit context kwargs, then runs every metric whose
    ``required_inputs`` are satisfied.

    Parameters
    ----------
    metrics : list of CorrespondenceMetric, or dict of str → CorrespondenceMetric
        If a list, metric names are taken from the class name.

    Examples
    --------
    >>> evaluator = EvaluationManager([
    ...     GeodesicErrorMetric(),
    ...     CoverageMetric(),
    ...     SoftGeodesicErrorMetric(),
    ... ])
    >>> result = matcher(shape_a, shape_b)          # CorrespondenceResult
    >>> scores = evaluator.compute(
    ...     result, shape_a=shape_a, shape_b=shape_b,
    ...     corr_a=corr_a, corr_b=corr_b, dist_a=dist_a,
    ... )
    >>> # {"GeodesicErrorMetric": 0.043, "CoverageMetric": 0.97}
    """

    def __init__(self, metrics):
        if isinstance(metrics, list):
            self.metrics = {type(m).__name__: m for m in metrics}
        else:
            self.metrics = dict(metrics)

    def compute(
        self,
        result,
        *,
        shape_a=None,
        shape_b=None,
        corr_a=None,
        corr_b=None,
        dist_a=None,
        mask_a=None,
    ) -> dict:
        """Compute all applicable metrics.

        Parameters
        ----------
        result : CorrespondenceResult or dict
            Matcher output.  ``CorrespondenceResult.to_dict()`` is called
            automatically; non-None fields are added to the inputs dict.
        shape_a, shape_b : Shape, optional
        corr_a, corr_b : array-like, optional
            Ground-truth correspondence indices.
        dist_a : array-like, optional
            Geodesic distance matrix on shape A.
        mask_a : array-like, optional
            Binary overlap mask on shape A.

        Returns
        -------
        dict of str → float
            Only metrics whose required inputs were present are included.
        """
        if hasattr(result, "to_dict"):
            inputs = result.to_dict()
        else:
            inputs = {k: v for k, v in result.items() if v is not None}

        for key, val in [
            ("shape_a", shape_a),
            ("shape_b", shape_b),
            ("corr_a", corr_a),
            ("corr_b", corr_b),
            ("dist_a", dist_a),
            ("mask_a", mask_a),
        ]:
            if val is not None:
                inputs[key] = val

        results = {}
        for name, metric in self.metrics.items():
            if all(k in inputs for k in metric.required_inputs):
                results[name] = float(metric(inputs))
        return results
