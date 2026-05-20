"""Base classes for shape matchers."""

import abc
from dataclasses import dataclass

import gsops.backend as gs

from geomfum.convert import BaseNeighborFinder, NeighborFinder
from geomfum.descriptor.pipeline import DescriptorPipeline, L2InnerNormalizer
from geomfum.descriptor.spectral import WaveKernelSignature


@dataclass
class CorrespondenceResult:
    """Result of a matching operation (for both Matcher and Model).

    This is the unified output format for all correspondence methods,
    including classical functional map matchers and learning-based models.

    Parameters
    ----------
    fmap12 : array-like, shape=[spectrum_size_b, spectrum_size_a]
        Functional map matrix from shape_a to shape_b.
    p2p21 : array-like, shape=[n_vertices_b]
        Point-to-point correspondence from shape_b to shape_a.
        For each vertex i in shape_b, p2p21[i] gives the corresponding
        vertex index in shape_a.
    fmap21 : array-like, shape=[spectrum_size_a, spectrum_size_b], optional
        Functional map matrix from shape_b to shape_a (for bidirectional).
    p2p12 : array-like, shape=[n_vertices_a], optional
        Point-to-point correspondence from shape_a to shape_b (for bidirectional).
    descr_a : array-like, shape=[n_descr, n_vertices_a], optional
        Descriptors on shape_a.
    descr_b : array-like, shape=[n_descr, n_vertices_b], optional
        Descriptors on shape_b.
    refined_fmap12 : array-like, shape=[spectrum_size_b, spectrum_size_a], optional
        Refined functional map matrix (if refinement was applied).
    refined_fmap21 : array-like, shape=[spectrum_size_a, spectrum_size_b], optional
        Refined functional map matrix from B to A (if bidirectional refinement).
    soft_perm_ab : array-like, shape=[n_vertices_a, n_vertices_b], optional
        Soft permutation matrix mapping b vertices to a domain (P12 in RobustFMNet).
        soft_perm_ab[i, j] = probability that vertex i in a corresponds to vertex j in b.
    soft_perm_ba : array-like, shape=[n_vertices_b, n_vertices_a], optional
        Soft permutation matrix mapping a vertices to b domain (P21 in RobustFMNet).
        soft_perm_ba[i, j] = probability that vertex i in b corresponds to vertex j in a.
    overlap_ab : array-like, shape=[n_vertices_a], optional
        Predicted overlap scores (0–1) for shape A. overlap_ab[i] = probability
        that vertex i in A is in the overlap region with shape B.
    overlap_ba : array-like, shape=[n_vertices_b], optional
        Predicted overlap scores (0–1) for shape B. overlap_ba[j] = probability
        that vertex j in B is in the overlap region with shape A.
    """

    fmap12: "gs.ndarray" = None
    p2p21: "gs.ndarray" = None
    fmap21: "gs.ndarray" = None
    p2p12: "gs.ndarray" = None
    descr_a: "gs.ndarray" = None
    descr_b: "gs.ndarray" = None
    refined_fmap12: "gs.ndarray" = None
    refined_fmap21: "gs.ndarray" = None
    soft_perm_ab: "gs.ndarray" = None
    soft_perm_ba: "gs.ndarray" = None
    overlap_ab: "gs.ndarray" = None
    overlap_ba: "gs.ndarray" = None

    def to_dict(self):
        """Convert to dictionary (for backward compatibility).

        Returns
        -------
        dict
            Dictionary with all non-None fields.

        Notes
        -----
        This method avoids using `asdict()` from dataclasses because it
        performs deep copying, which fails for PyTorch tensors that are
        part of the computation graph (non-leaf tensors) during training.
        """
        return {
            k: getattr(self, k)
            for k in self.__dataclass_fields__
            if getattr(self, k) is not None
        }

    @property
    def is_bidirectional(self):
        """Check if result contains bidirectional correspondences.

        Returns
        -------
        bool
            True if fmap21 and p2p12 are available.
        """
        return self.fmap21 is not None and self.p2p12 is not None


class BaseMatcher(abc.ABC):
    """Abstract base class for shape matchers."""

    @abc.abstractmethod
    def __call__(self, shape_a, shape_b):
        """Compute correspondence between two shapes.

        Parameters
        ----------
        shape_a : Shape
            First shape (target for p2p21).
        shape_b : Shape
            Second shape (source for p2p21).

        Returns
        -------
        result : CorrespondenceResult
            Correspondence result containing:
            - p2p21: point-to-point correspondence from B to A
            - fmap12: functional map from A to B (if applicable)
        """


class DescriptorMatcher(BaseMatcher):
    """Descriptor-based matcher using nearest neighbor in descriptor space.

    This matcher directly computes correspondences by:
    1. Computing descriptors/features on both shapes either indicating descriptor or pipeline.
    2. Finding nearest neighbors in the descriptor space
    3. Optionally refining the correspondence using CorrespondenceRefinementPipeline

    This is simpler and faster than the functional map approach,
    but may be less robust for complex deformations.

    Parameters
    ----------
    descriptor_pipeline : DescriptorPipeline, optional
        Descriptor pipeline to compute descriptors. If None, uses default WKS-based pipeline.
    neighbor_finder : NeighborFinder, optional
        Nearest neighbor finder. If None, uses default.
    """

    def __init__(
        self,
        descriptor_pipeline: DescriptorPipeline = None,
        neighbor_finder: BaseNeighborFinder = None,
    ):
        self.descriptor_pipeline = (
            descriptor_pipeline or self._build_default_descriptor_pipeline()
        )
        self.neighbor_finder = neighbor_finder or NeighborFinder(n_neighbors=1)

    def _build_default_descriptor_pipeline(self):
        """Build the default descriptor pipeline.

        Returns
        -------
        pipeline : DescriptorPipeline
        """
        return DescriptorPipeline(
            [WaveKernelSignature(n_domain=200, k=200), L2InnerNormalizer()]
        )

    def __call__(self, shape_a, shape_b, bidirectional=False):
        """Compute correspondence between two shapes.

        Parameters
        ----------
        shape_a : Shape
            First shape (target for p2p21).
        shape_b : Shape
            Second shape (source for p2p21).
        bidirectional : bool
            If True, compute correspondences in both directions.

        Returns
        -------
        result : CorrespondenceResult
            Matching result containing:
            - p2p21: point-to-point correspondence from B to A
            - p2p12: (if bidirectional=True) correspondence from A to B
        """
        # Compute descriptors
        descr_a = self.descriptor_pipeline.apply(shape_a)
        descr_b = self.descriptor_pipeline.apply(shape_b)

        # Find nearest neighbors in descriptor space
        # descr shape is [n_descr, n_vertices], we need [n_vertices, n_descr]
        feat_a = descr_a.T
        feat_b = descr_b.T
        # Find for each vertex in B, the nearest vertex in A (p2p21: B -> A)
        p2p21 = self.neighbor_finder(feat_b, feat_a).flatten()

        # Compute reverse direction if bidirectional
        p2p12 = None
        if bidirectional:
            p2p12 = self.neighbor_finder(feat_a, feat_b).flatten()

        return CorrespondenceResult(
            p2p21=p2p21,
            p2p12=p2p12,
            descr_a=descr_a,
            descr_b=descr_b,
        )


class SpatialNearestNeighborMatcher(BaseMatcher):
    """Matcher based on spatial nearest neighbors.

    This matcher computes correspondences by finding the nearest vertex in
    Euclidean space. It does not use any descriptors or functional maps.

    Parameters
    ----------
    neighbor_finder : BaseNeighborFinder, optional
        Nearest neighbor finder. If None, uses default.
    """

    def __init__(
        self,
        neighbor_finder: BaseNeighborFinder = None,
    ):
        self.neighbor_finder = neighbor_finder or NeighborFinder(n_neighbors=1)

    def __call__(self, shape_a, shape_b, bidirectional=False):
        """Compute correspondence between two shapes.

        Parameters
        ----------
        shape_a : Shape
            First shape (target for p2p21).
        shape_b : Shape
            Second shape (source for p2p21).
        bidirectional : bool
            If True, compute correspondences in both directions.

        Returns
        -------
        result : CorrespondenceResult
            Matching result containing:
            - p2p21: point-to-point correspondence from B to A
            - p2p12: (if bidirectional=True) correspondence from A to B
        """
        # Find for each vertex in B, the nearest vertex in A (p2p21: B -> A)
        p2p21 = self.neighbor_finder(shape_b.vertices, shape_a.vertices).flatten()

        # Compute reverse direction if bidirectional
        p2p12 = None
        if bidirectional:
            p2p12 = self.neighbor_finder(shape_a.vertices, shape_b.vertices).flatten()

        return CorrespondenceResult(
            p2p21=p2p21,
            p2p12=p2p12,
            descr_a=None,
            descr_b=None,
        )
