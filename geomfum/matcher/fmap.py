"""Functional map based matchers."""

from geomfum.convert import P2pFromFmConverter
from geomfum.descriptor.pipeline import (
    ArangeSubsampler,
    DescriptorPipeline,
    L2InnerNormalizer,
)
from geomfum.descriptor.spectral import WaveKernelSignature
from geomfum.functional_map import FunctionalMap
from geomfum.matcher.base import BaseMatcher, CorrespondenceResult
from geomfum.refine import Refiner, ZoomOut


class FunctionalMapMatcher(BaseMatcher):
    """Functional map based matcher with configurable pipeline.

    This matcher follows the standard functional map pipeline:
    1. Compute basis (Laplacian eigenfunctions) for both shapes
    2. Compute descriptors (WKS, landmarks if available)
    3. Optimize functional map with various constraints
    4. Convert to point-to-point correspondence

    Parameters
    ----------
    fmap_size : int
        Number of eigenfunctions to use for the functional map optimization.
    descriptor_pipeline : DescriptorPipeline, optional
        Descriptor pipeline to compute descriptors. If None, builds from descriptor.
    fmap_optimizer : FunctionalMap, optional
        Optimizer for functional map. If None, uses default.
    p2p_converter : P2pFromFmConverter, optional
        Converter from functional map to point-to-point. If None, uses default.
    """

    def __init__(
        self,
        fmap_size: int = 30,
        descriptor_pipeline: DescriptorPipeline = None,
        fmap_optimizer: FunctionalMap = None,
        p2p_converter: P2pFromFmConverter = None,
    ):
        self.fmap_size = fmap_size
        self.descriptor_pipeline = (
            descriptor_pipeline or self._build_default_descriptor_pipeline()
        )
        self.fmap_optimizer = fmap_optimizer or FunctionalMap()
        self.p2p_converter = p2p_converter or P2pFromFmConverter()

    def _build_default_descriptor_pipeline(self):
        """Build the default descriptor pipeline.

        Returns
        -------
        pipeline : DescriptorPipeline
        """
        return DescriptorPipeline(
            [
                WaveKernelSignature(n_domain=200, k=200),
                ArangeSubsampler(subsample_step=10),
                L2InnerNormalizer(),
            ]
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
            - fmap12: functional map from A to B
            - p2p21: point-to-point correspondence from B to A
            - fmap21, p2p12: (if bidirectional=True) reverse direction
        """
        # Step 1: Compute descriptors
        descr_a = self.descriptor_pipeline.apply(shape_a)
        descr_b = self.descriptor_pipeline.apply(shape_b)

        # Step 2: Get truncated basis views at fmap_size (no mutation of shape.basis.use_k)
        basis_a = shape_a.basis.truncate(self.fmap_size)
        basis_b = shape_b.basis.truncate(self.fmap_size)

        # Step 3: Optimize functional map (fmap12: A -> B)
        fmap12 = self.fmap_optimizer(basis_a, basis_b, descr_a, descr_b)

        # Step 5: Convert to point-to-point correspondence (p2p21: B -> A)
        p2p21 = self.p2p_converter(fmap12, basis_a, basis_b)

        # Initialize reverse direction as None
        fmap21 = None
        p2p12 = None

        # Step 6: Compute reverse direction if bidirectional
        if bidirectional:
            fmap21 = self.fmap_optimizer(basis_b, basis_a, descr_b, descr_a)
            p2p12 = self.p2p_converter(fmap21, basis_b, basis_a)

        return CorrespondenceResult(
            fmap12=fmap12,
            p2p21=p2p21,
            fmap21=fmap21,
            p2p12=p2p12,
            descr_a=descr_a,
            descr_b=descr_b,
        )

class ZoomOutMatcher(BaseMatcher):
    """ZoomOut functional map matcher.

    This matcher implements the ZoomOut algorithm for functional map optimization.
    It inherits from FunctionalMapMatcher and overrides the optimizer with a ZoomOut implementation.
    """

    def __init__(
        self,
        fmap_size: tuple = (30, 30),
        descriptor_pipeline: DescriptorPipeline = None,
        fmap_optimizer: FunctionalMap = None,
        refiner: Refiner = None,
        p2p_converter: P2pFromFmConverter = None,
    ):
        self.fmap_size = fmap_size
        self.descriptor_pipeline = (
            descriptor_pipeline or self._build_default_descriptor_pipeline()
        )
        self.fmap_optimizer = fmap_optimizer or FunctionalMap()
        self.p2p_converter = p2p_converter or P2pFromFmConverter()
        self.refiner = refiner or ZoomOut()

    def _build_default_descriptor_pipeline(self):
        """Build the default descriptor pipeline.

        Returns
        -------
        pipeline : DescriptorPipeline
        """
        return DescriptorPipeline(
            [
                WaveKernelSignature(n_domain=200, k=200),
                ArangeSubsampler(subsample_step=10),
                L2InnerNormalizer(),
            ]
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
            - fmap12: functional map from A to B
            - p2p21: point-to-point correspondence from B to A
            - fmap21, p2p12: (if bidirectional=True) reverse direction
        """
        # Step 1: Compute descriptors
        descr_a = self.descriptor_pipeline.apply(shape_a)
        descr_b = self.descriptor_pipeline.apply(shape_b)

        # Step 2: Get truncated basis views at fmap_size (no mutation of shape.basis.use_k)
        basis_a = shape_a.basis.truncate(self.fmap_size[1])
        basis_b = shape_b.basis.truncate(self.fmap_size[0])

        # Step 3: Optimize functional map (fmap12: A -> B)
        fmap12 = self.fmap_optimizer(basis_a, basis_b, descr_a, descr_b)

        # Step 4: Refine functional map with ZoomOut (larger k views)
        zoom_k_a = self.fmap_size[1] + self.refiner.nit * self.refiner.step[1]
        zoom_k_b = self.fmap_size[0] + self.refiner.nit * self.refiner.step[0]
        if shape_a.basis.full_spectrum_size < zoom_k_a or shape_b.basis.full_spectrum_size < zoom_k_b:
            raise ValueError(
                f"Not enough eigenvalues computed for ZoomOut refinement. "
                f"Required: ({zoom_k_b}, {zoom_k_a}), but got "
                f"({shape_b.basis.full_spectrum_size}, {shape_a.basis.full_spectrum_size}). "
                f"Consider increasing the number of eigenvalues computed for the shapes."
            )
        zoom_basis_a = shape_a.basis.truncate(zoom_k_a)
        zoom_basis_b = shape_b.basis.truncate(zoom_k_b)
        fmap12_refined = self.refiner(fmap12, zoom_basis_a, zoom_basis_b)

        # Step 5: Convert to point-to-point correspondence (p2p21: B -> A)
        p2p21 = self.p2p_converter(fmap12_refined, zoom_basis_a, zoom_basis_b)

        # Initialize reverse direction as None
        fmap21 = None
        p2p12 = None

        # Step 6: Compute reverse direction if bidirectional
        if bidirectional:
            fmap21 = self.fmap_optimizer(basis_b, basis_a, descr_b, descr_a)
            fmap21_refined = self.refiner(fmap21, zoom_basis_b, zoom_basis_a)
            p2p12 = self.p2p_converter(fmap21_refined, zoom_basis_b, zoom_basis_a)

        
        return CorrespondenceResult(
            fmap12=fmap12,
            p2p21=p2p21,
            fmap21=fmap21,
            p2p12=p2p12,
            descr_a=descr_a,
            descr_b=descr_b,
            refined_fmap12=fmap12_refined,
            refined_fmap21=fmap21_refined if bidirectional else None,
        )
