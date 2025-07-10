"""Models for learning features for functional maps.

References
----------
.. "Deep Geometric Functional Maps: Robust Feature Learning for Shape Correspondence" by Nicolas Donati, Abhishek Sharma, Maks Ovsjanikov.
.. "Deep Functional Maps: Structured Prediction for Dense Shape Correspondence" by O. Litany, T. Remez, E. Rodola, A. Bronstein, M. Bronstein.
"""

import abc

import torch.nn as nn

from geomfum.convert import P2pFromFmConverter
from geomfum.descriptor.learned import FeatureExtractor, LearnedDescriptor
from geomfum.forward_functional_map import ForwardFunctionalMap


class BaseModel(abc.ABC, nn.Module):
    """Base class for all models."""


class FMNet(BaseModel):
    """Functional Map Network Model.

    Parameters
    ----------
    feature_extractor : FeatureExtractor
        Feature extractor to use for the descriptors.
    fmap_module : ForwardFunctionalMap
        Functional map module to use for the forward pass.
    converter : P2pFromFmConverter
        Converter to convert functional maps to point-to-point correspondences.
    """

    def __init__(
        self,
        feature_extractor=FeatureExtractor.from_registry(which="diffusionnet"),
        fmap_module=ForwardFunctionalMap(),
        converter=P2pFromFmConverter(),
        
    ):
        super(FMNet, self).__init__()

        self.feature_extractor = feature_extractor
        self.descriptors_module = LearnedDescriptor(
            feature_extractor=self.feature_extractor
        )
        self.fmap_module = fmap_module
        self.converter = converter

    def forward(self, mesh_a, mesh_b, as_dict=True):
        """Compute the functional map between two shapes.

        Parameters
        ----------
        mesh_a : TriangleMesh or dict
            The first shape, either as a TriangleMesh object or a dictionary containing 'basis', 'evals', and 'pinv'.
        mesh_b : TriangleMesh or dict
            The second shape, either as a TriangleMesh object or a dictionary containing 'basis', 'evals', and 'pinv'.
        as_dict : bool, optional
            If True, returns a dictionary with functional maps and, optionally, point-to-point correspondences.
            If False, returns the functional maps and, optionally, point-to-point correspondences as separate tensors.

        Returns
        -------
            fmap12 : array-like, shape=[..., spectrum_size_b, spectrum_size_a]
                Functional map from shape a to shape b.
            fmap21 : array-like, shape=[..., spectrum_size_a, spectrum_size_b]
                Functional map from shape b to shape a.
            p2p21 : array-like, shape=[..., num_points_b]
                Point-to-point correspondence from shape a to shape b.
            p2p12 : array-like, shape=[..., num_points_a]
                Point-to-point correspondence from shape b to shape a.
        """

        desc_a = self.descriptors_module(mesh_a)
        desc_b = self.descriptors_module(mesh_b)


        fmap12, fmap21 = self.fmap_module(mesh_a, mesh_b, desc_a, desc_b)
        p2p12 = p2p21 = None
        if not self.training:
            p2p21 = self.converter(fmap12, mesh_a.basis, mesh_b.basis)
            p2p12 = self.converter(fmap21, mesh_b.basis, mesh_a.basis)

        if as_dict:
            result = {"fmap12": fmap12, "fmap21": fmap21}
            if not self.training:
                result.update({"p2p12": p2p12, "p2p21": p2p21})
            return result
        else:
            return (
                (fmap12, fmap21, p2p12, p2p21)
                if not self.training
                else (fmap12, fmap21)
            )
