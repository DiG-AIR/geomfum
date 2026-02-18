"""Implementation of the learned descriptor.

The learned descriptor is a descriptor that uses a neural network to compute features.
"""

import abc

import gsops.backend as gs
import torch
import torch.nn as nn

from geomfum._registry import FeatureExtractorRegistry, WhichRegistryMixins
from geomfum.descriptor._base import Descriptor


class BaseFeatureExtractor(abc.ABC):
    """Abstract base class for neural network feature extractors."""

    def __init__(self):
        super().__init__()
        self.model = None  # Placeholder, must be set in subclass

    def load_from_path(self, path):
        """Load model parameters from the provided file path.

        Parameters
        ----------
        path : str
            Path to the saved model parameters
        """
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found: {path}") from e
        except Exception as e:
            raise ValueError(f"Failed to load model from {path}: {e}") from e

    def save(self, path):
        """Save model parameters to the specified file path.

        Parameters
        ----------
        path : str
            Path to the saved model parameters
        """
        torch.save(self.model.state_dict(), path)


class FeatureExtractor(WhichRegistryMixins):
    """Factory for creating feature extractors from registry."""

    _Registry = FeatureExtractorRegistry


class LearnedDescriptor(Descriptor, abc.ABC, nn.Module):
    """Descriptor computed using neural network feature extractors.

    Parameters
    ----------
    feature_extractor: Feature Extractor
        Feature extractor to use.

    """

    def __init__(self, feature_extractor=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor.from_registry(
                which="diffusionnet"
            )

    def forward(self, shape):
        """Compute descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape.

        Returns
        -------
        features : array-like, shape=[..., n_features, n_vertices]
            Descriptors of the shape, where `n_features` is the number of features extracted by the feature extractor.
        """
        features = self.feature_extractor(shape)
        features = gs.array(features.squeeze().double()).T

        return features
