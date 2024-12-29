"""
In this file we implement the classes to use learned descriptors.
"""
from geomfum.descriptor import Descriptor
import abc
from geomfum._registry import (
    LearnedDescriptorsRegistry,
    WhichRegistryMixins,
)

class LearnedDescriptor(WhichRegistryMixins, Descriptor):
    """Descriptor representing the output of a feature extractor."""
    _Registry = LearnedDescriptorsRegistry

    def __init__(self, n_features):
        self.n_features = n_features
        self.features = None
        self.trained = False

    @abc.abstractmethod
    def __call__(self, mesh):
        """Compute descriptor.

        Parameters
        ----------
        basis : mesh (or data).
            Basis.
        """
        @abc.abstractmethod
    def load(self, path):
        """Compute descriptor.

        Parameters
        ----------
        path : pathfile.
            Basis.
        """

