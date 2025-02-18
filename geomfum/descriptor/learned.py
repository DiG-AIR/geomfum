"""
In this file we implement the classes to use descriptors that are computed from parametric feature extractors.
"""
import abc
from geomfum._registry import (
    LearnedDescriptorsRegistry,
    WhichRegistryMixins,
)

class LearnedDescriptor(WhichRegistryMixins):
    """Descriptor representing the output of a feature extractor."""
    _Registry = LearnedDescriptorsRegistry