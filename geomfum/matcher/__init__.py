"""Module containing different matching algorithms."""

from .base import BaseMatcher, DescriptorMatcher, SpatialNearestNeighborMatcher
from .deep_fmap import DeepFMMatcher
from .fmap import FunctionalMapMatcher, ZoomOutMatcher
