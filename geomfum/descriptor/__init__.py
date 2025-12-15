"""Descriptors Module. This module contains various shape descriptors used in Geomfum. Including spectral descriptors, distance from landmarks, and feature extractors."""

import geomfum.wrap as _wrap  # for register

from ._base import Descriptor, DistanceFromLandmarksDescriptor, SpectralDescriptor
