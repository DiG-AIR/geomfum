import random

import pytest
from polpo.testing import DataBasedParametrizer

from geomfum.descriptor.spectral import (
    HeatKernelSignature,
    WaveKernelSignature,
    LandmarkHeatKernelSignature,
    LandmarkWaveKernelSignature,
)
from tests.cases.cmp import SpectralDescriptorCmpCase
from tests.utils import landmark_randomly

from .data.spectral import SpectralDescriptorCmpData


@pytest.fixture(
    scope="class",
    params=[
        ("hks", False),
        ("hks", True),
        ("wks", False),
        ("wks", True),
        ("l-hks", False),
        ("l-hks", True),
        ("l-wks", False),
        ("l-wks", True),
    ],
)
def spectral_descriptors(request):
    descr_type, scale = request.param

    n_domain = random.randint(2, 5)
    request.cls.spectrum_size = spectrum_size = random.randint(3, 5)

    testing_data = request.cls.testing_data
    shapes = testing_data.shapes

    shapes.set_spectrum_finder(spectrum_size=spectrum_size)
    shapes.set_landmarks(landmark_randomly)

    if descr_type == "hks":
        descriptor_a = HeatKernelSignature(n_domain=n_domain, scale=scale)
        descriptor_b = HeatKernelSignature.from_registry(scale=scale, n_domain=n_domain)

    elif descr_type == "wks":
        descriptor_a = WaveKernelSignature(scale=scale, n_domain=n_domain)
        descriptor_b = WaveKernelSignature.from_registry(scale=scale, n_domain=n_domain)

    elif descr_type == "l-hks":
        descriptor_a = LandmarkHeatKernelSignature(n_domain=n_domain, scale=scale)
        descriptor_b = LandmarkHeatKernelSignature.from_registry(
            scale=scale, n_domain=n_domain
        )

    elif descr_type == "l-wks":
        descriptor_a = LandmarkWaveKernelSignature(scale=scale, n_domain=n_domain)
        descriptor_b = LandmarkWaveKernelSignature.from_registry(
            scale=scale, n_domain=n_domain
        )

    else:
        raise ValueError(f"Unknown descriptor type: {descr_type}")

    request.cls.descriptor_a = descriptor_a
    request.cls.descriptor_b = descriptor_b


@pytest.mark.usefixtures("data_check", "spectral_descriptors")
class TestSpectralDescriptorCmp(
    SpectralDescriptorCmpCase, metaclass=DataBasedParametrizer
):
    testing_data = SpectralDescriptorCmpData()
