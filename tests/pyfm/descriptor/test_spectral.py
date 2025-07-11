import random

import numpy as np
import pyFM.signatures as sg
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


def _pyfm_hks(num_T, use_landmarks=False):
    def hks(_cls, mesh, domain=None):
        landmarks = mesh.landmark_indices if use_landmarks else None

        if domain is None:
            return sg.mesh_HKS(
                mesh,
                num_T=num_T,
                landmarks=landmarks,
            ).T

        else:
            if use_landmarks:
                return sg.lm_HKS(
                    mesh.eigenvalues,
                    mesh.eigenvectors,
                    landmarks,
                    domain,
                    scaled=True,
                ).T

            return sg.HKS(
                mesh.eigenvalues,
                mesh.eigenvectors,
                domain,
                scaled=True,
            ).T

    return hks


def _pyfm_wks(num_T, sigma, use_landmarks=False):
    def wks(_cls, mesh, domain=None):
        landmarks = mesh.landmark_indices if use_landmarks else None

        if domain is None:
            abs_ev = sorted(np.abs(mesh.eigenvalues))
            e_min, e_max = np.log(abs_ev[1]), np.log(abs_ev[-1])
            e_min += 2 * sigma
            e_max -= 2 * sigma
            domain = np.linspace(e_min, e_max, num_T)

        if use_landmarks:
            return sg.lm_WKS(
                mesh.eigenvalues,
                mesh.eigenvectors,
                landmarks,
                domain,
                sigma,
                scaled=True,
            ).T

        return sg.WKS(
            mesh.eigenvalues,
            mesh.eigenvectors,
            domain,
            sigma,
            scaled=True,
        ).T

    return wks


@pytest.fixture(
    scope="class",
    params=[
        ("hks", False),
        ("hks", True),
        ("wks", False),
        ("wks", True),
    ],
)
def spectral_descriptors(request):
    descr_type, use_landmarks = request.param

    n_domain = random.randint(2, 5)
    request.cls.spectrum_size = spectrum_size = random.randint(3, 5)

    testing_data = request.cls.testing_data
    shapes = testing_data.shapes

    shapes.set_spectrum_finder(spectrum_size=spectrum_size)

    if use_landmarks:
        shapes.set_landmarks(landmark_randomly)

    if descr_type == "hks":
        if not use_landmarks:
            descriptor_a = HeatKernelSignature.from_registry(
                scale=True, n_domain=n_domain
            )
        else:
            descriptor_a = LandmarkHeatKernelSignature.from_registry(
                scale=True, n_domain=n_domain
            )
        descriptor_b = _pyfm_hks(num_T=n_domain, use_landmarks=use_landmarks)

    elif descr_type == "wks":
        sigma = np.random.uniform(low=0.1, high=2.0, size=1)[0]
        if not use_landmarks:
            descriptor_a = WaveKernelSignature.from_registry(
                scale=True, n_domain=n_domain, sigma=sigma
            )
        else:
            descriptor_a = LandmarkWaveKernelSignature.from_registry(
                scale=True, n_domain=n_domain, sigma=sigma
            )
        descriptor_b = _pyfm_wks(
            num_T=n_domain, sigma=sigma, use_landmarks=use_landmarks
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
