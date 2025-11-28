"""Optimization of the functional map with a forward pass."""

import abc

import gsops.backend as gs
import torch
import torch.nn as nn


class ForwardFunctionalMap(abc.ABC, nn.Module):
    """Class for the forward pass of the functional map.

    Parameters
    ----------
    lmbda : float
        Weight of the mask (default: 1e3).
    resolvent_gamma: float
        Resolvant of the regularized functional map (default: 1).
    bijective: bool
        Whether we compute the map in both the directions (default: True).
    fmap_shape: tuple, optional
        Shape of fmap12, i.e (spectrum_size_b, spectrum_size_a). If None, the shape is inferred from the input shapes.
    """

    def __init__(self, lmbda=1e3, resolvent_gamma=1, bijective=True, fmap_shape=None):
        super(ForwardFunctionalMap, self).__init__()
        self.lmbda = lmbda
        self.resolvent_gamma = resolvent_gamma
        self.bijective = bijective
        self.fmap_shape = fmap_shape

    def _compute_functional_map(self, sdescr_a, sdescr_b, mask):
        """Compute the functional map between two shapes.

        Parameters
        ----------
        sdescr_a : array-like, shape=[..., spectrum_size_a]
            Spectral descriptors on first basis.
        sdescr_b : array-like, shape=[..., spectrum_size_b]
            Spectral descriptors on second basis.
        mask: array-like, shape=[..., spectrum_size_b, spectrum_size_a]
            Mask for the functional map.

        Returns
        -------
            fmap12 : array-like, shape=[..., spectrum_size_b, spectrum_size_a]
                Functional map from shape a to shape b.
        """
        At_A = sdescr_a.T @ sdescr_a
        Bt_A = sdescr_b.T @ sdescr_a

        fmap = []
        for i in range(mask.shape[0]):
            if self.lmbda == 0:
                map_row = gs.linalg.inv(At_A) @ Bt_A[i, :].reshape(-1, 1)
            else:
                MASK_i = gs.diag(mask[i, :].flatten())
                map_row = gs.linalg.inv(At_A + self.lmbda * MASK_i) @ Bt_A[
                    i, :
                ].reshape(-1, 1)
            fmap.append(map_row.T)

        fmap = gs.concatenate(fmap, 0)

        return fmap

    def __call__(self, mesh_a, mesh_b, descr_a, descr_b):
        """Compute the functional map between two shapes.

        Parameters
        ----------
        mesh_a : TriangleMesh
            Mesh object representing the first shape.
        mesh_b : TriangleMesh
            Mesh object representing the second shape.
        descr_a : array-like, shape=[D, ...]
            Spectral descriptors on the first shape.
        descr_b : array-like, shape=[D, ...]
            Spectral descriptors on the second shape.

        Returns
        -------
        fmap_12 : array-like, shape[spectrum_size_b, spectrum_size_a]
            Functional map from shape a to shape b.
        fmap_21: array-like, shape=[spectrum_size_a, spectrum_size_b] or None
            Functional map from shape b to shape a if bijective, otherwise None.
        """
        if self.fmap_shape is not None:
            mesh_a.basis.use_k = self.fmap_shape[1]
            mesh_b.basis.use_k = self.fmap_shape[0]

        evals_a = mesh_a.basis.vals
        sdescr_a = mesh_a.basis.project(descr_a)
        evals_b = mesh_b.basis.vals
        sdescr_b = mesh_b.basis.project(descr_b)

        mask = self._compute_mask(evals_a, evals_b, self.resolvent_gamma)
        fmap_12 = self._compute_functional_map(sdescr_a, sdescr_b, mask)
        fmap_21 = None
        if self.bijective:
            mask = self._compute_mask(evals_b, evals_a, self.resolvent_gamma)
            fmap_21 = self._compute_functional_map(sdescr_b, sdescr_a, mask)
        return fmap_12, fmap_21

    def _compute_mask(self, evals_a, evals_b, resolvant_gamma):
        """Compute the mask for the functional map.

        Parameters
        ----------
        evals_a : array-like, shape=[..., spectrum_size_a]
            Eigenvalues of the first shape.
        evals_b : array-like, shape=[..., spectrum_size_b]
            Eigenvalues of the second shape.
        resolvant_gamma : float
            Resolvent of the regularized functional map.

        Returns
        -------
        mask : array-like, shape=[..., spectrum_size_b, spectrum_size_a]
            Mask for the functional map.
        """
        evals_a = gs.array(evals_a)
        evals_b = gs.array(evals_b)

        scaling_factor = max(max(evals_a), max(evals_b))
        evals_a, evals_b = evals_a / scaling_factor, evals_b / scaling_factor
        evals_gamma_a = gs.power(evals_a, resolvant_gamma)[None, :]
        evals_gamma_b = gs.power(evals_b, resolvant_gamma)[:, None]
        M_re = evals_gamma_b / (gs.square(evals_gamma_b) + 1) - evals_gamma_a / (
            gs.square(evals_gamma_a) + 1
        )
        M_im = 1 / (gs.square(evals_gamma_b) + 1) - 1 / (gs.square(evals_gamma_a) + 1)
        return gs.square(M_re) + gs.square(M_im)
