"""Conversion between pointwise and functional maps. In this module we define various converters to go from pointwise maps to functional maps and viceversa."""

import abc

import gsops.backend as gs
import scipy
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

import geomfum.wrap as _wrap  # noqa (for register)
from geomfum._registry import NeighborFinderRegistry, WhichRegistryMixins
from geomfum.neural_adjoint_map import NeuralAdjointMap


class BaseNeighborFinder(abc.ABC):
    """Base class for a Neighbor finder.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors to find.
    """

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors


class NeighborFinder(WhichRegistryMixins, BaseNeighborFinder):
    """Base class for a Neighbor finder.

    A simplified blueprint of ``sklearn.NearestNeighbors`` implementation.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors.
    """

    _Registry = NeighborFinderRegistry

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.sklearn_neighbor_finder = NearestNeighbors(
            n_neighbors=self.n_neighbors, leaf_size=40, algorithm="kd_tree", n_jobs=1
        )

    def __call__(self, X, Y):
        """Return indices of the points in `X` nearest to the points in `Y`.

        Parameters
        ----------
        X : array-like, shape=[n_points_x, n_features]
            Reference points.
        Y : array-like, shape=[n_points_y, n_features]
            Query points.

        Returns
        -------
        neigs : array-like, shape=[n_points_x, n_neighbors]
            Indices of the nearest neighbors in Y for each point in X.
        """
        self.sklearn_neighbor_finder.fit(gs.to_device(Y, "cpu"))
        neigs = self.sklearn_neighbor_finder.kneighbors(
            gs.to_device(X, "cpu"), return_distance=False
        )

        return gs.from_numpy(neigs)


class BaseP2pFromFmConverter(abc.ABC):
    """Pointwise map from functional map."""


class P2pFromFmConverter(BaseP2pFromFmConverter):
    """Pointwise map from functional map.

    Parameters
    ----------
    neighbor_finder : NeighborFinder
        Nearest neighbor finder.
    adjoint : bool
        Whether to use adjoint method.

    References
    ----------
    .. [OCSBG2012] Maks Ovsjanikov, Mirela Ben-Chen, Justin Solomon,
        Adrian Butscher, and Leonidas Guibas.
        “Functional Maps: A Flexible Representation of Maps between
        Shapes.” ACM Transactions on Graphics 31, no. 4 (2012): 30:1-30:11.
        https://doi.org/10.1145/2185520.2185526.
    .. [VM2023] Giulio Viganò  Simone Melzi. “Adjoint Bijective ZoomOut:
        Efficient Upsampling for Learned Linearly-Invariant Embedding.”
        The Eurographics Association, 2023. https://doi.org/10.2312/stag.20231293.
    """

    def __init__(self, neighbor_finder=None, adjoint=False, bijective=False):
        if neighbor_finder is None:
            neighbor_finder = NeighborFinder(n_neighbors=1)
        if neighbor_finder.n_neighbors > 1:
            raise ValueError("Expects `n_neighors = 1`.")

        self.neighbor_finder = neighbor_finder
        self.adjoint = adjoint
        self.bijective = bijective

    def __call__(self, fmap_matrix, basis_a, basis_b):
        """Convert functional map.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        basis_a : Basis,
            Basis of the source shape.
        basis_b : Basis,
            Basis of the target shape.

        Returns
        -------
        p2p : array-like, shape=[{n_vertices_b, n_vertices_a}]
            Pointwise map. ``bijective`` controls shape.
        """
        k2, k1 = fmap_matrix.shape

        if self.adjoint:
            emb1 = basis_a.full_vecs[:, :k1]
            emb2 = basis_b.full_vecs[:, :k2] @ fmap_matrix

        else:
            emb1 = basis_a.full_vecs[:, :k1] @ fmap_matrix.T
            emb2 = basis_b.full_vecs[:, :k2]

        if self.bijective:
            emb1, emb2 = emb2, emb1

        p2p = self.neighbor_finder(emb2, emb1).flatten()

        return p2p


class SoftmaxNeighborFinder(BaseNeighborFinder, nn.Module):
    """Softmax neighbor finder.

    Finds neighbors using softmax regularization.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors.
    tau : float
        Temperature parameter for softmax regularization.
    """

    def __init__(self, n_neighbors=1, tau=0.07):
        BaseNeighborFinder.__init__(self, n_neighbors=n_neighbors)
        nn.Module.__init__(self)
        self.tau = tau

    def __call__(self, X, Y):
        """Return indices of the points in `X` nearest to the points in `Y`.

        Parameters
        ----------
        X : array-like, shape=[n_points_x, n_features]
            Reference points.
        Y : array-like, shape=[n_points_y, n_features]
            Query points.

        Returns
        -------
        neigs : array-like, shape=[n_points_x, n_neighbors]
            Indices of the nearest neighbors in Y for each point in X.
        """
        return self.forward(X, Y)

    def forward(self, X, Y):
        """Find k nearest neighbors using softmax regularization.

        Parameters
        ----------
        X : array-like, shape=[n_points_x, n_features]
            Reference points.
        Y : array-like, shape=[n_points_y, n_features]
            Query points.

        Returns
        -------
        neigs : array-like, shape=[n_points_x, n_neighbors]
            Indices of the nearest neighbors in Y for each point in X.
        """
        P = self.softmax_matrix(X, Y)
        # Get the indices of the top-k (self.n_neighbors) highest values for each row
        indices = torch.topk(P, self.n_neighbors, dim=-1)[1]
        return indices

    def softmax_matrix(self, X, Y):
        """Compute the permutation matrix P as a softmax of the similarity.

        Parameters
        ----------
        X : array-like, shape=[n_points_x, n_features]
            Reference points.
        Y : array-like, shape=[n_points_y, n_features]
            Query points.

        Returns
        -------
        P : array-like, shape=[n_points_x, n_points_y]
            Permutation matrix, where each row sums to 1.
        """
        similarity = torch.mm(X, Y.T)

        P = torch.exp(
            similarity / self.tau
            - torch.logsumexp(similarity / self.tau, dim=-1, keepdim=True)
        )

        return P


class SinkhornP2pFromFmConverter(P2pFromFmConverter):
    """Pointwise map from functional map using Sinkhorn filters.

    Parameters
    ----------
    neighbor_finder : SinkhornKNeighborsFinder
        Nearest neighbor finder.
    adjoint : bool
        Whether to use adjoint method.
    bijective : bool
        Whether to use bijective method. Check [VM2023]_.

    References
    ----------
    .. [PRMWO2021] Gautam Pai, Jing Ren, Simone Melzi, Peter Wonka, and Maks Ovsjanikov.
        "Fast Sinkhorn Filters: Using Matrix Scaling for Non-Rigid Shape Correspondence
        with Functional Maps." Proceedings of the IEEE/CVF Conference on Computer Vision
        and Pattern Recognition (CVPR), 2021, pp. 11956-11965.
        https://hal.science/hal-03184936/document
    """

    def __init__(
        self,
        neighbor_finder=None,
        adjoint=False,
        bijective=False,
    ):
        if neighbor_finder is None:
            neighbor_finder = NeighborFinder.from_registry(which="pot")

        super().__init__(
            neighbor_finder=neighbor_finder,
            adjoint=adjoint,
            bijective=bijective,
        )


class BaseFmFromP2pConverter(abc.ABC):
    """Functional map from pointwise map."""


class FmFromP2pConverter(BaseFmFromP2pConverter):
    """Functional map from pointwise map.

    Parameters
    ----------
    pseudo_inverse : bool
        Whether to solve using pseudo-inverse.
    """

    # TODO: add subsampling
    def __init__(self, pseudo_inverse=False):
        self.pseudo_inverse = pseudo_inverse

    def __call__(self, p2p, basis_a, basis_b):
        """Convert point to point map.

        Parameters
        ----------
        p2p : array-like, shape=[n_vertices_b]
            Pointwise map.
        basis_a : Basis,
            Basis of the source shape.
        basis_b : Basis,
            Basis of the target shape.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        """
        evects1_pb = basis_a.vecs[p2p, :]

        if self.pseudo_inverse:
            return basis_b.vecs.T @ (basis_b._shape.laplacian.mass_matrix @ evects1_pb)

        return gs.from_numpy(scipy.linalg.lstsq(basis_b.vecs, evects1_pb)[0])


class FmFromP2pBijectiveConverter(BaseFmFromP2pConverter):
    """Bijective functional map from pointwise map method.

    References
    ----------
    .. [VM2024] Giulio Viganò  Simone Melzi. Bijective upsampling and learned embedding for point clouds correspondences.
        Computers and Graphics, 2024. https://doi.org/10.1016/j.cag.2024.103985.
    """

    def __init__(self, pseudo_inverse=False):
        self.pseudo_inverse = pseudo_inverse

    def __call__(self, p2p, basis_a, basis_b):
        """Convert point to point map.

        Parameters
        ----------
        p2p : array-like, shape=[n_vertices_a]
            Pointwise map.
        basis_a : Basis,
            Basis of the source shape.
        basis_b : Basis,
            Basis of the target shape.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        """
        evects2_pb = basis_b.vecs[p2p, :]

        if self.pseudo_inverse:
            return gs.linalg.pinv(evects2_pb) @ basis_a.vecs

        return gs.from_numpy(scipy.linalg.lstsq(evects2_pb, basis_a.vecs)[0])


class NamFromP2pConverter(BaseFmFromP2pConverter):
    """Neural Adjoint Map from pointwise map using Neural Adjoint Maps (NAMs)."""

    def __init__(self, iter_max=200, patience=10, min_delta=1e-4, device="cpu"):
        """Initialize the converter.

        Parameters
        ----------
        iter_max : int, optional
            Maximum number of iterations for training the Neural Adjoint Map.
        patience : int, optional
            Number of iterations with no improvement after which training will be stopped.
        min_delta : float, optional
            Minimum change in the loss to qualify as an improvement.
        device : str, optional
            Device to use for the Neural Adjoint Map (e.g., 'cpu' or 'cuda').
        """
        self.iter_max = iter_max
        self.device = device
        self.min_delta = min_delta
        self.patience = patience

    def __call__(self, p2p, basis_a, basis_b, optimizer=None):
        """Convert point to point map.

        Parameters
        ----------
        p2p : array-like, shape=[n_vertices_b]
            Pointwise map.
        basis_a : Basis,
            Basis of the source shape.
        basis_b : Basis,
            Basis of the target shape.
        optimizer : torch.optim.Optimizer, optional
            Optimizer for training the Neural Adjoint Map.

        Returns
        -------
        nam: NeuralAdjointMap , shape=[spectrum_size_b, spectrum_size_a]
            Neural Adjoint Map model.
        """
        evects1_pb = gs.to_torch(basis_a.vecs[p2p, :]).to(self.device).double()
        evects2 = gs.to_torch(basis_b.vecs).to(self.device).double()
        nam = NeuralAdjointMap(
            input_dim=basis_a.spectrum_size,
            output_dim=basis_b.spectrum_size,
            device=self.device,
        ).double()

        if optimizer is None:
            optimizer = torch.optim.Adam(nam.parameters(), lr=0.01, weight_decay=1e-5)

        best_loss = float("inf")
        wait = 0

        for _ in range(self.iter_max):
            optimizer.zero_grad()

            pred = nam(evects1_pb)

            loss = torch.nn.functional.mse_loss(pred, evects2)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss - self.min_delta:
                best_loss = loss.item()
                wait = 0
            else:
                wait += 1
            if wait >= self.patience:
                break

        return nam


class P2pFromNamConverter(BaseP2pFromFmConverter):
    """Pointwise map from Neural Adjoint Map (NAM).

    Parameters
    ----------
    neighbor_finder : NeighborFinder
        Nearest neighbor finder.
    """

    def __init__(self, neighbor_finder=None):
        if neighbor_finder is None:
            neighbor_finder = NeighborFinder(n_neighbors=1)
        if neighbor_finder.n_neighbors > 1:
            raise ValueError("Expects `n_neighors = 1`.")

        self.neighbor_finder = neighbor_finder

    def __call__(self, nam, basis_a, basis_b):
        """Convert neural adjoint map.

        Parameters
        ----------
        nam : NeuralAdjointMap, shape=[spectrum_size_b, spectrum_size_a]
            Nam model.
        basis_a : Basis,
            Basis of the source shape.
        basis_b : Basis,
            Basis of the target shape.

        Returns
        -------
        p2p : array-like, shape=[n_vertices_b]
            Pointwise map.
        """
        k2, k1 = nam.shape

        emb1 = nam(gs.to_torch(basis_a.full_vecs[:, :k2]).to(nam.device).double())
        emb2 = gs.to_torch(basis_b.full_vecs[:, :k1]).to(nam.device).double()

        p2p = self.neighbor_finder(emb2.detach().cpu(), emb1.detach().cpu()).flatten()
        return p2p
