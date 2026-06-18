"""Tools to handle collections of shapes with functional map framework."""

import random
from collections import defaultdict

import gsops.backend as gs
import networkx as nx
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from scipy.optimize import linprog
from tqdm import tqdm

from geomfum.convert import NeighborFinder
from geomfum.numerics.eig import ScipyEigsh
from geomfum.shape.hierarchical import HierarchicalMesh


def _euclidean_fps(vertices, n_samples):
    """Farthest point sampling based on Euclidean distance.

    Parameters
    ----------
    vertices : array-like, shape=[n_vertices, 3]
        Vertex coordinates.
    n_samples : int
        Number of points to sample.

    Returns
    -------
    indices : array-like, shape=[n_samples]
        Indices of the sampled vertices.
    """
    vertices = np.asarray(vertices)
    n_vertices = vertices.shape[0]

    rng = np.random.default_rng()
    indices = [int(rng.integers(n_vertices))]
    dists = np.linalg.norm(vertices - vertices[indices[0]], axis=1)

    for _ in range(n_samples - 1):
        new_index = int(np.argmax(dists))
        indices.append(new_index)
        dists = np.minimum(dists, np.linalg.norm(vertices - vertices[new_index], axis=1))

    return np.asarray(indices)


def _leaf(shape):
    """Return the underlying mesh from a shape, unwrapping hierarchical wrappers."""
    return shape.low if hasattr(shape, "low") else shape


def _fm_from_p2p(p2p, basis_a, basis_b, sub_a=None, sub_b=None):
    """Compute a functional map from a pointwise map, with optional subsampling.

    Parameters
    ----------
    p2p : array-like, shape=[{n_vertices_b, len(sub_b)}]
        Pointwise map, with values indexing the (sub)vertices of shape a.
    basis_a : Basis
        Basis of the source shape, truncated to the desired spectrum size.
    basis_b : Basis
        Basis of the target shape, truncated to the desired spectrum size.
    sub_a : array-like, shape=[size_a]
        Indices of the subsampled vertices of shape a. If None, uses all vertices.
    sub_b : array-like, shape=[size_b]
        Indices of the subsampled vertices of shape b. If None, uses all vertices.

    Returns
    -------
    fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
        Functional map matrix.
    """
    vecs_a = basis_a.vecs if sub_a is None else basis_a.vecs[sub_a]
    vecs_b = basis_b.vecs if sub_b is None else basis_b.vecs[sub_b]

    evects_a_pb = vecs_a[p2p]

    return gs.from_numpy(scipy.linalg.lstsq(vecs_b, evects_a_pb)[0])


def _clb_quad_form(conn, weights, edges, n_shapes, M):
    """Build the block-sparse quadratic form for Consistent Latent Basis computation.

    Parameters
    ----------
    conn : dict
        Functional maps, keyed by edge ``(i, j)``.
    weights : scipy.sparse matrix, shape=[n_shapes, n_shapes]
        Edge weights.
    edges : list[tuple[int, int]]
        Network edges.
    n_shapes : int
        Number of shapes in the network.
    M : int
        Functional map dimension to use.

    Returns
    -------
    W : scipy.sparse.csr_matrix, shape=[n_shapes * M, n_shapes * M]
        Quadratic form for CLB computation.
    """
    grid = [[None for _ in range(n_shapes)] for _ in range(n_shapes)]
    for i in range(n_shapes):
        grid[i][i] = scipy.sparse.csr_matrix((M, M))

    for i, j in edges:
        fm = conn[(i, j)][:M, :M]
        w_ij = weights[i, j]

        grid[i][i] = grid[i][i] + scipy.sparse.csr_matrix(w_ij * (fm.T @ fm))
        grid[j][j] = grid[j][j] + scipy.sparse.csr_matrix(w_ij * np.eye(M))

        if grid[i][j] is None:
            grid[i][j] = scipy.sparse.csr_matrix((M, M))
        grid[i][j] = grid[i][j] - scipy.sparse.csr_matrix(w_ij * fm.T)

        if grid[j][i] is None:
            grid[j][i] = scipy.sparse.csr_matrix((M, M))
        grid[j][i] = grid[j][i] - scipy.sparse.csr_matrix(w_ij * fm)

    return scipy.sparse.bmat(grid, format="csr")


class FunctionalMapNetwork:
    """Functional map network.

    Parameters
    ----------
    shapes : list[Shape]
        Vertices of the graph.
    """

    def __init__(self, shapes):
        # TODO: add iso as flag
        self.shapes = tuple(shapes)
        self.conn = {}

        self._M = None
        self.subsample = None

        self.cycles = None
        self.A = None
        self.A_sub = None

        self.use_icsm = False
        self.cycle_weight = None
        self.edge_weights = None
        self.weights = None

        self.W = None
        self.CLB = None
        self.CCLB = None
        self.cclb_eigenvalues = None

        self.p2p = None

    @property
    def n_shapes(self):
        """Number of shapes in the network.

        Returns
        -------
        n_shapes : int
        """
        return len(self.shapes)

    @property
    def M(self):
        """Shared functional map dimension.

        If not set explicitly, falls back to the size of the first edge's
        functional map.

        Returns
        -------
        M : int
            Size of the functional maps.
        """
        if self._M is not None:
            return self._M

        return self.conn[self.edges[0]].shape[0]

    @M.setter
    def M(self, value):
        self._M = value

    def add_edge(self, i, j, fmap):
        """Add edge.

        Parameters
        ----------
        i : int
            Initial vertex of edge.
        j : int
            End vertex of edge.
        fmap : array-like
            Functional map between S_i and S_j.

        Returns
        -------
        self : FunctionalMapNetwork
        """
        self.conn[(i, j)] = fmap
        return self

    def number_of_edges(self):
        """Return number of edges in the network.

        Returns
        -------
        n_edges : int
        """
        return len(self.conn)

    @property
    def edges(self):
        """Get all edges.

        Returns
        -------
        edges : list[tuple[int; 2]]
        """
        return list(self.conn.keys())
    
    def low_resolution(self, min_n_samples=2000):
        """Wrap each shape in a HierarchicalMesh and update the network.

        Parameters
        ----------
        min_n_samples : int
            Target vertex count for the low-resolution remeshing.
        """
        self.shapes = [
            HierarchicalMesh.from_registry(mesh, min_n_samples=min_n_samples)
            for mesh in tqdm(self.shapes)
        ]

    def compute_spectrum(self, spectrum_size, set_as_basis=True):
        """Compute the Laplacian spectrum for every shape in the network.

        Parameters
        ----------
        spectrum_size : int
            Number of eigenpairs to compute.
        set_as_basis : bool
            Whether to store the result as the shape's basis.
        """
        for mesh in tqdm(self.shapes):
            _leaf(mesh).laplacian.find_spectrum(
                spectrum_size=spectrum_size, set_as_basis=set_as_basis
            )

    def get_fmap(self, i, j):
        """Get functional map for a given edge.

        Parameters
        ----------
        i : int
            Initial vertex of edge.
        j : int
            End vertex of edge.

        Returns
        -------
        self : FunctionalMapNetwork
        """
        return self.conn[(i, j)]

    def to_networkx(self):
        """Get networkx representation.

        Keeps only network topology.

        Returns
        -------
        graph : nx.Digraph
            Topology of network.
        """
        return nx.DiGraph(self.edges)

    def simply_cycles(self):
        """Find simple cycles (elementary circuits) of the network.

        Returns
        -------
        cycles : list[list[int]]
            Cycles represented as a list of nodes.
        """
        return list(nx.simple_cycles(self.to_networkx()))

    def edges_starting_at_index(self, i):
        """Get all the edges starting at shape i.

        Returns
        -------
        edges : list[tuple[int; 2]]
        """
        return filter(self.edges, lambda edge: edge[0] == i)

    def edges_ending_at_index(self, i):
        """Get all the edges ending at shape i.

        Returns
        -------
        edges : list[tuple[int; 2]]
        """
        return filter(self.edges, lambda edge: edge[1] == i)

    def add_fully_connected_random_edges(self, N, seed=None):
        """Add N edges ensuring full connectivity first, without functional maps.

        Parameters
        ----------
        N : int
            Total number of edges to create.
        seed : int or None
            Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)

        num_shapes = len(self.shapes)
        if N < num_shapes - 1:
            raise ValueError("Not enough edges to ensure full connectivity.")

        tree = nx.random_tree(num_shapes, seed=seed)
        for i, j in tree.edges():
            self.conn[(i, j)] = None

        existing_edges = set(self.conn.keys())
        all_possible_edges = set((i, j) for i in range(num_shapes) for j in range(num_shapes) if i != j)
        available_edges = list(all_possible_edges - existing_edges)

        additional_edges_needed = N - len(existing_edges)
        if additional_edges_needed > 0:
            new_edges = random.sample(available_edges, additional_edges_needed)
            for i, j in new_edges:
                self.conn[(i, j)] = None

        return self

    def _check_maps(self):
        """Check that every edge has a functional map set.

        Raises
        ------
        ValueError
            If the network has no edges or some edges have no functional map set.
        """
        if not self.conn:
            raise ValueError("Functional maps should be set.")

        if any(fmap is None for fmap in self.conn.values()):
            raise ValueError("All edges must have a functional map set.")

    def _reset_map_attributes(self):
        """Reset attributes that depend on the functional maps."""
        if self.use_icsm:
            self.use_icsm = False
            self.cycle_weight = None
            self.edge_weights = None
            self.weights = None

        self.W = None
        self.CLB = None
        self.CCLB = None
        self.cclb_eigenvalues = None
        self.p2p = None

    def set_isometries(self, M=None):
        """Symmetrize bidirectional edges.

        For each pair of edges ``(i, j)`` and ``(j, i)``, replaces the one
        whose functional map is further from orthogonal with the transpose
        of the other.

        Parameters
        ----------
        M : int
            Functional map dimension to use for the orthogonality comparison.
            If None, uses ``self.M``.

        Returns
        -------
        self : FunctionalMapNetwork
        """
        if M is None:
            M = self.M

        visited = defaultdict(bool)

        for i, j in self.edges:
            if visited[(i, j)] or (j, i) not in self.conn:
                continue

            fm_ij = self.conn[(i, j)][:M, :M]
            fm_ji = self.conn[(j, i)][:M, :M]

            dist_ij = np.linalg.norm(fm_ij.T @ fm_ij - np.eye(fm_ij.shape[1]))
            dist_ji = np.linalg.norm(fm_ji.T @ fm_ji - np.eye(fm_ji.shape[1]))

            if dist_ij <= dist_ji:
                self.conn[(j, i)] = np.transpose(self.conn[(i, j)])
            else:
                self.conn[(i, j)] = np.transpose(self.conn[(j, i)])

            visited[(j, i)] = True

        self._reset_map_attributes()
        return self

    def set_subsample(self, subsample):
        """Set per-shape vertex subsamples.

        Parameters
        ----------
        subsample : array-like, shape=[n_shapes, size]
            Indices of the vertices to subsample on each shape.

        Returns
        -------
        self : FunctionalMapNetwork
        """
        self.subsample = subsample
        return self

    def compute_subsample(self, size=1000, verbose=False):
        """Compute per-shape vertex subsamples via farthest point sampling.

        Parameters
        ----------
        size : int
            Number of vertices to subsample on each shape.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        self : FunctionalMapNetwork
        """
        if verbose:
            print(f"Computing a {size}-sized subsample for each shape")

        self.subsample = np.zeros((self.n_shapes, size), dtype=int)
        for i, shape in enumerate(self.shapes):
            self.subsample[i] = _euclidean_fps(_leaf(shape).vertices, size)

        return self

    def extract_3_cycles(self):
        """Extract all 3-cycles ``(i, j, k)`` of the network.

        Returns
        -------
        self : FunctionalMapNetwork
        """
        edges = set(self.edges)
        self.cycles = []

        for i in range(self.n_shapes):
            for j in range(i):
                for k in range(j):
                    if (i, j) in edges and (j, k) in edges and (k, i) in edges:
                        self.cycles.append((i, j, k))

            for j in range(i + 1, self.n_shapes):
                for k in range(j + 1, self.n_shapes):
                    if (i, j) in edges and (j, k) in edges and (k, i) in edges:
                        self.cycles.append((i, j, k))

        return self

    def compute_Amat(self):
        """Compute the cycle/edge incidence matrix for ICSM weight optimization.

        Returns
        -------
        self : FunctionalMapNetwork
        """
        edges = self.edges
        edge2ind = {edge: ind for ind, edge in enumerate(edges)}

        self.A = np.zeros((len(self.cycles), len(edges)))
        for cycle_ind, (i, j, k) in enumerate(self.cycles):
            self.A[cycle_ind, edge2ind[(i, j)]] = 1
            self.A[cycle_ind, edge2ind[(j, k)]] = 1
            self.A[cycle_ind, edge2ind[(k, i)]] = 1

        self.A_sub = np.where(self.A.sum(0) > 0)[0]

        return self

    def get_cycle_weight(self, cycle, M=None):
        """Compute the cost of a 3-cycle.

        The cost is the maximum deviation from the identity map when going
        through the complete cycle, starting at each of its 3 shapes.

        Parameters
        ----------
        cycle : tuple[int, int, int]
            Cycle ``(i, j, k)``.
        M : int
            Functional map dimension to use. If None, uses ``self.M``.

        Returns
        -------
        cost : float
            Cycle cost.
        """
        if M is None:
            M = self.M

        i, j, k = cycle

        fm_ij = self.conn[(i, j)][:M, :M]
        fm_jk = self.conn[(j, k)][:M, :M]
        fm_ki = self.conn[(k, i)][:M, :M]

        fm_ii = fm_ij @ fm_jk @ fm_ki
        fm_jj = fm_jk @ fm_ki @ fm_ij
        fm_kk = fm_ki @ fm_ij @ fm_jk

        cost_i = np.linalg.norm(fm_ii - np.eye(M))
        cost_j = np.linalg.norm(fm_jj - np.eye(M))
        cost_k = np.linalg.norm(fm_kk - np.eye(M))

        return max(cost_i, cost_j, cost_k)

    def compute_3cycle_weights(self, M=None):
        """Compute per-cycle and per-edge costs for ICSM optimization.

        Parameters
        ----------
        M : int
            Functional map dimension to use. If None, uses ``self.M``.

        Returns
        -------
        self : FunctionalMapNetwork
        """
        if M is None:
            M = self.M

        self.cycle_weight = np.array(
            [self.get_cycle_weight(cycle, M=M) for cycle in self.cycles]
        )

        self.edge_weights = np.zeros(len(self.edges))
        self.edge_weights[self.A_sub] = 1 / (
            self.A[:, self.A_sub] * self.cycle_weight[:, None]
        ).sum(0)

        return self

    def optimize_icsm(self, verbose=False):
        r"""Solve the linear program for ICSM edge weights.

        Solves :math:`\min w^\top x` subject to :math:`Ax \geq c` and
        :math:`x \geq 0`. Edges that are not part of any 3-cycle are given
        zero weight.

        Parameters
        ----------
        verbose : bool
            Whether to print progress.

        Returns
        -------
        opt_weights : array-like, shape=[n_edges]
            Non-negative weight for each edge.
        """
        self.compute_3cycle_weights(M=self.M)

        if verbose:
            print("Optimizing cycle weights...")

        res = linprog(
            self.edge_weights,
            A_ub=-self.A,
            b_ub=-self.cycle_weight,
            bounds=(0, float("inf")),
            method="highs-ds",
        )

        opt_weights = np.zeros(len(self.edges))
        opt_weights[self.A_sub] = res.x[self.A_sub]

        return opt_weights

    def set_weights(self, weights=None, weight_type="icsm", verbose=False):
        """Set edge weights for Consistent Latent Basis computation.

        Parameters
        ----------
        weights : scipy.sparse matrix, shape=[n_shapes, n_shapes]
            Precomputed edge weights. If given, ``weight_type`` is ignored.
        weight_type : str
            "icsm" or "adjacency". Ignored if ``weights`` is given.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        self : FunctionalMapNetwork
        """
        if weights is not None:
            self.use_icsm = False
            self.weights = weights
            return self

        edges = self.edges
        rows = [edge[0] for edge in edges]
        cols = [edge[1] for edge in edges]

        if weight_type == "icsm":
            self.use_icsm = True

            if self.cycles is None:
                if verbose:
                    print("Computing cycle information")
                self.extract_3_cycles()
                self.compute_Amat()

            weight_arr = self.optimize_icsm(verbose=verbose)
            median_val = np.median(weight_arr[self.A_sub])
            if np.isclose(median_val, 0, atol=1e-4):
                weight_arr = weight_arr / np.mean(weight_arr[self.A_sub])
            else:
                weight_arr = weight_arr / median_val

            new_w = np.exp(-np.square(weight_arr) / 2)

            self.weights = scipy.sparse.csr_matrix(
                (new_w, (rows, cols)), shape=(self.n_shapes, self.n_shapes)
            )

        elif weight_type == "adjacency":
            self.use_icsm = False

            values = np.ones(len(edges))
            self.weights = scipy.sparse.csr_matrix(
                (values, (rows, cols)), shape=(self.n_shapes, self.n_shapes)
            )

        else:
            raise ValueError(
                f"`weight_type` should be 'icsm' or 'adjacency', not {weight_type!r}."
            )

        return self

    def compute_W(self, M=None, verbose=False):
        """Compute the Consistent Latent Basis quadratic form.

        Parameters
        ----------
        M : int
            Functional map dimension to use. If None, uses ``self.M``.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        self : FunctionalMapNetwork
        """
        self._check_maps()

        if self.weights is None:
            self.set_weights(verbose=verbose)

        if M is not None:
            self.M = M

        self.W = _clb_quad_form(
            self.conn, self.weights, self.edges, self.n_shapes, self.M
        )
        return self

    def compute_CLB(self, equals_id=False, verbose=False):
        """Compute the Consistent Latent Basis (CLB).

        Parameters
        ----------
        equals_id : bool
            If True, the generalized eigenproblem uses the identity as the
            mass matrix. If False (default), uses ``(1 / n_shapes) * identity``.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        self : FunctionalMapNetwork
        """
        if self.W is None:
            self.compute_W(verbose=verbose)

        if verbose:
            print(f"Computing {self.M} CLB eigenvectors...")

        eig_solver = ScipyEigsh(spectrum_size=self.M, sigma=-1e-6, which="LM")

        if equals_id:
            eigenvalues, eigenvectors = eig_solver(self.W)
        else:
            M_mat = (1 / self.n_shapes) * scipy.sparse.eye(self.W.shape[0])
            eigenvalues, eigenvectors = eig_solver(self.W, M=M_mat)

        eigenvalues[0] = 0

        self.CLB = eigenvectors.reshape((self.n_shapes, self.M, self.M))
        return self

    def compute_CCLB(self, m, verbose=False):
        """Compute the Canonical Consistent Latent Basis (CCLB).

        Parameters
        ----------
        m : int
            Size of the CCLB to compute.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        self : FunctionalMapNetwork
        """
        if self.CLB is None:
            self.compute_CLB(verbose=verbose)

        E_mat = np.zeros((m, m))
        for i, shape in enumerate(self.shapes):
            Y = self.CLB[i, :, :m]
            evals = _leaf(shape).basis.truncate(self.M).vals
            E_mat += Y.T @ (evals[:, None] * Y)

        eigenvalues, eigenvectors = scipy.linalg.eig(E_mat, b=self.n_shapes * np.eye(m))

        eigenvalues = np.real(eigenvalues)
        sorting = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sorting]
        eigenvectors = np.real(eigenvectors)[:, sorting]

        self.cclb_eigenvalues = eigenvalues
        self.CCLB = np.array(
            [self.CLB[i, :, :m] @ eigenvectors for i in range(self.n_shapes)]
        )

        return self

    def get_CSD(self, i):
        """Get the Characteristic Shape Difference (CSD) operators for shape i.

        Parameters
        ----------
        i : int
            Shape index.

        Returns
        -------
        csd_area : array-like, shape=[m, m]
            Area CSD expressed in the latent space.
        csd_conformal : array-like, shape=[m, m]
            Conformal CSD expressed in the latent space.
        """
        fm = self.CCLB[i]

        csd_area = fm.T @ fm

        evals = _leaf(self.shapes[i]).basis.truncate(self.M).vals
        csd_conformal = (
            np.linalg.pinv(np.diag(self.cclb_eigenvalues)) @ fm.T @ (evals[:, None] * fm)
        )

        return csd_area, csd_conformal

    def get_LB(self, i, complete=True):
        """Get the latent basis for shape i.

        Parameters
        ----------
        i : int
            Shape index.
        complete : bool
            If False, restricts to the ``self.subsample[i]`` vertices.

        Returns
        -------
        latent_basis : array-like, shape=[{n_vertices_i, subsample_size}, m]
            Latent basis.
        """
        cclb = self.CCLB[i]
        vecs = _leaf(self.shapes[i]).basis.truncate(self.M).vecs

        if not complete and self.subsample is not None:
            vecs = vecs[self.subsample[i]]

        return vecs @ cclb

    def compute_p2p(self, complete=True):
        """Compute pointwise maps from the CCLB latent embeddings.

        Parameters
        ----------
        complete : bool
            If False, computes pointwise maps between the ``self.subsample``
            vertices of each shape.

        Returns
        -------
        self : FunctionalMapNetwork
        """
        neighbor_finder = NeighborFinder(n_neighbors=1)

        lb_cache = {}

        def _get_lb(i):
            if i not in lb_cache:
                lb_cache[i] = self.get_LB(i, complete=complete)
            return lb_cache[i]

        self.p2p = {}
        for i, j in self.edges:
            lb_i = _get_lb(i)
            lb_j = _get_lb(j)

            self.p2p[(i, j)] = neighbor_finder(lb_j, lb_i).flatten()

        return self

    def compute_maps(self, M, complete=True):
        """Convert pointwise maps to functional maps of size M.

        Parameters
        ----------
        M : int
            Size of the functional maps to compute.
        complete : bool
            If False, uses ``self.subsample`` for the conversion.

        Returns
        -------
        self : FunctionalMapNetwork
        """
        self.M = M

        use_sub = not complete and self.subsample is not None

        for i, j in self.edges:
            basis_a = _leaf(self.shapes[i]).basis.truncate(M)
            basis_b = _leaf(self.shapes[j]).basis.truncate(M)

            sub_a = self.subsample[i] if use_sub else None
            sub_b = self.subsample[j] if use_sub else None

            self.conn[(i, j)] = _fm_from_p2p(
                self.p2p[(i, j)], basis_a, basis_b, sub_a=sub_a, sub_b=sub_b
            )

        self._reset_map_attributes()
        return self

    def zoomout_iteration(
        self,
        cclb_size,
        M_init,
        M_final,
        isometric=True,
        weight_type="icsm",
        equals_id=False,
        complete=False,
        verbose=False,
    ):
        """Perform one iteration of Consistent ZoomOut refinement.

        Parameters
        ----------
        cclb_size : int
            Size of the CCLB to compute.
        M_init : int
            Functional map dimension at the start of the iteration.
        M_final : int
            Functional map dimension at the end of the iteration.
        isometric : bool
            Whether to symmetrize bidirectional edges before refinement
            (ConsistentZoomout-iso strategy).
        weight_type : str
            "icsm" or "adjacency".
        equals_id : bool
            Whether the CLB optimization uses the identity or
            ``n_shapes * identity`` as the generalized eigenproblem mass matrix.
        complete : bool
            Whether pointwise and functional maps are computed using all
            vertices instead of ``self.subsample``.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        self : FunctionalMapNetwork
        """
        if isometric:
            self.set_isometries(M=M_init)

        if weight_type == "icsm":
            self.set_weights(weight_type=weight_type, verbose=verbose)
        elif self.weights is None:
            self.set_weights(weight_type="adjacency", verbose=verbose)

        self.compute_W(M=M_init, verbose=verbose)
        self.compute_CLB(equals_id=equals_id, verbose=verbose)
        self.compute_CCLB(cclb_size, verbose=verbose)
        self.compute_p2p(complete=complete)
        self.compute_maps(M_final, complete=complete)

        return self

    def zoomout_refine(
        self,
        nit=10,
        step=1,
        subsample=1000,
        isometric=True,
        weight_type="icsm",
        M_init=None,
        cclb_ratio=0.9,
        equals_id=False,
        verbose=False,
    ):
        """Refine the functional maps via Consistent ZoomOut.

        Parameters
        ----------
        nit : int
            Number of zoomout iterations.
        step : int
            Functional map dimension increase at each iteration.
        subsample : int or array-like or None
            Size of the per-shape vertex subsample, an explicit
            ``(n_shapes, size)`` index array, or ``None``/``0`` to use all
            vertices.
        isometric : bool
            Whether to use the reduced space strategy of ConsistentZoomout-iso.
        weight_type : str
            "icsm" or "adjacency".
        M_init : int
            Initial functional map dimension. If None, uses ``self.M``.
        cclb_ratio : float
            Size of the CCLB as a ratio of the current dimension M.
        equals_id : bool
            Whether the CLB optimization uses the identity or
            ``n_shapes * identity`` as the generalized eigenproblem mass matrix.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        self : FunctionalMapNetwork
        """
        self._check_maps()

        if (
            np.issubdtype(type(subsample), np.integer) and subsample == 0
        ) or subsample is None:
            use_sub = False
            self.subsample = None
        else:
            use_sub = True
            if np.issubdtype(type(subsample), np.integer):
                self.compute_subsample(size=subsample, verbose=verbose)
            else:
                self.set_subsample(subsample)

        if M_init is not None:
            self.M = M_init

        for _ in range(nit - 1):
            new_M = self.M + step
            cclb_size = int(cclb_ratio * self.M)

            self.zoomout_iteration(
                cclb_size,
                self.M,
                new_M,
                isometric=isometric,
                weight_type=weight_type,
                equals_id=equals_id,
                complete=not use_sub,
                verbose=verbose,
            )

        return self
