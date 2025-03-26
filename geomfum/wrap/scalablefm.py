import time
from functools import partial

import multiprocess
import numpy as np
import pymeshlab
import scipy.sparse as sparse
from pyFM.mesh import TriMesh as TM
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, RadiusNeighborsRegressor
from tqdm.notebook import tqdm

import geomfum.linalg as la
from geomfum.laplacian import LaplacianSpectrumFinder
from geomfum.sample import NearestNeighborsIndexSampler
from geomfum.shape.hierarchical import HierarchicalShape
from geomfum.shape.point_cloud import PointCloud


class ScalableLaplacianSpectrumFinder(LaplacianSpectrumFinder):
    """Algorithm to find the Laplacian of a mesh.

    Parameters
    ----------
    mollify_factor : float
        Amount of intrinsic mollification to perform.
    """

    def __init__(self, evecs, evals):
        self.evecs = evecs
        self.evals = evals

    def __call__(self, shape):
        """Apply algorithm.

        Parameters
        ----------
        shape : TriangleMesh
            Mesh.

        Returns
        -------
        stiffness_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Stiffness matrix.
        mass_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Diagonal lumped mass matrix.
        """
        return self.evals, self.evecs


class ScalableHierarchicalMesh(HierarchicalShape):
    """Hierarchical mesh from Scalable Functional Maps.

    Based on [MO2023]_.

    Parameters
    ----------
    mesh : TriangleMesh
        High-resolution mesh.
    min_n_samples : int
        Minimum number of vertices in low-resolution approximated mesh.

    References
    ----------
    .. [MO2023] Robin Magnet, Maks Ovsianikov.
        "Scalable Functional Maps.” arXiv.
    """

    def __init__(self, mesh, min_n_samples, params=None):
        if min_n_samples > mesh.n_vertices:
            raise ValueError(
                f"Number of samples ({min_n_samples}) is greater than number of"
                f"vertices of high-resolution mesh ({mesh.n_vertices})"
            )
        if params == None:
            self.params = {
                "dist_ratio": 3,  # rho = dist_ratio * average_radius
                "self_limit": 0.25,  # Minimum value for self weight
                "correct_dist": False,
                "interpolation": "poly",
                "return_dist": True,
                "adapt_radius": True,
                "n_jobs": 10,
                "n_clusters": 100,
                "verbose": True,
            }
        else:
            self.params = params

        # TODO: allows sampler to come from outside
        # TODO: don't think it naming is correct, i.e. not sure min_n_samples
        # is guaranteed
        sampler = NearestNeighborsIndexSampler(n_samples=min_n_samples)
        low = self._scalable_fm(mesh, sampler)

        super().__init__(low=low, high=mesh)

    def _scalable_fm(self, mesh, sampler):
        vertices = mesh.vertices
        faces = mesh.faces
        lmu = LargeMesh()

        rhigh = TM(vertices, faces).process(k=0, intrinsic=True)

        U1, Ab1, Wb1, sub1, distmat1 = lmu.process_mesh(rhigh, sampler, **self.params)
        evals, evecs = lmu.get_approx_spectrum(Wb1, Ab1, verbose=True)

        rlow = PointCloud(vertices[sub1])

        # load evecs and evals
        rlow.laplacian.find_spectrum(spectrum_size=10, set_as_basis=True)
        spectrum_finder = ScalableLaplacianSpectrumFinder(evals, evecs)
        rlow.laplacian.find_spectrum(laplacian_spectrum_finder=spectrum_finder)

        self._rhigh = rhigh
        self._rlow = rlow

        self._baryc_map = U1
        return PointCloud(np.array(rlow.vertices))

    def scalar_low_high(self, scalar):
        """Transfer scalar from low-resolution to high.

        Parameters
        ----------
        scalar : array-like, shape=[..., low.n_vertices]
            Scalar map on the low-resolution shape.

        Returns
        -------
        high_scalar : array-like, shape=[..., high.n_vertices]
            Scalar map on the high-resolution shape.
        """
        return la.matvecmul(self._baryc_map, scalar)


class LargeMesh:
    """
    Class to save the utils functions of Scalble functional maps

    """

    def linear_compact(self, x):
        """
        Linearly decreasing function between 0 and 1
        """
        return 1 - x

    def poly_compact(self, x):
        """
        Polynomial decreasing function between 0 and 1
        """
        return 1 - 3 * x**2 + 2 * x**3

    def exp_compact(self, x):
        """
        Exponential decreasing function between 0 and 1
        """
        return np.exp(1 - 1 / (1 - x**2))

    def local_f(self, vals, rho, chi="exp", return_zeros_indices=False):
        """
        Apply a local function to a set of values

        Parameters:
        ----------------------
        vals : (n,) values
        rho  : float - radius in which the local function is supporter
        chi  : str - type of local function to use: 'exp', 'linear', 'poly
        return_zeros_indices : bool - whether to return zero-values

        Output
        ----------------------
        newv : (n) new values
        zero_inds : (n,) boolean array, where entry i is True iif newv[i]==0
        """
        newv = np.zeros_like(vals)

        # Check indices in radius
        inds = vals < rho

        # Compute local function
        if chi == "exp":
            newv[inds] = self.exp_compact(vals[inds] / rho)
        elif chi == "poly":
            newv[inds] = self.poly_compact(vals[inds] / rho)
        elif chi == "linear":
            newv[inds] = self.linear_compact(vals[inds] / rho)
        else:
            raise ValueError(
                'Only "poly", "exp" and "linear" accepted as local interpolation functions'
            )

        if return_zeros_indices:
            return newv, ~inds
        return newv

    def select_new_inds(self, choices, vert1, rho):
        """
        Given a set of unseen vertices, select some sources from which a new local dijkstra can be
        computed without interference with each other.

        Given a subset of sources chosen among the potential set of new sources, any of the vertex in the set
        which lies at a euclidean distance bigger than rho to the subset won't be visited during dijkstra, and
        can therefore be added.

        This has no guarentee of selecting all the required sources, however it can't select too many


        Parameters:
        ----------------------
        choices : (p,) indices of unseen vertices
        vert1  : (n,3) coordinates of all vertices
        rho  : value of radius

        Output
        ----------------------
        selection : (p,) indices of vertices to use as new sources
        """

        assert choices.ndim == 1, "Problem of dimension"

        vertices_set = vert1[choices]  # (p,3)

        # Select a first random source
        rng = np.random.default_rng()
        inds = [rng.integers(choices.size)]

        # Iteratively add unseen vertices who lie at an euclidean distance bigger than rho
        # to the current selected vertices

        dists = np.linalg.norm(
            vertices_set[inds[0]][None, :] - vertices_set, axis=1
        )  # (p,)
        while np.any(dists > rho):
            newid = np.argmax(dists)
            inds.append(newid)

            new_distance = np.linalg.norm(
                vertices_set[newid][None, :] - vertices_set, axis=1
            )

            dists = np.minimum(dists, new_distance)

        return choices[inds]

    def update_sampled_points(
        self,
        graph,
        subsample,
        unseen_inds,
        rho,
        I,
        J,
        V,
        vert1=None,
        correct_dist=True,
        directed=True,
        fast_update=True,
        verbose=False,
    ):
        """
        Add some unseen vertices to the sample set and compute the distances.

        Parameters:
        ----------------------
        graph       : adjacency matrix of the mesh
        subsample   : (p,) current set of sampled points
        unseen_inds : indices of unseen vertices on the mesh
        rho         : current radius
        I,J,V       : current entries of the sparse distance matrix in COO format
        vert1       : vertices on the shape
        correct_dist: Whether to approximate dijkstra with L2 a posteriori
        directed    : Whether the adjacency is directed or not
        fast_update : Whether to add new vertices by batch (with some sanity check)

        Output
        ----------------------
        I,J,V : new entries of the distance matrix
        subsample : new sampled points
        """
        if correct_dist and (vert1 is None):
            raise ValueError(
                "Can only correct distances if vertex coordinates are given"
            )

        if fast_update and (vert1 is None):
            raise ValueError(
                "Can do fast update of samples if vertex coordinates are given"
            )

        n_samples = subsample.size

        # Loop while some vertices are unseen
        all_seen = len(unseen_inds) == 0
        iteration = 0
        while not all_seen:
            iteration += 1

            if verbose:
                print(
                    f"Iteration {iteration} : " f"{unseen_inds.size} unseen vertices..."
                )

            # Select one or multiple new sources
            if fast_update:
                new_inds = self.select_new_inds(unseen_inds, vert1, rho)
            else:
                new_inds = [unseen_inds[0]]

            if verbose:
                print(f"\tAdding {len(new_inds)} vertices")

            # Add vertices to the sample
            subsample = np.append(subsample, new_inds)

            # Compute distances from the new source
            dists_new = sparse.csgraph.dijkstra(
                graph, directed=directed, indices=new_inds, limit=rho
            )
            I_new, J_new = np.where(dists_new < np.inf)

            # Correct distances if necessary
            if correct_dist:
                V_new = np.linalg.norm(vert1[J_new] - vert1[new_inds[I_new]], axis=1)
            else:
                V_new = dists_new[(I_new, J_new)]

            # Add the values to the (I,J,V) COO entries
            I = np.concatenate([I, I_new + n_samples])
            J = np.concatenate([J, J_new])
            V = np.concatenate([V, V_new])

            n_samples += len(new_inds)

            # Update unseen vertices
            unseen_inds = np.array([x for x in unseen_inds if x not in J_new])
            all_seen = len(unseen_inds) == 0

        return I, J, V, subsample

    def compute_sparse_dijkstra(
        self,
        mesh1,
        subsample,
        rho,
        real_rho=None,
        n_clusters=4,
        n_jobs=1,
        correct_dist=True,
        verbose=True,
    ):
        N = mesh1.n_vertices
        kmeans = KMeans(n_clusters=n_clusters, max_iter=10).fit(
            mesh1.vertices[subsample]
        )

        edges = mesh1.edges

        # print('B')

        I_base = edges[:, 0]  # (p,)
        J_base = edges[:, 1]  # (p,)
        V_base = np.linalg.norm(
            mesh1.vertices[J_base] - mesh1.vertices[I_base], axis=1
        )  # (p,)

        graph = sparse.csc_matrix((V_base, (I_base, J_base)), shape=(N, N))

        In = np.array([], dtype=int)
        Jn = np.array([], dtype=int)
        Vn = np.array([], dtype=float)

        rad_tree = RadiusNeighborsRegressor(
            radius=1.01 * rho, algorithm="ball_tree", leaf_size=40, n_jobs=n_jobs
        ).fit(mesh1.vertices, np.ones(mesh1.n_vertices))

        seen = np.full(mesh1.n_vertices, False)

        iterable = range(n_clusters)
        if verbose:
            iterable = tqdm(iterable)

        for class_index in iterable:
            subsample_sub_test = kmeans.labels_ == class_index  # (m,)

            subsample_sub_inds = np.where(subsample_sub_test)[0]  # (m_curr,)

            if len(subsample_sub_inds) == 0:
                continue

            subsample_curr = subsample[subsample_sub_test]  # (m_curr,)

            # dists, _ = utils.knn_query(mesh1.vertices[subsample_curr], mesh1.vertices, k=1, return_distance=True, n_jobs=n_jobs)  # (n,)

            # vertices_2keep_test = (dists < rho)  # (n,)
            # vertices_2keep = np.where(vertices_2keep_test)[0]  # (n_curr, )

            res_test = rad_tree.radius_neighbors(
                mesh1.vertices[subsample_curr], return_distance=False
            )
            # vertices_2keep = np.unique(np.concatenate([np.unique(x) for x in res_test]))  # (n_curr, )
            vertices_2keep = np.unique(np.concatenate(res_test))  # (n_curr, )

            subgraph = graph[
                np.ix_(vertices_2keep, vertices_2keep)
            ]  # (n_curr, n_curr) sparse

            # sources_in_curr = utils.knn_query(mesh1.vertices[vertices_2keep], mesh1.vertices[subsample_curr], n_jobs=n_jobs)  # (m_curr,)
            # sources_in_curr = np.asarray([np.where(vertices_2keep == x)[0].item() for x in subsample_curr])

            sources_in_curr = np.where(
                np.isin(vertices_2keep, subsample_curr, assume_unique=True)
            )[0]

            dists_curr = sparse.csgraph.dijkstra(
                subgraph, directed=False, indices=sources_in_curr, limit=rho
            )  # (m_curr, n_curr) with many np.inf

            finite_test = dists_curr < np.inf  # (m_curr, n_curr) with many np.inf
            I, J = np.where(finite_test)

            if correct_dist:
                V = np.linalg.norm(
                    mesh1.vertices[vertices_2keep[J]]
                    - mesh1.vertices[subsample[subsample_sub_inds[I]]],
                    axis=1,
                )

                if real_rho is not None:
                    test_dist = V < real_rho
                    I = I[test_dist]
                    J = J[test_dist]
                    V = V[test_dist]

            else:
                V = dists_curr[(I, J)]

            In = np.concatenate([In, subsample_sub_inds[I]])
            Jn = np.concatenate([Jn, vertices_2keep[J]])
            Vn = np.concatenate([Vn, V])

            seen[vertices_2keep[J]] = True
            # print(np.sum(~seen))
            # unseen_new_curr = np.all(~finite_test, axis=0)  # (n_curr,)
            # unseen[vertices_2keep] = np.logical_and(unseen[vertices_2keep], unseen_new_curr)
            # print(unseen.sum())

        unseen_inds = np.where(~seen)[0]

        # print(In.max(), subsample.size, Jn.max(), mesh1.n_vertices)
        # unseen_inds = np.where(np.all(dists == np.inf, axis=0))[0]
        return In, Jn, Vn, unseen_inds

    def compute_sparse_dijkstra_mp(
        self,
        mesh1,
        subsample,
        rho,
        real_rho=None,
        n_clusters=4,
        n_jobs=1,
        correct_dist=True,
        verbose=True,
    ):
        """
        Compute local dijkstra at samples using multiprocessing and a trick using KMeans to help memory consumption.

        Parameters:
        ----------------------
        mesh1       : pyFM TriMesh class
        subsample   : (p,) current set of sampled points
        rho         : current radius
        real_rho    :
        n_clusters  : separate the shape in n_clusters part using KMeans
        correct_dist: Whether to approximate dijkstra with L2 a posteriori
        n_jobs      : number of parallel jobs
        fast_update : Whether to add new vertices by batch (with some sanity check)

        Output
        ----------------------
        I,J,V : new entries of the distance matrix
        unseen_inds : indices of unseen vertices
        """

        # Divide the samples in n_clusters parts
        N = mesh1.n_vertices
        kmeans = KMeans(n_clusters=n_clusters, max_iter=10).fit(
            mesh1.vertices[subsample]
        )

        edges = mesh1.edges

        # Build adjacency matrix
        I_base = edges[:, 0]  # (p,)
        J_base = edges[:, 1]  # (p,)
        V_base = np.linalg.norm(
            mesh1.vertices[J_base] - mesh1.vertices[I_base], axis=1
        )  # (p,)

        graph = sparse.csc_matrix((V_base, (I_base, J_base)), shape=(N, N))

        # Fit a ball tree to the vertices
        rad_tree = RadiusNeighborsRegressor(
            radius=1.01 * rho, algorithm="ball_tree", leaf_size=40, n_jobs=1
        )
        rad_tree.fit(mesh1.vertices, np.ones(mesh1.n_vertices))

        # Run in parallel function `distances_in_cluster` for each cluster
        local_dists = partial(
            self.distances_in_cluster,
            kmeans.labels_,
            mesh1.vertices,
            subsample,
            rad_tree,
            graph,
            rho,
            real_rho=real_rho,
            correct_dist=correct_dist,
        )

        PROCESSES = min(
            n_jobs if n_jobs > 0 else multiprocess.cpu_count(),
            multiprocess.cpu_count(),
            n_clusters,
        )
        with multiprocess.Pool(PROCESSES) as pool:
            results = [pool.apply_async(local_dists, (i,)) for i in range(n_clusters)]
            if verbose:
                res = [r.get() for r in tqdm(results)]
            else:
                res = [r.get() for r in results]

        # Accuùulate I,J,V
        In = np.concatenate([x[0] for x in res])
        Jn = np.concatenate([x[1] for x in res])
        Vn = np.concatenate([x[2] for x in res])

        seen = np.full(mesh1.n_vertices, False)
        seen[Jn] = True
        unseen_inds = np.where(~seen)[0]

        return In, Jn, Vn, unseen_inds

    def build_local_mat_new(
        self,
        mesh1,
        subsample,
        rho,
        update_sample=True,
        fast_update=True,
        interpolation="poly",
        correct_dist=True,
        return_dist=False,
        self_limit=0.1,
        adapt_radius=False,
        batch_size=None,
        n_clusters=4,
        n_jobs=1,
        verbose=False,
    ):
        """
        Compute the local basis (and local distance matrix).

        Parameters:
        ----------------------
        mesh1     : pyFM.TriMesh object with n vertices
        subsample : (p,) indices of current samples
        rho       : radius
        update_sample : whether to add unseen vertices to the sample
        fast_update   : Make the sample update faster by batch-adding vertices
        interpolation  : 'poly', 'linear', 'exp' - type of local function
        correct_dist  : If True, Replace dijkstra dist with euclidean after dijkstra
        return_dist   : If True, return the sparse distance matrix
        self_limit    : Minimum value for self weight
        adapt_radius  : Whether to use the adaptive radius sttrategy
        batch_size    : Size of batches to use
        n_cluster     : Number of cluster to use to first divide the shape (memory issues)
        n_jobs        : number of parallel workers to use

        Output
        ----------------------
        U : (n,p) sparse local functions at each columns
        subsample : indices of sampled points
        distmat : if return_dist is True, the sparse distance matrix (before applying local function)
        """

        N = mesh1.n_vertices
        n_samples_base = subsample.size

        edges = mesh1.edges

        I = edges[:, 0]  # (p,)
        J = edges[:, 1]  # (p,)
        V = np.linalg.norm(mesh1.vertices[J] - mesh1.vertices[I], axis=1)  # (p,)

        graph = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsc()
        if verbose:
            print("Computing First Dijkstra run...")
            start_time = time.time()

        # Compute first run of local dijkstra
        rho_prerun = 3 * rho if correct_dist else rho
        # I, J, V, unseen_inds = compute_sparse_dijkstra(mesh1, subsample, rho_prerun, real_rho=rho, n_clusters=n_clusters, n_jobs=n_jobs, correct_dist=correct_dist, verbose=verbose)
        I, J, V, unseen_inds = self.compute_sparse_dijkstra_mp(
            mesh1,
            subsample,
            rho_prerun,
            real_rho=rho,
            n_clusters=n_clusters,
            n_jobs=n_jobs,
            correct_dist=correct_dist,
            verbose=verbose,
        )

        # unseen_inds = np.array([],dtype=int)
        if verbose:
            print(f"\tDone in {time.time() - start_time:.2f}s\n")

        # Add unseen samples
        if update_sample:
            if verbose:
                print("Update sampled points")
            I, J, V, subsample = self.update_sampled_points(
                graph,
                subsample,
                unseen_inds,
                rho,
                I,
                J,
                V,
                vert1=mesh1.vertices,
                correct_dist=correct_dist,
                directed=False,
                fast_update=fast_update,
                verbose=verbose,
            )

        if verbose and update_sample:
            print(
                f"{subsample.size - n_samples_base:d} vertices have been added to the sample\n"
            )

        # Modify the radius at some points
        if adapt_radius:
            if verbose:
                print("Update radius :")

            # Build the distance matrix
            distmat = sparse.csc_matrix(
                (V, (J, I)), shape=(mesh1.n_vertices, subsample.size)
            )

            # Adapt the necessary radius
            U, dist_data_new, per_vertex_rho = self.adapt_rho(
                distmat,
                subsample,
                rho,
                threshold=self_limit,
                rho_decrease_ratio=2,
                interpolation=interpolation,
                verbose=verbose,
            )
            # Eliminate new zeros added in the mix
            U.eliminate_zeros()

            # Update the distance matrix for same sparsity pattern as U
            if return_dist or update_sample:
                # Remove 0 in the distance matrix (but not on the diagonal)
                distmat.data = dist_data_new
                distmat += sparse.csc_matrix(
                    (
                        np.full(subsample.size, 1e-10),
                        (subsample, np.arange(subsample.size)),
                    ),
                    shape=distmat.shape,
                )

                distmat.eliminate_zeros()

                # Update unseen indices
                unseen_inds = np.where(U.sum(1) == 0)[0]
                if update_sample and len(unseen_inds) > 0:
                    # Find unseen inds and compute new neighbords
                    unseen_inds = np.where(U.sum(1) == 0)[0]
                    newI, newJ, newV, subsample = self.update_sampled_points(
                        graph,
                        subsample,
                        unseen_inds,
                        per_vertex_rho.min(),
                        [],
                        [],
                        [],
                        vert1=mesh1.vertices,
                        correct_dist=correct_dist,
                        directed=False,
                        fast_update=fast_update,
                        verbose=verbose,
                    )

                    U = U.tocoo(copy=False)

                    # Add new values to U with the small rho
                    Vn = np.concatenate(
                        [
                            U.data,
                            self.local_f(newV, per_vertex_rho.min(), chi=interpolation),
                        ]
                    )
                    Jn = np.concatenate([U.row, newJ])
                    In = np.concatenate([U.col, newI])
                    U = sparse.csr_matrix(
                        (Vn, (Jn, In)), shape=(mesh1.n_vertices, subsample.size)
                    )

                    # Update distance matrix accordingly
                    if return_dist:
                        distmat = distmat.tocoo(copy=False)

                        Vn = np.concatenate([distmat.data, newV])
                        Jn = np.concatenate([distmat.row, newJ])
                        In = np.concatenate([distmat.col, newI])
                        distmat = sparse.csr_matrix(
                            (Vn, (Jn, In)), shape=(mesh1.n_vertices, subsample.size)
                        )
                        # distmat += sparse.csc_matrix((newV, (newJ, newI)), shape=(mesh1.n_vertices, subsample.size))

                distmat = distmat.tocsr(copy=False)

            else:
                del distmat

        # If not radius adaptation
        else:
            # Build distance matrix
            U = sparse.csr_matrix((V, (J, I)), shape=(mesh1.n_vertices, subsample.size))
            if return_dist:
                distmat = U.copy()

            # U is a simple transformation of the distance matrix
            U.data = self.local_f(U.data, rho, chi=interpolation)

        # Normalize U
        U = sparse.diags(1 / np.asarray(U.sum(1)).squeeze(), 0) @ U

        if verbose:
            n_nz = np.array((U > 0).sum(1), dtype=int).squeeze()
            print(
                f"Nonzero elements :\n"
                f"\tMean : {n_nz.mean():.2f} +- {n_nz.std():.2f}\n"
                f"\tMin : {n_nz.min():d}\tMax: {n_nz.max():d}"
            )

        if return_dist:
            return U, subsample, distmat
        return U, subsample

    def adapt_rho(
        self,
        distmat,
        subsample,
        rho,
        threshold=0.1,
        rho_decrease_ratio=2,
        interpolation="poly",
        adapt_self=False,
        verbose=False,
    ):
        """
        Modify radius locally.

        Parameters:
        ----------------------
        distmat   : CSC sparse amtrix of distances
        subsample : (p,) indices of current samples
        rho       : radius
        threshold : minimal value of self weight
        rho_decrease_ratio : at each step divide the radius by this value
        interpolation : 'poly', 'linear', 'exp' - type of local function
        adapt_self : deprecated

        Output
        ----------------------
        U : (n,p) sparse local functions at each columns
        dist_data_new : distmat.data but new unseen values are set to 0.
        per_vertex_rho : per-vertex value
        """

        max_self_val = 1 / threshold

        # Apply chi to the distmat
        U_data = self.local_f(distmat.data, rho, chi=interpolation)

        dist_data_new = distmat.data.copy()

        # Build un-normalized U matrix
        U = sparse.csc_matrix(
            (U_data, distmat.indices, distmat.indptr), shape=distmat.shape
        ).tocsr()

        # Indices with too low self weight
        subinds_toadapt = np.where(
            np.asarray(U[subsample].sum(1)).squeeze() > max_self_val
        )[0]
        n_inds2adapt = subinds_toadapt.size

        # Extract the second point witht the biggest influence (biggest being itself with value 1 by definition)
        red_mat = U[subsample[subinds_toadapt]] - sparse.csr_matrix(
            (np.ones(n_inds2adapt), (np.arange(n_inds2adapt), subinds_toadapt)),
            shape=(n_inds2adapt, subsample.size),
        )

        subinds_2change = np.unique(np.asarray(red_mat.argmax(axis=1)).flatten())

        per_vertex_rho = np.full(subsample.size, rho)
        rho_curr = rho
        iteration = 0
        # While still indices to add (and limit radius to rho/2**7)
        while len(subinds_2change) > 0 and iteration < 7:
            iteration += 1
            if verbose:
                print(
                    f"Iteration {iteration} : "
                    f"Modifying {subinds_2change.size} sampled points"
                )

            # compute new rho
            rho_curr /= rho_decrease_ratio

            # reduce radius of samples to modify
            per_vertex_rho[subinds_2change] = rho_curr

            # Extract indices of data points to use (play with sparse matrices)
            data_indices = self.get_data_inds_of_columns_inds_csc(
                distmat.indptr, subinds_2change
            )

            if verbose:
                print(f"\tRecomputing {data_indices.size} values\n")

            # Check new values of local functions (unnormalized)
            res, inds_2rmv = self.local_f(
                distmat.data[data_indices],
                rho_curr,
                chi=interpolation,
                return_zeros_indices=True,
            )

            # update U accordingly
            U_data[data_indices] = res
            dist_data_new[data_indices[inds_2rmv]] = 0

            U = sparse.csc_matrix(
                (U_data, distmat.indices, distmat.indptr), shape=distmat.shape
            ).tocsr()

            # indices with too low self-weight
            subinds_toadapt = np.where(
                np.asarray(U[subsample].sum(1)).squeeze() > max_self_val
            )[0]
            n_inds2adapt = subinds_toadapt.size

            # print(np.asarray(U[subsample].sum(1)).squeeze()[subinds_toadapt])

            # Extract the second point witht the biggest influence (biggest being itself with value 1 by definition)
            red_mat = U[subsample[subinds_toadapt]] - sparse.csr_matrix(
                (np.ones(n_inds2adapt), (np.arange(n_inds2adapt), subinds_toadapt)),
                shape=(n_inds2adapt, subsample.size),
            )

            subinds_2change = np.unique(np.array(red_mat.argmax(axis=1)).flatten())

        if verbose:
            nnz_before = U.nnz
        U.eliminate_zeros()
        if verbose:
            nnz_after = U.nnz
            print(f"Removed {nnz_before-nnz_after} values")

        return U, dist_data_new, per_vertex_rho

    def get_data_inds_of_columns_inds_csc(self, indptr, col_inds):
        """
        Given indptr of a csc matrix and indices of columns, return the indices of entries of the columns
        in the list of data of a csc-matrix
        """
        start_vals = indptr[col_inds]
        n_vals = np.diff(indptr)[col_inds]

        data_inds = np.concatenate(
            [start_vals[i] + np.arange(n_vals[i]) for i in range(len(col_inds))]
        )

        return data_inds

    def build_red_matrices(
        self,
        mesh1,
        subsample,
        dist_ratio=3,
        update_sample=True,
        interpolation="poly",
        correct_dist=True,
        return_dist=False,
        adapt_radius=False,
        self_limit=0.1,
        batch_size=None,
        n_jobs=1,
        n_clusters=4,
        verbose=False,
    ):
        """
        Build matrices U, A_bar; W_bar and distance matrix

        Parameters:
        ----------------------
        mesh1         : pyFM.TriMesh object with n vertices
        subsample     : (p,) indices of current samples
        dist_ratio    : rho = dist_ratio * average_radius
        update_sample : whether to add unseen vertices to the sample
        interpolation : 'poly', 'linear', 'exp' - type of local function
        correct_dist  : If True, Replace dijkstra dist with euclidean after dijkstra
        return_dist   : If True, return the sparse distance matrix
        adapt_radius  : Whether to use the adaptive radius sttrategy
        self_limit    : Minimum value for self weight
        batch_size    : Size of batches to use
        n_clusters    : Number of cluster to use to first divide the shape (memory issues)
        n_jobs        : number of parallel workers to use

        Output
        ----------------------
        U : (n,p) sparse local functions at each columns
        A_bar : U^T A U
        W_bar : U^T W U
        subsample : indices of sampled points
        distmat : if return_dist is True, the sparse distance matrix (before applying local function)
        """
        n_samples = len(subsample)
        avg_radius = np.sqrt(mesh1.area / (np.pi * n_samples))
        rho = dist_ratio * avg_radius

        result_local = self.build_local_mat_new(
            mesh1,
            subsample,
            rho,
            update_sample=update_sample,
            interpolation=interpolation,
            batch_size=batch_size,
            correct_dist=correct_dist,
            return_dist=return_dist,
            adapt_radius=adapt_radius,
            self_limit=self_limit,
            n_jobs=n_jobs,
            n_clusters=n_clusters,
            verbose=verbose,
        )

        if return_dist:
            U, subsample, distmat = result_local
        else:
            U, subsample = result_local

        A_bar = U.T @ mesh1.A @ U
        W_bar = U.T @ mesh1.W @ U

        if return_dist:
            return U, A_bar, W_bar, subsample, distmat
        return U, A_bar, W_bar, subsample

    def process_mesh(
        self,
        mesh1,
        sampler,
        dist_ratio=3,
        update_sample=True,
        interpolation="poly",
        correct_dist=True,
        return_dist=False,
        batch_size=None,
        adapt_radius=False,
        self_limit=0.1,
        n_jobs=1,
        n_clusters=4,
        verbose=False,
    ):
        """
        Build matrices U, A_bar; W_bar and distance matrix

        Parameters:
        ----------------------
        mesh1         : pyFM.TriMesh object with n vertices
        dist_ratio    : rho = dist_ratio * average_radius
        update_sample : whether to add unseen vertices to the sample
        interpolation : 'poly', 'linear', 'exp' - type of local function
        correct_dist  : If True, Replace dijkstra dist with euclidean after dijkstra
        return_dist   : If True, return the sparse distance matrix
        adapt_radius  : Whether to use the adaptive radius sttrategy
        self_limit    : Minimum value for self weight
        batch_size    : Size of batches to use
        n_clusters    : Number of cluster to use to first divide the shape (memory issues)
        n_jobs        : number of parallel workers to use

        Output
        ----------------------
        U : (n,p) sparse local functions at each columns
        A_bar : U^T A U
        W_bar : U^T W U
        subsample : indices of sampled points
        distmat : if return_dist is True, the sparse distance matrix (before applying local function)
        """
        if verbose:
            # TODO: use logger instead
            print(f"Sampling {sampler.n_samples} vertices out of {mesh1.n_vertices}...")
            start_time = time.time()
        subsample = sampler.sample(mesh1)
        if verbose:
            print(
                f"\t{subsample.size} samples extracted in {time.time() - start_time:.2f}s"
            )

        result_red = self.build_red_matrices(
            mesh1,
            subsample,
            dist_ratio=dist_ratio,
            update_sample=update_sample,
            interpolation=interpolation,
            correct_dist=correct_dist,
            return_dist=return_dist,
            adapt_radius=adapt_radius,
            self_limit=self_limit,
            n_jobs=n_jobs,
            n_clusters=n_clusters,
            batch_size=batch_size,
            verbose=verbose,
        )

        return result_red

    def get_approx_spectrum(self, Wb, Ab, k=150, verbose=False):
        """
        Eigendecopositionusing Wbar and Abar
        """
        if verbose:
            print(f"Computing {k} eigenvectors")
            start_time = time.time()
        eigenvalues, eigenvectors = sparse.linalg.eigsh(Wb, k=k, M=Ab, sigma=-0.01)
        if verbose:
            print(f"\tDone in {time.time()-start_time:.2f} s")
        return eigenvalues, eigenvectors

    def distances_in_cluster(
        self,
        kmeans_labels,
        vertices,
        subsample,
        ball_tree,
        graph,
        rho,
        class_index,
        real_rho=None,
        correct_dist=False,
    ):
        """
        Compute local dijkstra in a cluster

        Parameters:
        ----------------------
        kmeans_labels : (p,) labels for kmeans on the samples
        vertices      : (n,3) vertices
        subsample     : (p,) indices of sampled points
        ball_tree     : ball tree on vertices
        graph         : adjacency matrix with distances at edge
        rho           : radius
        class_index   : index to focus on
        real_rho      : Radius to use after correcting distances
        correct_dist  : Whether to correct distances

        Output
        ----------------------
        In, Jn, Vn : values for a sparse distance matrix with sources on the cluster
        """

        # Find which samples are in the current cluster
        subsample_sub_test = kmeans_labels == class_index  # (m,
        subsample_sub_inds = np.where(subsample_sub_test)[0]  # (m_curr,)
        subsample_curr = subsample[subsample_sub_test]  # (m_curr,)

        if len(subsample_sub_inds) == 0:
            return [], [], []

        # Find vertices at distance rho of the cluster
        res_test = ball_tree.radius_neighbors(
            vertices[subsample_curr], return_distance=False
        )

        vertices_2keep = np.unique(np.concatenate(res_test))  # (n_curr, )

        # Take subgraph in memory
        subgraph = graph[
            np.ix_(vertices_2keep, vertices_2keep)
        ]  # (n_curr, n_curr) sparse

        # Find sources in current cluster and compute dijkstra there
        sources_in_curr = np.where(
            np.isin(vertices_2keep, subsample_curr, assume_unique=True)
        )[0]
        dists_curr = sparse.csgraph.dijkstra(
            subgraph, directed=False, indices=sources_in_curr, limit=rho
        )  # (m_curr, n_curr) with many np.inf

        finite_test = dists_curr < np.inf  # (m_curr, n_curr) with many np.inf
        I, J = np.where(finite_test)

        # If needed correct distances
        if correct_dist:
            V = np.linalg.norm(
                vertices[vertices_2keep[J]]
                - vertices[subsample[subsample_sub_inds[I]]],
                axis=1,
            )

            if real_rho is not None:
                test_dist = V < real_rho
                I = I[test_dist]
                J = J[test_dist]
                V = V[test_dist]

        else:
            V = dists_curr[(I, J)]

        # Give back indices for the full shape
        In = subsample_sub_inds[I]
        Jn = vertices_2keep[J]
        Vn = V

        return In, Jn, Vn
