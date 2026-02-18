"""Routines for working with graphs."""

import heapq

from networkx.algorithms.shortest_paths.weighted import _weight_function


def single_source_partial_dijkstra_path_length(graph, source, k, weight="weight"):
    """Compute shortest-path distances from a source node to the k closest nodes.

    Based on cumulative path cost, using an early-stopped Dijkstra's algorithm.

    The search terminates once k nodes (including the source itself) have been reached.

    Parameters
    ----------
    graph : networkx.Graph
        The input graph. Can be directed or undirected.
        Edge weights must be non-negative.
    source : node
        The starting node for paths.
    k : int
        Number of nodes to find distances to (including the source itself).

    Returns
    -------
    length : dict
        Dict keyed by node to shortest path length from source.
    """
    heap = [(0, source)]
    visited = set()
    length = {}

    weight = _weight_function(graph, weight)

    while heap and len(length) < k:
        dist_u, u = heapq.heappop(heap)
        if u in visited:
            continue

        visited.add(u)
        length[u] = dist_u

        for v, edata in graph[u].items():
            if v not in visited:
                w = weight(u, v, edata)
                heapq.heappush(heap, (dist_u + w, v))

    return length
