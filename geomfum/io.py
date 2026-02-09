"""Utilities for loading geometric data."""

import gsops.backend as gs
import meshio


def load_mesh(filename):
    """Load a mesh from a file.

    Parameters
    ----------
    filename : str
        File name.

    Returns
    -------
    vertices : array-like, shape=[n_vertices, 3]
    faces : array_like, shape=[n_faces, 3]
    """
    mesh = meshio.read(filename)
    return gs.from_numpy(mesh.points), gs.from_numpy(mesh.cells[0].data)


def load_tetrahedral_mesh(filename):
    """Load a tetrahedral mesh from a file.

    Parameters
    ----------
    filename : str
        File name (.mesh, .vtk, .vtu, etc.).

    Returns
    -------
    vertices : array-like, shape=[n_vertices, 3]
    tets : array-like, shape=[n_tets, 4]
    """
    mesh = meshio.read(filename)

    tets = None
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            tets = cell_block.data
            break

    if tets is None:
        raise ValueError(
            f"No tetrahedral cells found in '{filename}'. "
            f"Available cell types: {[c.type for c in mesh.cells]}"
        )

    return gs.array(mesh.points), gs.array(tets)


def load_pointcloud(filename):
    """Load a point cloud from a file.

    Parameters
    ----------
    filename : str
        File name.

    Returns
    -------
    vertices : array-like, shape=[n_vertices, 3]
    """
    point_cloud = meshio.read(filename)
    return gs.from_numpy(point_cloud.points)
