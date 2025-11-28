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
