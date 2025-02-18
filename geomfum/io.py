import meshio
import torch


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
    return mesh.points, mesh.cells[0].data

def load_mesh_torch(filename):
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
    return torch.tensor(mesh.points),torch.tensor(mesh.cells[0].data)
