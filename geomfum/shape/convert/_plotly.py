"""Conversion functions to Plotly objects for visualization."""

import plotly.graph_objects as go


def to_go_mesh3d(mesh):
    """Convert a TriangleMesh object to a plotly Mesh3d object."""
    x, y, z = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
    f1, f2, f3 = mesh.faces[:, 0], mesh.faces[:, 1], mesh.faces[:, 2]

    return go.Mesh3d(x=x, y=y, z=z, i=f1, j=f2, k=f3)


def to_go_pointcloud(mesh):
    """Convert a PointCloud object to a plotly Scatter3d object."""
    return go.Scatter3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        mode="markers",
        marker=dict(size=2, color="blue"),
    )
