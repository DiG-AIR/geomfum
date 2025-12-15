"""Conversion functions to PyVista objects for visualization."""

import gsops.backend as gs
import pyvista as pv


def to_pv_polydata(mesh):
    """Convert a TriangleMesh object to a PyVista PolyData object."""
    return pv.PolyData.from_regular_faces(
        points=gs.to_numpy(mesh.vertices), faces=gs.to_numpy(mesh.faces)
    )
