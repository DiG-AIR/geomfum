"""Wraps pyvista functions."""

import pyvista as pv

from geomfum.plot import ShapePlotter
from geomfum.shape.convert import to_pv_polydata


class PvMeshPlotter(ShapePlotter):
    """Plotting object to display meshes."""

    # NB: for now assumes only one mesh is plotted

    def __init__(self, colormap="viridis", **kwargs):
        self.colormap = colormap

        self._plotter = pv.Plotter(**kwargs)
        self._mesh = None
        self._add_mesh = None

    def __getattr__(self, name):
        """Get attribute.

        It is only called when ``__getattribute__`` fails.
        Delegates attribute calling to plotter.
        """
        return getattr(self._plotter, name)

    def add_mesh(self, mesh, **kwargs):
        """Add mesh to plot.

        Parameters
        ----------
        mesh : TriangleMesh
            Mesh to be plotted.
        """
        self._mesh = to_pv_polydata(mesh)

        self._add_mesh = lambda mesh: self._plotter.add_mesh(
            mesh, cmap=self.colormap, **kwargs
        )

        return self

    def set_vertex_scalars(self, scalars, name="scalars"):
        """Set vertex scalars on mesh.

        Parameters
        ----------
        scalars : array-like
            Value at each vertex.
        name : str
            Scalar field name.
        """
        self._mesh.point_data.set_scalars(scalars, name=name)

        return self

    def highlight_vertices(self, coords, color="red", size=0.01):
        """
        Highlight vertices on the mesh using PyVista.

        Parameters
        ----------
        coords : array-like, shape = [n_vertices, 3]
            Coordinates of vertices to highlight.
        color : str or tuple
            Color of the highlighted vertices.
        size : float
            Size of the highlighted vertices (radius of spheres).
        """
        name = "Highlighted_points"
        points = pv.PolyData(coords)
        glyphs = points.glyph(scale=False, geom=pv.Sphere(radius=size))
        self._plotter.add_mesh(glyphs, color=color, name=name)
        return self

    def show(self):
        """Display plot."""
        self._add_mesh(self._mesh)
        self._plotter.show()


class PvPointCloudPlotter(ShapePlotter):
    """Plotting object to display point clouds."""

    # NB: for now assumes only one point cloud is plotted

    def __init__(self, colormap="viridis", **kwargs):
        self.colormap = colormap

        self._plotter = pv.Plotter(**kwargs)
        self._point_cloud = None
        self._add_point_cloud = None

    def __getattr__(self, name):
        """Get attribute.

        It is only called when ``__getattribute__`` fails.
        Delegates attribute calling to plotter.
        """
        return getattr(self._plotter, name)

    def add_point_cloud(self, pointcloud, **kwargs):
        """Add point cloud to plot.

        Parameters
        ----------
        pointcloud : PointCloud
            Point cloud to be plotted.
        """
        # Convert point cloud to PyVista PolyData
        vertices = pointcloud.vertices
        self._point_cloud = pv.PolyData(vertices)

        self._add_point_cloud = lambda pc: self._plotter.add_mesh(
            pc,
            cmap=self.colormap,
            point_size=5,
            render_points_as_spheres=True,
            **kwargs,
        )

        return self

    def set_vertex_scalars(self, scalars, name="scalars"):
        """Set vertex scalars on point cloud.

        Parameters
        ----------
        scalars : array-like
            Value at each vertex.
        name : str
            Scalar field name.
        """
        self._point_cloud.point_data.set_scalars(scalars, name=name)

        return self

    def highlight_vertices(self, coords, color="red", size=0.01):
        """
        Highlight specific points in the point cloud.

        Parameters
        ----------
        coords : array-like, shape = [n_vertices, 3]
            Coordinates of vertices to highlight.
        color : str or tuple
            Color of the highlighted vertices.
        size : float
            Size of the highlighted vertices (radius of spheres).
        """
        name = "Highlighted_points"
        points = pv.PolyData(coords)
        glyphs = points.glyph(scale=False, geom=pv.Sphere(radius=size))
        self._plotter.add_mesh(glyphs, color=color, name=name)
        return self

    def show(self):
        """Display plot."""
        self._add_point_cloud(self._point_cloud)
        self._plotter.show()
