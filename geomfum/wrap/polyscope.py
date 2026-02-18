"""Wraps polyscope functions."""

import gsops.backend as gs
import polyscope as ps

from geomfum.plot import ShapePlotter


class PsMeshPlotter(ShapePlotter):
    """Plotting object to display meshes."""

    # NB: for now assumes only one mesh is plotted

    def __init__(self, colormap="viridis", backend=""):
        super().__init__()

        self.colormap = colormap

        self._plotter = ps
        self._name = "Mymesh"

        self._plotter.init(backend)

    def add_mesh(self, mesh):
        """Add mesh to plot.

        Parameters
        ----------
        mesh : TriangleMesh
            Mesh to be plotted.
        """
        self._plotter.register_surface_mesh(
            self._name, gs.to_numpy(mesh.vertices), gs.to_numpy(mesh.faces)
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
        ps.get_surface_mesh(self._name).add_scalar_quantity(
            name,
            scalars,
            defined_on="vertices",
            cmap=self.colormap,
            enabled=True,
        )
        return self

    def set_vertex_colors(self, colors, name="colors"):
        """Set vertex colors on mesh.

        Parameters
        ----------
        colors : array-like, shape=[n_vertices, 3] or [n_vertices, 4]
            RGB or RGBA color values for each vertex (values in range [0, 1]).
        name : str
            Name for the color quantity.
        """
        # Polyscope expects RGB values in [0, 1]
        if colors.max() > 1.0:
            colors = colors / 255.0

        # If RGBA, use only RGB channels
        if colors.shape[1] == 4:
            colors = colors[:, :3]
        ps.get_surface_mesh(self._name).add_color_quantity(
            name,
            colors,
            defined_on="vertices",
            enabled=True,
        )
        return self

    def highlight_vertices(
        self,
        coords,
        color=(1.0, 0.0, 0.0),
        size=0.01,
    ):
        """
        Highlight vertices on a mesh using Polyscope by adding a point cloud.

        Parameters
        ----------
        coords : array-like, shape = [n_vertices, 3]
            Coordinates of vertices to highlight.
        color : tuple
            Color of the highlighted vertices (e.g., (1.0, 0.0, 0.0)).
        radius : float
            Radius of the rendered points (visual size).
        """
        name = "Highlighted_points"
        self._plotter.register_point_cloud(name, coords, radius=size, color=color)
        return self

    def show(self):
        """Display plot."""
        self._plotter.show()


class PsPointCloudPlotter(ShapePlotter):
    """Plotting object to display point clouds."""

    # NB: for now assumes only one point cloud is plotted

    def __init__(self, colormap="viridis", backend=""):
        super().__init__()

        self.colormap = colormap

        self._plotter = ps
        self._name = "MyPointCloud"

        self._plotter.init(backend)

    def add_point_cloud(self, pointcloud):
        """Add point cloud to plot.

        Parameters
        ----------
        pointcloud : PointCloud
            Point cloud to be plotted.
        """
        self._plotter.register_point_cloud(self._name, gs.to_numpy(pointcloud.vertices))
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
        ps.get_point_cloud(self._name).add_scalar_quantity(
            name,
            scalars,
            enabled=True,
            cmap=self.colormap,
        )
        return self

    def set_vertex_colors(self, colors, name="colors"):
        """Set vertex colors on point cloud.

        Parameters
        ----------
        colors : array-like, shape=[n_vertices, 3] or [n_vertices, 4]
            RGB or RGBA color values for each vertex (values in range [0, 1]).
        name : str
            Name for the color quantity.
        """
        import numpy as np

        colors_array = np.asarray(colors)

        # Polyscope expects RGB values in [0, 1]
        if colors_array.max() > 1.0:
            colors_array = colors_array / 255.0

        # If RGBA, use only RGB channels
        if colors_array.shape[1] == 4:
            colors_array = colors_array[:, :3]

        ps.get_point_cloud(self._name).add_color_quantity(
            name,
            colors_array,
            enabled=True,
        )
        return self

    def highlight_vertices(
        self,
        coords,
        color=(1.0, 0.0, 0.0),
        size=0.01,
    ):
        """
        Highlight specific points in the point cloud.

        Parameters
        ----------
        coords : array-like, shape = [n_vertices, 3]
            Coordinates of vertices to highlight.
        color : tuple
            Color of the highlighted vertices (e.g., (1.0, 0.0, 0.0)).
        size : float
            Radius of the rendered points (visual size).
        """
        name = "Highlighted_points"
        self._plotter.register_point_cloud(name, coords, radius=size, color=color)
        return self

    def show(self):
        """Display plot."""
        self._plotter.show()
