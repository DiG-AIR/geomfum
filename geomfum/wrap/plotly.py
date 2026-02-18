"""Plotly plotting module."""

import plotly.graph_objects as go
from matplotlib import colors as mcolors

from geomfum.plot import ShapePlotter
from geomfum.shape.convert import to_go_mesh3d, to_go_pointcloud


class PlotlyShapePlotter(ShapePlotter):
    """Base plotting object for 3D shapes using Plotly."""

    def __init__(self, colormap="viridis"):
        self.colormap = colormap
        self._plotter = self.fig = go.Figure(
            data=[],
            layout=go.Layout(scene=dict(aspectmode="data")),
        )

    def highlight_vertices(self, coords, color="red", size=4):
        """Highlight vertices on shape.

        Parameters
        ----------
        coords : array-like, shape=[n_vertices, 3]
            Coordinates of vertices to highlight.
        color : str
            Color of the highlighted vertices.
        size : int
            Size of the highlighted vertices.
        """
        marker = go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(size=size, color=color),
            name="Highlighted_points",
        )
        self._plotter.add_trace(marker)
        return self

    def add_vectors(self, origins, vectors, color="blue", scale=1.0, name="vectors"):
        """Add vector field visualization.

        Parameters
        ----------
        origins : array-like, shape=[n_points, 3]
            Starting points for vectors.
        vectors : array-like, shape=[n_points, 3]
            Vector directions and magnitudes.
        color : str
            Color of the vectors.
        scale : float
            Scale factor for vector length.
        name : str
            Name for the vector trace.
        """
        # Create lines for each vector
        x_lines, y_lines, z_lines = [], [], []

        for i in range(len(origins)):
            start = origins[i]
            end = start + scale * vectors[i]

            # Add line from start to end
            x_lines.extend([start[0], end[0], None])
            y_lines.extend([start[1], end[1], None])
            z_lines.extend([start[2], end[2], None])

        vector_trace = go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode="lines",
            line=dict(color=color, width=3),
            name=name,
        )
        self._plotter.add_trace(vector_trace)
        return self

    def set_colormap(self, colormap):
        """Update the colormap.

        Parameters
        ----------
        colormap : str
            Name of the colormap to use.
        """
        self.colormap = colormap
        if len(self._plotter.data) > 0:
            self._plotter.data[0]["colorscale"] = colormap
        return self

    def show(self):
        """Display plot."""
        self._plotter.show()

    def save(self, filename, **kwargs):
        """Save plot to file.

        Parameters
        ----------
        filename : str
            Filename to save to.
        **kwargs
            Additional arguments passed to plotly's write_html or write_image.
        """
        if filename.endswith(".html"):
            self._plotter.write_html(filename, **kwargs)
        else:
            self._plotter.write_image(filename, **kwargs)
        return self


class PlotlyMeshPlotter(PlotlyShapePlotter):
    """Plotting object to display meshes."""

    def __init__(self, colormap="viridis"):
        super().__init__(colormap=colormap)

    def add_mesh(self, mesh, **kwargs):
        """Add mesh to plot.

        Parameters
        ----------
        mesh : TriangleMesh
            Mesh to be plotted.
        """
        plotly_obj = to_go_mesh3d(mesh)

        plotly_obj.update(colorscale=self.colormap, **kwargs)
        self._plotter.update(data=[plotly_obj])

        # Add hover text with vertex indices
        hover_text = [f"Index: {index}" for index in range(len(mesh.vertices))]
        self._plotter.data[0]["text"] = hover_text
        return self

    def set_vertex_scalars(self, scalars, name="scalars"):
        """Set vertex scalars on mesh."""
        data = self._plotter.data[0]
        data["intensity"] = scalars
        data["colorscale"] = self.colormap
        self._plotter.data[0].update(data)
        return self

    def set_vertex_colors(self, colors):
        """Set vertex colors on mesh."""
        data = self._plotter.data[0]
        data["vertexcolor"] = colors
        self._plotter.data[0].update(data)
        return self


class PlotlyPointCloudPlotter(PlotlyShapePlotter):
    """Plotting object to display point clouds."""

    def __init__(self, colormap="viridis"):
        super().__init__(colormap=colormap)

    def add_point_cloud(self, pointcloud, **kwargs):
        """Add point cloud to plot.

        Parameters
        ----------
        pointcloud : PointCloud
            Point cloud to be plotted.
        """
        plotly_obj = to_go_pointcloud(pointcloud)

        plotly_obj.update(marker=dict(colorscale=self.colormap), **kwargs)
        self._plotter.update(data=[plotly_obj])

        # Add hover text with vertex indices
        hover_text = [f"Index: {index}" for index in range(len(pointcloud.vertices))]
        self._plotter.data[0]["text"] = hover_text
        return self

    def set_vertex_scalars(self, scalars, name="scalars"):
        """Set vertex scalars on point cloud."""
        data = self._plotter.data[0]
        data["marker"]["color"] = scalars
        data["marker"]["colorscale"] = self.colormap
        self._plotter.data[0].update(data)
        return self

    def set_vertex_colors(self, colors):
        """Set vertex colors on point cloud."""
        data = self._plotter.data[0]
        colors_hex = [mcolors.rgb2hex(color) for color in colors]
        data["marker"]["color"] = colors_hex
        self._plotter.data[0].update(data)
        return self
