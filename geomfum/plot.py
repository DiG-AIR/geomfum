"""Plotting functions.

In this file we define the plotting logic.
Since each plotting library has its own method of plotting,
we define general functions that works with any library implemented
"""

import abc

from geomfum._registry import (
    MeshPlotterRegistry,
    PointCloudPlotterRegistry,
    WhichRegistryMixins,
)


class ShapePlotter(abc.ABC):
    """Plotting object.

    Primitive clas to plot meshes, pointclouds or specific useful informations
    (scalar functions, landmarks, etc..)
    """

    @abc.abstractmethod
    def show(self):
        """Display plot."""

    def set_vertex_scalars(self, scalars):
        """Set vertex scalars on mesh."""
        raise NotImplementedError("Not implemented for this plotter.")

    def highlight_vertices(self, coords, color, size):
        """Highlight vertices on mesh."""
        raise NotImplementedError("Not implemented for this plotter.")

    def set_vertex_colors(self, colors):
        """Set vertex colors on mesh."""
        raise NotImplementedError("Not implemented for this plotter.")


class MeshPlotter(WhichRegistryMixins, ShapePlotter):
    """Plotting object to display meshes."""

    _Registry = MeshPlotterRegistry


class PointCloudPlotter(WhichRegistryMixins, ShapePlotter):
    """Plotting object to display point clouds."""

    _Registry = PointCloudPlotterRegistry
