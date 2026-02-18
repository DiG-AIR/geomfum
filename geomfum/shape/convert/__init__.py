"""Conversion utilities for shape data to external formats.

This module provides optional converters to PyVista and Plotly formats.
Import errors are silently handled for missing dependencies.
"""

try:
    from ._pyvista import to_pv_polydata  # noqa:F401
except ImportError:
    pass

try:
    from ._plotly import to_go_mesh3d, to_go_pointcloud  # noqa:F401
except ImportError:
    pass
