"""Utility functions for visualizing meshes and point clouds in Jupyter notebooks using PyVista."""

import numpy as np
import pyvista as pv


def setup_pyvista(backend="static"):
    """Configure PyVista for inline rendering in notebooks."""
    pv.set_jupyter_backend(backend)


def pv_faces(faces):
    """Convert triangle faces (m, 3) to PyVista's flat face format."""
    faces = np.asarray(faces, dtype=np.int64)
    return np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()


def show_mesh(
    vertices,
    faces,
    scalars=None,
    scalar_name="value",
    cmap="viridis",
    smooth=True,
    show_edges=False,
):
    """Visualize a single mesh with optional scalar coloring."""
    mesh = pv.PolyData(np.asarray(vertices), pv_faces(faces))
    if scalars is not None:
        mesh.point_data[scalar_name] = np.asarray(scalars)

    plotter = pv.Plotter(notebook=True)
    plotter.add_mesh(
        mesh,
        scalars=scalar_name if scalars is not None else None,
        cmap=cmap,
        color=None if scalars is not None else "lightgray",
        smooth_shading=smooth,
        show_edges=show_edges,
    )
    plotter.add_axes()
    plotter.show(jupyter_backend="static")


def show_multiple_meshes(
    vertices_list,
    faces_list,
    scalar_dict_list=None,
    cmap="viridis",
    smooth=True,
    n_cols=3,
):
    """Visualize multiple meshes in a grid layout, each with optional scalar coloring."""
    n = len(vertices_list)
    n_rows = (n + n_cols - 1) // n_cols
    plotter = pv.Plotter(shape=(n_rows, n_cols), notebook=True, border=False)

    for idx in range(n):
        row, col = divmod(idx, n_cols)
        plotter.subplot(row, col)
        mesh = pv.PolyData(np.asarray(vertices_list[idx]), pv_faces(faces_list[idx]))
        if scalar_dict_list is not None:
            for name, scalars in scalar_dict_list[idx].items():
                mesh.point_data[name] = np.asarray(scalars)
            plotter.add_mesh(
                mesh,
                scalars=list(scalar_dict_list[idx].keys())[0],
                cmap=cmap,
                smooth_shading=smooth,
            )
        else:
            plotter.add_mesh(mesh, color="lightgray", smooth_shading=smooth)
        plotter.add_axes()

    plotter.link_views()
    plotter.show(jupyter_backend="static")


def show_point_cloud(
    points,
    scalars=None,
    scalar_name="value",
    cmap="viridis",
    point_size=8,
):
    """Visualize a point cloud with optional scalar coloring."""
    cloud = pv.PolyData(np.asarray(points))
    if scalars is not None:
        cloud.point_data[scalar_name] = np.asarray(scalars)

    plotter = pv.Plotter(notebook=True)
    plotter.add_mesh(
        cloud,
        scalars=scalar_name if scalars is not None else None,
        cmap=cmap,
        color=None if scalars is not None else "lightgray",
        point_size=point_size,
        render_points_as_spheres=True,
    )
    plotter.add_axes()
    plotter.show(jupyter_backend="static")


def show_mesh_with_highlights(
    vertices,
    faces,
    highlight_coords,
    highlight_color="red",
    highlight_size=12,
    mesh_color="lightgray",
    smooth=True,
):
    """Visualize a mesh with specific vertices highlighted as spheres."""
    mesh = pv.PolyData(np.asarray(vertices), pv_faces(faces))
    pts = pv.PolyData(np.asarray(highlight_coords))

    plotter = pv.Plotter(notebook=True)
    plotter.add_mesh(mesh, color=mesh_color, smooth_shading=smooth)
    plotter.add_mesh(
        pts,
        color=highlight_color,
        point_size=highlight_size,
        render_points_as_spheres=True,
    )
    plotter.add_axes()
    plotter.show(jupyter_backend="static")


def show_multiple_scalars(
    vertices,
    faces,
    scalar_dict,
    cmap="viridis",
    smooth=True,
    n_cols=3,
):
    """Visualize multiple scalar fields on the same mesh in a grid layout."""
    names = list(scalar_dict.keys())
    n = len(names)
    n_rows = (n + n_cols - 1) // n_cols
    plotter = pv.Plotter(shape=(n_rows, n_cols), notebook=True, border=False)

    for idx, name in enumerate(names):
        row, col = divmod(idx, n_cols)
        plotter.subplot(row, col)
        mesh = pv.PolyData(np.asarray(vertices), pv_faces(faces))
        mesh.point_data[name] = np.asarray(scalar_dict[name])
        plotter.add_mesh(mesh, scalars=name, cmap=cmap, smooth_shading=smooth)
        plotter.add_text(name, font_size=10)
        plotter.add_axes()

    plotter.link_views()
    plotter.show(jupyter_backend="static")


def show_mesh_with_vectors(
    vertices,
    faces,
    scalar=None,
    scalar_name="value",
    cmap="viridis",
    vectors=None,
    vector_points=None,
    vector_name="vectors",
    vector_scale=0.03,
    vector_color="orangered",
    smooth=True,
):
    """Visualize a mesh with optional scalar coloring and overlaid vector field."""
    mesh = pv.PolyData(np.asarray(vertices), pv_faces(faces))
    if scalar is not None:
        mesh.point_data[scalar_name] = np.asarray(scalar)

    plotter = pv.Plotter(notebook=True)
    plotter.add_mesh(
        mesh,
        scalars=scalar_name if scalar is not None else None,
        cmap=cmap,
        color=None if scalar is not None else "lightgray",
        smooth_shading=smooth,
    )

    if vectors is not None:
        points = np.asarray(vertices if vector_points is None else vector_points)
        vector_field = pv.PolyData(points)
        vector_field[vector_name] = np.asarray(vectors)
        glyph = vector_field.glyph(orient=vector_name, scale=False, factor=vector_scale)
        plotter.add_mesh(glyph, color=vector_color)

    plotter.add_axes()
    plotter.show(jupyter_backend="static")
