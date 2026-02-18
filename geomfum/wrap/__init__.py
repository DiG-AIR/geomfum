"""Registration of various wrapped components in GeomFum."""

from geomfum._registry import (
    register_face_divergence_operator,
    register_face_orientation_operator,
    register_face_valued_gradient,
    register_feature_extractor,
    register_heat_distance_metric,
    register_heat_kernel_signature,
    register_hierarchical_mesh,
    register_landmark_heat_kernel_signature,
    register_landmark_wave_kernel_signature,
    register_laplacian_finder,
    register_mesh_plotter,
    register_neighbor_finder,
    register_point_cloud_plotter,
    register_poisson_sampler,
    register_wave_kernel_signature,
)
from geomfum._utils import has_package

register_laplacian_finder(
    True,
    "pyfm",
    "PyfmMeshLaplacianFinder",
    requires="pyFM",
    as_default=not has_package("robust_laplacian"),
)

register_laplacian_finder(
    True,
    "robust",
    "RobustMeshLaplacianFinder",
    requires="robust_laplacian",
    as_default=has_package("robust_laplacian"),
)

register_laplacian_finder(True, "igl", "IglMeshLaplacianFinder", requires="igl")


register_laplacian_finder(
    False, "robust", "RobustPointCloudLaplacianFinder", requires="robust_laplacian"
)

register_heat_kernel_signature(
    "pyfm", "PyfmHeatKernelSignature", requires="pyFM", as_default=True
)

register_landmark_heat_kernel_signature(
    "pyfm", "PyfmLandmarkHeatKernelSignature", requires="pyFM", as_default=True
)

register_landmark_wave_kernel_signature(
    "pyfm", "PyfmLandmarkWaveKernelSignature", requires="pyFM", as_default=True
)

register_wave_kernel_signature(
    "pyfm", "PyfmWaveKernelSignature", requires="pyFM", as_default=True
)

register_face_valued_gradient(
    "pyfm", "PyfmFaceValuedGradient", requires="pyFM", as_default=True
)

register_face_divergence_operator(
    "pyfm", "PyfmFaceDivergenceOperator", requires="pyFM", as_default=True
)

register_face_orientation_operator(
    "pyfm", "PyFmFaceOrientationOperator", requires="pyFM", as_default=True
)

register_hierarchical_mesh(
    "pyrmt", "PyrmtHierarchicalMesh", requires="PyRMT", as_default=True
)

register_poisson_sampler(
    "pymeshlab", "PymeshlabPoissonSampler", requires="pymeshlab", as_default=True
)

register_mesh_plotter("plotly", "PlotlyMeshPlotter", requires="plotly", as_default=True)

register_mesh_plotter("pyvista", "PvMeshPlotter", requires="pyvista", as_default=False)

register_mesh_plotter(
    "polyscope", "PsMeshPlotter", requires="polyscope", as_default=False
)

register_point_cloud_plotter(
    "plotly", "PlotlyPointCloudPlotter", requires="plotly", as_default=True
)

register_point_cloud_plotter(
    "pyvista", "PvPointCloudPlotter", requires="pyvista", as_default=False
)

register_point_cloud_plotter(
    "polyscope", "PsPointCloudPlotter", requires="polyscope", as_default=False
)


register_feature_extractor(
    "pointnet", "PointnetFeatureExtractor", requires="torch", as_default=False
)

register_feature_extractor(
    "diffusionnet", "DiffusionnetFeatureExtractor", requires="torch", as_default=True
)

register_feature_extractor(
    "transformer",
    "TransformerFeatureExtractor",
    requires="torch",
    as_default=False,
)

register_neighbor_finder(
    "pot", "PotSinkhornNeighborFinder", requires="ot", as_default=True
)


register_heat_distance_metric(
    True, "pp3d", "Pp3dMeshHeatDistanceMetric", requires="potpourri3d", as_default=True
)

register_heat_distance_metric(
    False,
    "pp3d",
    "Pp3dPointSetHeatDistanceMetric",
    requires="potpourri3d",
    as_default=True,
)
