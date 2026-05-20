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
    register_neighbor_finder,
    register_poisson_sampler,
    register_precise_geodesic_distance_metric,
    register_wave_kernel_signature,
)
from geomfum._utils import has_package

register_laplacian_finder(
    "mesh",
    "pyfm",
    "PyfmMeshLaplacianFinder",
    requires="pyFM",
    as_default=not has_package("robust_laplacian"),
)

register_laplacian_finder(
    "mesh",
    "robust",
    "RobustMeshLaplacianFinder",
    requires="robust_laplacian",
    as_default=has_package("robust_laplacian"),
)

register_laplacian_finder("mesh", "igl", "IglMeshLaplacianFinder", requires="igl")


register_laplacian_finder(
    "pointcloud", "robust", "RobustPointCloudLaplacianFinder", requires="robust_laplacian"
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
    "mesh", "pp3d", "Pp3dMeshHeatDistanceMetric", requires="potpourri3d", as_default=True
)

register_heat_distance_metric(
    "pointcloud",
    "pp3d",
    "Pp3dPointSetHeatDistanceMetric",
    requires="potpourri3d",
    as_default=True,
)

register_precise_geodesic_distance_metric(
    "pp3d",
    "Pp3dEdgeFlipGeodesicMetric",
    requires="potpourri3d",
    as_default=True,
)
