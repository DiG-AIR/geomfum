import pymeshlab

from geomfum.sample import BaseSampler


class PymeshlabPoissonSampler(BaseSampler):
    """Poisson disk sampling.

    Parameters
    ----------
    n_samples : int
        Number of samples to target.
    """

    def __init__(self, n_samples):
        self.n_samples = n_samples

    def sample(self, shape):
        """Sample using Poisson disk sampling.

        Parameters
        ----------
        shape : TriangleMesh
            Mesh.

        Returns
        -------
        samples : array-like, shape=[n_samples, 3]
            Coordinates of samples.
        """
        ms = pymeshlab.MeshSet()
        ms.add_mesh(
            pymeshlab.Mesh(vertex_matrix=shape.vertices, face_matrix=shape.faces)
        )

        ms.generate_sampling_poisson_disk(samplenum=self.n_samples)

        return ms.current_mesh().vertex_matrix()
