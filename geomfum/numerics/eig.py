import gsops.backend as gs
import scipy


class ScipyEigsh:
    def __init__(
        self,
        spectrum_size=6,
        sigma=None,
        which="LM",
    ):
        self.spectrum_size = spectrum_size
        self.sigma = sigma
        self.which = which

    def __call__(self, A, M=None):
        vals, vecs = scipy.sparse.linalg.eigsh(
            gs.sparse.to_scipy_csc(A),
            k=self.spectrum_size,
            M=gs.sparse.to_scipy_dia(M),
            sigma=self.sigma,
            which=self.which,
        )
        return gs.from_numpy(vals), gs.from_numpy(vecs)
