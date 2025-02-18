import abc


class Descriptor(abc.ABC):
    pass


class SpectralDescriptor(Descriptor, abc.ABC):
    """Spectral descriptor.

    Parameters
    ----------
    n_domain : int
        Number of domain points. Ignored if ``domain`` is not a callable.
    domain : callable or array-like, shape=[n_domain]
        Method to compute domain points (``f(basis, n_domain)``) or
        domain points.
    use_landmarks : bool
        Whether to use landmarks.
    """

    # TODO: make general implementation

    def __init__(self, n_domain, domain, use_landmarks=False):
        self.n_domain = n_domain
        self.domain = domain
        self.use_landmarks = use_landmarks

    @abc.abstractmethod
    def __call__(self, basis, domain=None):
        """Compute descriptor.

        Parameters
        ----------
        basis : Eigenbasis.
            Basis.
        domain : array-like, shape=[n_domain]
            Domain points for computation.
        """

class LearnedDescriptor(Descriptor, abc.ABC):
    """Descriptor representing the output of a feature extractor."""

    def __init__(self, n_features=128):
        self.n_features = n_features
        self.features = None
        self.trained = False

    @abc.abstractmethod
    def __call__(self, mesh):
        """Compute descriptor.

        Parameters
        ----------
        basis : mesh (or data).
            Basis.
        """
    
    @abc.abstractmethod
    def load(self, path):
        """load learned parameters.

        Parameters
        ----------
        path : pathfile.
            Basis.
        """

