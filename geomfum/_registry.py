import abc
import inspect
import re
import sys

from geomfum._utils import has_package


class Registry(abc.ABC):
    """Abstract base class for managing registered implementations of components."""

    # whether geomfum provides an implementation
    has_internal = False

    @classmethod
    def register(cls, key, obj_name, requires=(), as_default=False):
        """Register an implementation with optional package dependencies.

        Parameters
        ----------
        key : str
            Key.
        obj_name : class name
            Name of the object to register.
        requires : str or tuple
            Required packages.
        as_default : bool
            Whether to set it as default.
        """
        if isinstance(requires, str):
            requires = [requires]

        for package_name in requires:
            if not has_package(package_name):
                missing_package = package_name
                break
        else:
            missing_package = None

        if as_default:
            cls.default = key

        cls.MAP[key] = (obj_name, missing_package)

    @classmethod
    def get(cls, key=None):
        """Retrieve registered implementation by key.

        Parameters
        ----------
        key : str
            Key.

        Returns
        -------
        Obj : class
            Registered object.
        """
        if key is None:
            key = cls.default

        if key == "geomfum":
            if cls.has_internal:
                return None

            cls.raise_if_no_internal()

        obj_name, missing_package = cls.MAP[key]
        if missing_package:
            raise ModuleNotFoundError(missing_package)

        module = __import__(f"geomfum.wrap.{key}", fromlist=[""])
        Obj = getattr(module, obj_name)

        return Obj

    @classmethod
    def list_available(cls):
        """List all registered implementation keys.

        Returns
        -------
        keys : list
            Registered keys.
        """
        return list(cls.MAP.keys())

    @classmethod
    def only_from_registry(cls):
        """Generate error message listing available implementations.

        Returns
        -------
        msg : str
            Message for no internal implementation with available
            implementations.
        """
        sign = str(inspect.signature(cls.get))[1:-1]
        return (
            f"No internal implementation. Use `.from_registry({sign}, **kwargs)`. "
            "Available implementations: "
            f"{', '.join([str(elem) for elem in cls.list_available()])}."
        )

    @classmethod
    def raise_if_no_internal(cls):
        """Raise ValueError if no internal implementation exists."""
        if not cls.has_internal:
            raise ValueError(cls.only_from_registry())


class NestedRegistry(abc.ABC):
    """Two-level registry for managing implementations with nested categorization."""

    @classmethod
    def _outer_registry(cls, key=None):
        """Retrieve outer registry by key.

        Parameters
        ----------
        key_out : Hashable
            Key for outer register dict.
            Defaults to first key if ``None``.

        Returns
        -------
        registry : Registry
        """
        if key is None:
            return cls.Registries[list(cls.Registries.keys())[0]]
        return cls.Registries[key]

    @classmethod
    def register(cls, key_out, key_in, obj_name, requires=(), as_default=False):
        """Register implementation in nested registry structure.

        Parameters
        ----------
        key_out : Hashable
            Key for outer register dict.
        key_in : str
            Key for object in inner register.
        obj_name : class name
            Name of the object to register.
        requires : str or tuple
            Required packages.
        as_default : bool
            Whether to set it as default.
        """
        return cls._outer_registry(key_out).register(
            key_in, obj_name, requires, as_default
        )

    @classmethod
    def get(cls, key_out, key_in):
        """Retrieve registered implementation using nested keys.

        Parameters
        ----------
        key_out : Hashable
            Key for outer register dict.
        key_in : str
            Key for object in inner register.

        Returns
        -------
        Obj : class
            Registered object.
        """
        return cls._outer_registry(key_out).get(key_in)

    @classmethod
    def list_available(cls, key_out=None):
        """List available implementation keys for specified or all outer categories.

        Returns
        -------
        keys : list or dict[list]
            Registered keys.
        """
        if key_out is not None:
            return cls._outer_registry(key_out).list_available()

        available = {}
        for key, Registry in cls.Registries.items():
            available[key] = Registry.list_available()

        return available

    @classmethod
    def only_from_registry(cls, key_out=None):
        """Generate error message for missing internal implementation.

        Parameters
        ----------
        key_out : Hashable
            Key for outer register dict.
            If ``None``, defaults to default outer key.

        Returns
        -------
        msg : str
            Message for no internal implementation with available
            implementations.
        """
        return cls._outer_registry(key_out).only_from_registry()

    @classmethod
    def raise_if_no_internal(cls, key_out=None):
        """Raise ValueError if no internal implementation exists.

        Parameters
        ----------
        key_out : Hashable
            Key for outer register dict.
            If ``None``, defaults to default outer key.
        """
        return cls._outer_registry(key_out).only_from_registry()


class WhichRegistryMixins:
    """Mixin enabling registry-based instantiation via 'which' parameter."""

    def __init__(self, *args, **kwargs):
        self._Registry.raise_if_no_internal()

        super().__init__(*args, **kwargs)

    @classmethod
    def from_registry(cls, *args, which=None, **kwargs):
        """Create instance from registered implementation.

        Parameters
        ----------
        which : str
            A registered implementation.

        Returns
        -------
        obj : BaseHeatKernelSignature
            Instantiated object.
        """
        instantiator = cls._Registry.get(which)
        if instantiator is None:
            obj = cls.__new__(cls)
            obj.__init__(*args, **kwargs)
            return obj

        return instantiator(*args, **kwargs)


class MeshWhichRegistryMixins:
    """Mixin for registry-based instantiation with mesh/point cloud distinction."""

    def __init__(self, *args, **kwargs):
        # TODO: has to be improved
        self._Registry.raise_if_no_internal()

        super().__init__(*args, **kwargs)

    @classmethod
    def from_registry(cls, *args, mesh=True, which=None, **kwargs):
        """Create instance from registered implementation based on shape type.

        Parameters
        ----------
        mesh : bool
            Whether a mesh or point cloud.
        which : str
            A registered implementation.

        Returns
        -------
        obj : Obj
            An instantiated object.
        """
        instantiator = cls._Registry.get(mesh, which)
        if instantiator is None:
            obj = cls.__new__(cls)
            obj.__init__(*args, **kwargs)
            return obj

        return instantiator(*args, **kwargs)


class _MeshLaplacianFinderRegistry(Registry):
    has_internal = True
    MAP = {}


class _PointSetLaplacianFinderRegistry(Registry):
    has_internal = False
    MAP = {}


class LaplacianFinderRegistry(NestedRegistry):
    Registries = {
        True: _MeshLaplacianFinderRegistry,
        False: _PointSetLaplacianFinderRegistry,
    }


class HeatKernelSignatureRegistry(Registry):
    has_internal = True
    MAP = {}


class LandmarkHeatKernelSignatureRegistry(Registry):
    has_internal = True
    MAP = {}


class WaveKernelSignatureRegistry(Registry):
    has_internal = True
    MAP = {}


class LandmarkWaveKernelSignatureRegistry(Registry):
    has_internal = True
    MAP = {}


class FaceValuedGradientRegistry(Registry):
    MAP = {}


class FaceDivergenceOperatorRegistry(Registry):
    MAP = {}


class FaceOrientationOperatorRegistry(Registry):
    MAP = {}


class HierarchicalMeshRegistry(Registry):
    MAP = {}


class PoissonSamplerRegistry(Registry):
    MAP = {}


class FarthestPointSamplerRegistry(Registry):
    MAP = {}


class FeatureExtractorRegistry(Registry):
    MAP = {}


class NeighborFinderRegistry(Registry):
    has_internal = True

    MAP = {}


class MeshPlotterRegistry(Registry):
    MAP = {}


class PointCloudPlotterRegistry(Registry):
    MAP = {}


class _MeshHeatDistanceMetricRegistry(Registry):
    has_internal = False
    MAP = {}


class _PointSetHeatDistanceMetricRegistry(Registry):
    has_internal = False
    MAP = {}


class HeatDistanceMetricRegistry(NestedRegistry):
    Registries = {
        True: _MeshHeatDistanceMetricRegistry,
        False: _PointSetHeatDistanceMetricRegistry,
    }


def _create_register_funcs(module):
    """Create ``register`` functions for each class registry in this module.

    Given a ``Registry`` (e.g. ``LaplacianFinderRegistry``),
    it creates a function ``register_`` by removing ``Registry``
    from the name and transforming it in snake case
    (e.g. ``register_laplacian_finder``).

    These functions are widely used within ``geomfum.wrap``.
    """
    for name, method in inspect.getmembers(module):
        if not (
            hasattr(method, "__bases__")
            and abc.ABC not in method.__bases__
            and name.endswith("Registry")
        ):
            continue

        # upper case split
        name_ls = ["register"] + [
            word.lower() for word in re.findall("[A-Z][^A-Z]*", name)[:-1]
        ]
        new_name = "_".join(name_ls)

        setattr(module, new_name, method.register)


_create_register_funcs(sys.modules[__name__])
