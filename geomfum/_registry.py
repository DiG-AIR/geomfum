import abc
import inspect
import re
import sys

from geomfum._utils import has_package


class Registry(abc.ABC):
    @classmethod
    def register(cls, key, obj_name, requires=(), as_default=False):
        """Register.

        Parameters
        ----------
        key : str or tuple
            Key. First element must be wrap name.
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
    def get(cls, key):
        """Get register object.

        Parameters
        ----------
        key : str or tuple
            Key. First element must be wrap name.

        Returns
        -------
        Obj : class
            Registered object.
        """
        if key is None:
            key = cls.default
        obj_name, missing_package = cls.MAP[key]
        if missing_package:
            raise ModuleNotFoundError(missing_package)

        package_name = key if isinstance(key, str) else key[0]

        module = __import__(f"geomfum.wrap.{package_name}", fromlist=[""])
        Obj = getattr(module, obj_name)

        return Obj

    @classmethod
    def list_available(cls):
        """List register keys.

        Returns
        -------
        keys : list
            Registered keys.
        """
        return list(cls.MAP.keys())

    @classmethod
    def only_from_registry(cls):
        """Message for no internal implementation.

        Returns
        -------
        msg : str
            Message for no internal implementation with available
            implementations.
        """
        sign = str(inspect.signature(cls.get))[1:-1]
        return (
            f"No internal implementation. Use`.from_registry({sign}, **kwargs)`. "
            "Available implementations: "
            f"{', '.join([str(elem) for elem in cls.list_available()])}."
        )


class WhichRegistry(Registry, abc.ABC):
    @classmethod
    def register(cls, which, obj_name, requires=(), as_default=False):
        """Register.

        Parameters
        ----------
        which : str
            Key.
        obj_name : class name
            Name of the object to register.
        requires : str or tuple
            Required packages.
        as_default : bool
            Whether to set it as default.
        """
        return super().register(which, obj_name, requires, as_default)

    @classmethod
    def get(cls, which):
        """Get register object.

        Parameters
        ----------
        which : str
            Key.

        Returns
        -------
        Obj : class
            Registered object.
        """
        return super().get(which)


class MeshWhichRegistry(Registry, abc.ABC):
    @classmethod
    def register(cls, mesh, which, obj_name, requires=(), as_default=False):
        """Register.

        Parameters
        ----------
        mesh : bool
            Whether a mesh or point cloud.
        which : str
            Key.
        obj_name : class name
            Name of the object to register.
        requires : str or tuple
            Required packages.
        as_default : bool
            Whether to set it as default.
        """
        return super().register((which, mesh), obj_name, requires)

    @classmethod
    def get(cls, mesh, which):
        """Get register object.

        Parameters
        ----------
        mesh : bool
            Whether a mesh or point cloud.
        which : str
            Key.

        Returns
        -------
        Obj : class
            Registered object.
        """
        return super().get((which, mesh))


class WhichRegistryMixins:
    def __init__(self, *args, **kwargs):
        raise ValueError(self._Registry.only_from_registry())

        super().__init__(*args, **kwargs)

    @classmethod
    def from_registry(cls, *args, which=None, **kwargs):
        """Instantiate registered implementation.

        Parameters
        ----------
        which : str
            A registered implementation.

        Returns
        -------
        obj : BaseHeatKernelSignature
            Instantiated object.
        """
        return cls._Registry.get(which)(*args, **kwargs)


class MeshWhichRegistryMixins:
    def __init__(self, *args, **kwargs):
        raise ValueError(self._Registry.only_from_registry())

        super().__init__(*args, **kwargs)

    @classmethod
    def from_registry(cls, *args, mesh=True, which="robust", **kwargs):
        """Instantiate wrapped implementation.

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
        return cls._Registry.get(mesh, which)(*args, **kwargs)


class LaplacianFinderRegistry(MeshWhichRegistry):
    """Laplacian finder registry."""

    MAP = {}


class HeatKernelSignatureRegistry(WhichRegistry):
    MAP = {}


class WaveKernelSignatureRegistry(WhichRegistry):
    MAP = {}


class FaceValuedGradientRegistry(WhichRegistry):
    MAP = {}


class FaceDivergenceOperatorRegistry(WhichRegistry):
    MAP = {}


class FaceOrientationOperatorRegistry(WhichRegistry):
    MAP = {}


class HierarchicalMeshRegistry(WhichRegistry):
    MAP = {}


class PoissonSamplerRegistry(WhichRegistry):
    MAP = {}


def _create_register_funcs(module):
    """Create ``register`` functions for each class registry in this module.

    Given a ``Registry`` (e.g. ``LaplacianFinderRegistry``),
    it creates a function ``register_`` by removing ``Registry``
    from the name and transforming it in snake case
    (e.g. ``register_laplacian_finder``).

    These functions are widely used within ``geomstats.wrap``.
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
