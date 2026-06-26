Usage
=====

This page provides comprehensive usage examples for GeomFuM, referencing our interactive notebook tutorials.

Quick Start
-----------

The best way to learn GeomFuM is through our interactive notebook tutorials.

.. toctree::
   :maxdepth: 1
   :caption: Start with:

   notebooks/how_to/00_load_mesh_from_file
   notebooks/how_to/01_mesh_laplacian
   notebooks/how_to/07_functional_map

Basic Usage Examples
--------------------

Here are some key code snippets from our tutorials:

Loading and Working with Meshes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   notebooks/how_to/00_load_mesh_from_file

.. code-block:: python

    from geomfum.dataset import NotebooksDataset
    from geomfum.shape import TriangleMesh
    
    # Load a mesh
    dataset = NotebooksDataset()
    mesh = TriangleMesh.from_file(dataset.get_filename("cat-00"))
    
    print(f"Mesh: {mesh.n_vertices} vertices, {mesh.n_faces} faces")

Computing Laplace-Beltrami Spectrum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   notebooks/how_to/02_mesh_laplacian_spectrum

.. code-block:: python

    # Compute eigenfunctions and eigenvalues
    mesh.laplacian.find_spectrum(spectrum_size=50, set_as_basis=True)
    
    # Access the basis
    eigenfunctions = mesh.basis.vecs
    eigenvalues = mesh.basis.vals
    
    print(f"Computed {len(eigenvalues)} eigenfunctions")

Computing Shape Descriptors
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   notebooks/how_to/03_descriptors

.. code-block:: python

    from geomfum.descriptor.spectral import HeatKernelSignature, WaveKernelSignature
    
    # Heat Kernel Signature
    hks = HeatKernelSignature(n_domain=4)
    hks_descriptors = hks(mesh)
    
    # Wave Kernel Signature
    wks = WaveKernelSignature(n_domain=3)
    wks_descriptors = wks(mesh)

Creating Descriptor Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   notebooks/how_to/04_descriptor_pipeline

.. code-block:: python
    
    from geomfum.descriptor.spectral import HeatKernelSignature, LandmarkHeatKernelSignature
    from geomfum.descriptor.pipeline import DescriptorPipeline, ArangeSubsampler, L2InnerNormalizer
    
    steps = [
        HeatKernelSignature(n_domain=4),
        LandmarkHeatKernelSignature(n_domain=4),
        ArangeSubsampler(subsample_step=2),
        WaveKernelSignature(n_domain=3),
        L2InnerNormalizer(),
    ]
    
    pipeline = DescriptorPipeline(steps)
    descriptors = pipeline.apply(mesh)

Computing Functional Maps
-------------------------
.. toctree::
   :maxdepth: 1

   notebooks/how_to/07_functional_map

.. code-block:: python

    from geomfum.functional_map import FactorSum, LBCommutativityEnforcing, SpectralDescriptorPreservation
    from geomfum.numerics.optimization import ScipyMinimize
    
    # Set up basis for both meshes
    mesh_a.laplacian.find_spectrum(spectrum_size=10, set_as_basis=True)
    mesh_b.laplacian.find_spectrum(spectrum_size=10, set_as_basis=True)
    
    # Compute descriptors
    descr_a = pipeline.apply(mesh_a)
    descr_b = pipeline.apply(mesh_b)
    
    # Create objective function
    factors = [
        SpectralDescriptorPreservation(
            mesh_a.basis.project(descr_a),
            mesh_b.basis.project(descr_b),
            weight=1.0,
        ),
        LBCommutativityEnforcing.from_bases(
            mesh_a.basis, mesh_b.basis, weight=1e-2,
        ),
    ]
    
    objective = FactorSum(factors)
    
    # Optimize
    optimizer = ScipyMinimize(method="L-BFGS-B")
    x0 = gs.zeros((mesh_b.basis.spectrum_size, mesh_a.basis.spectrum_size))
    res = optimizer.minimize(objective, x0, fun_jac=objective.gradient)
    fmap = res.x.reshape(x0.shape)

Converting to Pointwise Correspondences
---------------------------------------
.. toctree::
   :maxdepth: 1

   notebooks/how_to/10_pointwise_from_functional

.. code-block:: python

    from geomfum.convert import P2pFromFmConverter

    converter = P2pFromFmConverter()
    correspondences = converter(fmap, mesh_a, mesh_b)
    
    print(f"Computed {len(correspondences)} correspondences")

Refining Functional Maps
------------------------
.. toctree::
   :maxdepth: 1

   notebooks/how_to/15_refine_functional_map

.. code-block:: python

    from geomfum.refine import ZoomOutRefiner
    
    # Apply ZoomOut refinement
    zoomout = ZoomOutRefiner()
    refined_fmap = zoomout(fmap, mesh_a, mesh_b)
    
    # Convert refined map to correspondences
    refined_correspondences = converter(refined_fmap, mesh_a, mesh_b)

Advanced Techniques
-------------------


Rematching Algorithm
~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   notebooks/how_to/13_rematching

.. code-block:: python

    from geomfum.shape.hierarchical import HierarchicalMesh    
    # Apply Rematching
    hmesh_a = HierarchicalMesh.from_registry(mesh_a, min_n_samples=1000)

    hmesh_a.low.n_vertices, hmesh_a.low.n_faces


Deep Functional Maps
~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   notebooks/how_to/14_deep_functional_maps_models

.. code-block:: python

    from geomfum.descriptor.learned import FeatureExtractor
    from geomfum.forward_functional_map import ForwardFunctionalMap
    from geomfum.learning.models import FMNet

    functional_map_model = FMNet(
    feature_extractor=FeatureExtractor.from_registry(which="diffusionnet"))

    with torch.no_grad():
        output = functional_map_model(mesh_a, mesh_b, as_dict=True)
    fmap12, fmap21 = output["fmap12"], output["fmap21"]

Neural Adjoint Maps
~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   notebooks/how_to/18_neural_adjoint_maps


.. code-block:: python

    from geomfum.convert import NamFromP2pConverter

    mesh_a.basis.use_k = 10
    mesh_b.basis.use_k = 10
    nam_converter = NamFromP2pConverter(device="cpu")

    nam = nam_converter(p2p, mesh_a.basis, mesh_b.basis)

Visualization
-------------

Basic Visualization
~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   notebooks/how_to/16_vis_basic


.. code-block:: python

    from geomfum.plot import MeshPlotter

    plotter = MeshPlotter()
    plotter.add_mesh(mesh_a)
    plotter.show()


Interactive Learning
--------------------

For the best learning experience:

1. **Follow the notebook tutorials** in order
2. **Run each cell** to see the outputs
3. **Modify parameters** to experiment
4. **Try your own data** by changing file paths
5. **Use the links** at the end of each notebook to navigate

All examples in this page are taken from our actual working notebooks. For complete, runnable examples with outputs and visualizations, see the :doc:`notebooks/index` section.

Next Steps
----------

Now that you understand the basic usage patterns:

1. **Explore the notebooks** - Follow the interactive tutorials
2. **Read the concepts** - Understand the theoretical foundations
3. **Check the API** - Reference the complete API documentation
4. **Join the community** - Visit our Discord server

For questions and issues, visit our `GitHub repository <https://github.com/3diglab/geomfum>`_.