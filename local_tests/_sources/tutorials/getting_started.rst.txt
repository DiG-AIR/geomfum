Getting Started with GeomFuM
============================

Welcome to GeomFuM! This section will guide you through the basics using our comprehensive collection of Jupyter notebooks.

Quick Start
-----------

The best way to get started with GeomFuM is to follow our step-by-step notebook tutorials. These notebooks contain real, working examples that you can run and modify.

Essential Tutorials
-------------------

Start with these fundamental notebooks in order:

1. **Loading Meshes** - :doc:`../notebooks/how_to/00_load_mesh_from_file`
   - Learn how to load mesh files and point clouds
   - Understand the basic data structures

2. **Laplace-Beltrami Operator** - :doc:`../notebooks/how_to/01_mesh_laplacian`
   - Compute the Laplace-Beltrami matrix
   - Understand the fundamental operator for shape analysis

3. **Laplace-Beltrami Spectrum** - :doc:`../notebooks/how_to/02_mesh_laplacian_spectrum`
   - Compute eigenfunctions and eigenvalues
   - Set up the spectral basis for functional maps

4. **Shape Descriptors** - :doc:`../notebooks/how_to/03_descriptors`
   - Compute Heat Kernel Signatures (HKS)
   - Compute Wave Kernel Signatures (WKS)
   - Understand shape descriptors

5. **Descriptor Pipeline** - :doc:`../notebooks/how_to/04_descriptor_pipeline`
   - Create descriptor computation pipelines
   - Combine multiple descriptors efficiently

Functional Maps Tutorials
-------------------------

Once you understand the basics, dive into functional maps:

6. **Functional Maps** - :doc:`../notebooks/how_to/07_functional_map`
   - Compute functional maps between shapes
   - Understand the optimization process

7. **Pointwise Correspondences** - :doc:`../notebooks/how_to/10_pointwise_from_functional`
   - Convert functional maps to pointwise correspondences
   - Visualize correspondences

8. **Refinement** - :doc:`../notebooks/how_to/15_refine_functional_map`
   - Apply ZoomOut and other refinement techniques
   - Improve correspondence quality

Advanced Topics
---------------

For more advanced usage:

9. **Landmarks** - :doc:`../notebooks/how_to/06_landmarks`
   - Use landmark correspondences
   - Constrain functional maps with landmarks

10. **Hierarchical Meshes** - :doc:`../notebooks/how_to/11_hierarchical_mesh`
    - Work with multi-resolution meshes
    - Multi-scale functional maps

11. **Rematching** - :doc:`../notebooks/how_to/13_rematching`
    - Use the Rematching algorithm
    - Low-resolution shape correspondence

Deep Learning
-------------

For deep learning approaches:

12. **Deep Functional Maps** - :doc:`../notebooks/how_to/14_deep_functional_maps_models`
    - Use neural networks for feature extraction
    - Learn-based functional maps

13. **Neural Adjoint Maps** - :doc:`../notebooks/how_to/18_neural_adjoint_maps`
    - End-to-end correspondence learning
    - Neural adjoint map computation

Visualization
-------------

Learn to visualize your results:

14. **Basic Visualization** - :doc:`../notebooks/how_to/16_vis_basic`
    - Basic mesh and correspondence visualization
    - Plotting tools and techniques

15. **Distance Visualization** - :doc:`../notebooks/how_to/17_vis_dist`
    - Visualize geodesic distances
    - Error analysis and visualization

Running the Notebooks
---------------------

To run these tutorials:

1. **Install GeomFuM with all dependencies**:
   .. code-block:: bash

       pip install geomfum[opt,test-scripts,plotting-all]@git+https://github.com/3diglab/geomfum.git@main

2. **Start Jupyter**:
   .. code-block:: bash

       jupyter lab

3. **Navigate to the notebooks directory** and start with `00_load_mesh_from_file.ipynb`

4. **Follow the links** at the end of each notebook to progress through the tutorials

Interactive Learning
--------------------

Each notebook is designed to be interactive:

- **Run cells sequentially** to understand the concepts
- **Modify parameters** to see how they affect results
- **Experiment with different shapes** from the dataset
- **Try your own meshes** by changing the file paths

Dataset
-------

The tutorials use the GeomFuM dataset which includes:

- **Animal shapes**: cats, lions, horses, etc.
- **Human shapes**: various poses and subjects
- **Synthetic shapes**: for testing and validation

To use your own data, simply replace the dataset paths with your mesh files.

Next Steps
----------

After completing the tutorials:

1. **Explore the demos** - Check out the `demos/` directory for more complex examples
2. **Read the concepts** - Understand the theoretical foundations in :doc:`../concepts/index`
3. **Check the API** - Reference the complete API in :doc:`../api/index`
4. **Join the community** - Visit our `Discord server <https://discord.gg/6sYmEbUp>`_

For questions and issues, visit our `GitHub repository <https://github.com/3diglab/geomfum>`_. 