Installation
============

This page provides detailed installation instructions for GeomFuM.

Quick Installation
------------------

Install GeomFuM directly from GitHub:

.. code-block:: bash

    pip install geomfum@git+https://github.com/3diglab/geomfum.git@main

For development installation with all dependencies:

.. code-block:: bash

    pip install geomfum[opt]@git+https://github.com/3diglab/geomfum.git@main

Prerequisites
-------------

System Requirements
~~~~~~~~~~~~~~~~~~~

- **Python**: 3.9 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: At least 4GB RAM (8GB+ recommended for large meshes)

Required Dependencies
~~~~~~~~~~~~~~~~~~~~~

Core dependencies (installed automatically):

- ``numpy`` - Numerical computing
- ``scipy`` - Scientific computing
- ``scikit-learn`` - Machine learning utilities
- ``meshio`` - Mesh I/O
- ``pyfmaps`` - Functional maps in python
- ``torch`` - Deep learning (PyTorch backend)
- ``geomstats`` - Geometric Riemannian statistics

Optional Dependencies
---------------------

Install specific optional dependencies based on your needs:

Laplacian Computation
~~~~~~~~~~~~~~~~~~~~~

For robust Laplacian computation:

.. code-block:: bash

    pip install geomfum[lapl]

This installs:

- ``robust-laplacian`` - Robust Laplacian computation
- ``libigl`` - Geometric processing library

Metric Computation
~~~~~~~~~~~~~~~~~~

For advanced metric computations:

.. code-block:: bash

    pip install geomfum[metric]

This installs:

- ``networkx`` - Graph algorithms
- ``potpourri3d`` - 3D geometry processing

Sampling Tools
~~~~~~~~~~~~~~

For mesh sampling utilities:

.. code-block:: bash

    pip install geomfum[sampling]

This installs:

- ``pymeshlab`` - MeshLab Python bindings

Rematching Algorithm
~~~~~~~~~~~~~~~~~~~~

For the Rematching algorithm:

.. code-block:: bash

    pip install geomfum[rematching]

This installs:

- ``Rematching`` - Low-resolution shape correspondence

Sinkhorn Filtering
~~~~~~~~~~~~~~~~~~

For optimal transport-based refinement:

.. code-block:: bash

    pip install geomfum[sinkhorn]

This installs:

- ``POT`` - Python Optimal Transport

Visualization
~~~~~~~~~~~~~

For visualization capabilities:

.. code-block:: bash

    # Plotly-based visualization
    pip install geomfum[plotly]
    
    # PyVista-based visualization
    pip install geomfum[pyvista]
    
    # Polyscope-based visualization
    pip install geomfum[polyscope]
    
    # All visualization backends
    pip install geomfum[plotting-all]


Testing and Development
~~~~~~~~~~~~~~~~~~~~~~~

For development and testing:

.. code-block:: bash

    pip install geomfum[test]

This installs:

- ``pytest`` - Testing framework
- ``polpo`` - Geometric processing utilities
- All optional dependencies
- Testing scripts and notebooks

Complete Installation
~~~~~~~~~~~~~~~~~~~~~

Install everything:

.. code-block:: bash

    pip install geomfum[opt,test-scripts,plotting-all]@git+https://github.com/3diglab/geomfum.git@main

Backend Configuration
---------------------

GeomFuM supports multiple backends for numerical computations:

NumPy Backend (Default)
~~~~~~~~~~~~~~~~~~~~~~~

The NumPy backend is used by default and provides good performance for most use cases:

.. code-block:: python

    import os
    os.environ["GEOMSTATS_BACKEND"] = "numpy"
    import geomfum as gfm

PyTorch Backend
~~~~~~~~~~~~~~~

Use PyTorch backend for GPU acceleration and deep learning:

.. code-block:: python

    import os
    os.environ["GEOMSTATS_BACKEND"] = "pytorch"
    import geomfum as gfm

Check Current Backend
~~~~~~~~~~~~~~~~~~~~~

Verify which backend is currently active:

.. code-block:: python

    import geomstats.backend as gs
    print(f"Current backend: {gs.__name__}")

Troubleshooting
---------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

C++ Dependencies
^^^^^^^^^^^^^^^^

Some dependencies require C++ compilation. If you encounter issues:

1. **Install build tools**:

   - **Linux**:

     .. code-block:: bash

         sudo apt-get install build-essential
   - **macOS**:

     .. code-block:: bash

         Install Xcode Command Line Tools
   - **Windows**:

       Install ``Visual Studio Build Tools``

2. **Install Eigen (if required)**:

   - **Linux**:

     .. code-block:: bash

         sudo apt-get install libeigen3-dev
   - **macOS**:

     .. code-block:: bash

         brew install eigen

PyRMT Installation
^^^^^^^^^^^^^^^^^^

For the Rematching algorithm, follow the specific instructions:

.. code-block:: bash

    # Clone and install PyRMT
    git clone https://github.com/filthynobleman/rematching.git
    cd rematching
    git checkout python-binding
    pip install -e .


Next Steps
----------

.. toctree::
   :maxdepth: 1

   ../tutorials/getting_started
   notebooks/demo/First-FunctionalMap-Demo
   ../api/index




For issues and questions, visit our `GitHub repository <https://github.com/3diglab/geomfum>`_ or join our `Discord server <https://discord.gg/6sYmEbUp>`_.