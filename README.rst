.. image:: https://raw.githubusercontent.com/3diglab/geomfum/main/GeomFuMlogo.png
   :width: 800
   :alt: GeomFuM logo

**GeomFuM** is a Modular Python Package for Machine Learning with `Functional Maps <https://dl.acm.org/doi/10.1145/2185520.2185526>`_. 
Have a look at our `Software Paper Preprint <https://drive.google.com/file/d/1zr7ml2QWEOOlS9S3imER_HBuvYwMm3oo/view?usp=sharing>`_.

Installation
------------
We have a pipl package that you can install with the following command from your terminal
::
    
    pip install geomfum

Or directly from the GitHub repository
::
    
    pip install geomfum@git+https://github.com/3diglab/geomfum.git@main

Or

::
    
    pip install geomfum 
    pip install geomstats@git+https://github.com/geomstats/geomstats.git@main


Or the classic pipeline: ``clone + pip install``.

Make sure you have installed the most recent version of Geomstats to correctly handle the backend.

::
    pip install geomstats@git+https://github.com/geomstats/geomstats.git@main


⚠️ **ISSUES**



- Installation issues may arise from dependencies relying on C++ (particularly `robust_laplacian <https://pypi.org/project/robust-laplacian/>`_).

- Make sure all their requirements are installed.

Some functionality requires packages that are not published on PyPI and must be installed manually:

- `Rematching`: 

.. code-block:: bash

    pip install git+https://github.com/filthynobleman/rematching.git@python-binding


- `Polpo`: 

.. code-block:: bash

    pip install git+https://github.com/geometric-intelligence/polpo.git@main


How to use
----------

The `how-to notebooks <./notebooks/how_to>`_ are designed to safely let you dive in the package.

Why not starting from the `beginning <./notebooks/how_to/load_mesh_from_file.ipynb>`_ and simply follow the links that inspire you the most?

Choose the backend
------------------

GeomFuM can run seamlessly with ``numpy`` and ``pytorch``. 
By default, the ``numpy`` backend is used. The visualizations are only available with this backend.

The backend is based on the `Geomstats <https://github.com/geomstats/geomstats>`_ backend, which is installed automatically. The GeomFuM backend add functionality, especially regarding sparse matrices and device handling.

You can choose your backend by setting the environment variable
``GEOMSTATS_BACKEND`` to ``numpy``, or ``pytorch``, and
importing the backend module. From the command line:

::

    export GEOMSTATS_BACKEND=<backend_name>

and in the Python3 code:

::

    import gsops.backend as gs

Contributions
-------------

We welcome contributions from the community!  
If you have suggestions, bug reports, or want to improve the code or documentation, feel free to:

- Open an issue

- Submit a pull request

- Improve or add new examples/notebooks

Please follow our `contribution guidelines <https://3diglab.github.io/geomfum.github.io/contributing.html>`_ and adhere to best practices for clean, modular, and well-documented code.


Community
---------
Join our Discord Server! https://discord.gg/THHku2ckJs


List of Implemented Papers
--------------------------

1. `Functional Maps: A Flexible Representation of Maps Between Shapes <http://www.lix.polytechnique.fr/~maks/papers/obsbg_fmaps.pdf>`_
2. `Rematching: Low-resolution representations for scalable shape correspondence <https://arxiv.org/abs/2305.09274>`_
3. `ZoomOut: Spectral Upsampling for Efficient Shape Correspondence <https://arxiv.org/abs/1904.07865>`_
4. `Fast Sinkhorn Filters: Using Matrix Scaling for Non-Rigid Shape Correspondence with Functional Maps <https://openaccess.thecvf.com/content/CVPR2021/html/Pai_Fast_Sinkhorn_Filters_Using_Matrix_Scaling_for_Non-Rigid_Shape_Correspondence_CVPR_2021_paper.html>`_
5. `Structured regularization of functional map computations <https://www.lix.polytechnique.fr/Labo/Ovsjanikov.Maks/papers/resolvent_SGP19_small.pdf>`_
6. `Bijective upsampling and learned embedding for point clouds correspondences <https://www.sciencedirect.com/science/article/pii/S0097849324001201>`_
7. `Deep Geometric Functional Maps: Robust Feature Learning for Shape Correspondence <https://arxiv.org/abs/2003.14286>`_
8. `Laplace-Beltrami Eigenfunctions Towards an Algorithm That "Understands" Geometry <https://brunolevy.github.io/papers/Laplacian_SMI_2006.pdf>`_
9. `The Heat Method for Distance Computation <https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/>`_
10. `A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion <http://www.lix.polytechnique.fr/~maks/papers/hks.pdf>`_
11. `The Wave Kernel Signature: A Quantum Mechanical Approach To Shape Analysis <http://imagine.enpc.fr/~aubrym/projects/wks/index.html>`_
12. `Informative Descriptor Preservation via Commutativity for Shape Matching <https://www.lix.polytechnique.fr/Labo/Ovsjanikov.Maks/papers/fundescEG17.pdf>`_
13. `DiffusionNet: Discretization Agnostic Learning on Surfaces <https://arxiv.org/abs/2012.00888>`_
14. `PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation <https://arxiv.org/abs/1612.00593>`_

Acknowledgement
---------------
We thank the geometry processing and functional maps community for their foundational research and ongoing contributions that inspired this work, particularly open-source libraries on functional maps such as pyFM (RobinMagnet), FMNet (pvnieo), and Unsupervised-Learning-of-Robust-Spectral-Shape-Matching (dongliangcao), which provided valuable implementations and examples. We tried our best, referencing all relevant works in the library to give credit to researchers and developers; however, we acknowledge that we could have missed some! Please contact us and propose a change if you want recognition and think something is missing!

If you use Geomfum, please cite the `Software <https://doi.org/10.5281/zenodo.17194577>`_ to give recognition to any contributor of the project.

This work was partially supported by the European Union (Next Generation EU), MUR (REGAINS), NVIDIA Academic Hardware Grant, and the NSF (MRSEC and CAREER awards). 


**Have FuM!**
