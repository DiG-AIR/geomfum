Functional Maps
===============

Background
----------

We model a 3D shape :math:`\mathcal{X}_1` as a compact two-dimensional manifold embedded in :math:`\mathbb{R}^3`. The space of square-integrable real-valued functions on the surface, :math:`\mathcal{L}^2(\mathcal{X}_1)`, is defined as:

.. math::

    \mathcal{L}^2(\mathcal{X}_1) := \{ f : \mathcal{X}_1 \rightarrow \mathbb{R} \mid \int_{\mathcal{X}_1} |f(x)|^2 dx < \infty \}

This is a Hilbert space with inner product:

.. math::

    \langle f, g \rangle_{\mathcal{L}^2(\mathcal{X}_1)} = \int_{\mathcal{X}_1} f(x) g(x) dx

In practice, shapes are discretized as point clouds or meshes, with :math:`n_1` points :math:`\{x_i\}_{i=1}^{n_1}`. Functions are represented as vectors :math:`f \in \mathbb{R}^{n_1}` with entries :math:`f_i = f(x_i)`. The mass matrix :math:`M_1 \in \mathbb{R}^{n_1 \times n_1}` (diagonal, with entries :math:`m_i`) allows discretization of the inner product:

.. math::

    \langle f, g \rangle_{\mathcal{L}^2(\mathcal{X}_1)} \approx f^\top M_1 g

The Laplace-Beltrami operator :math:`\Delta_1` is discretized as a matrix. Its eigendecomposition yields eigenvalues :math:`\{\lambda_1^i\}` and eigenfunctions :math:`\{\phi_1^i\}` forming an orthonormal basis (LBO basis). For efficiency, we use a truncated basis :math:`\Phi_1^k = [\phi_1^1, \ldots, \phi_1^k] \in \mathbb{R}^{n_1 \times k}`.

.. figure:: ../_static/LaplacianBase.png
    :width: 110%
    :align: center

    **Figure:** Illustration of four Laplacian eigenfunctions on a mesh.

Functional Maps
---------------

Given two shapes :math:`\mathcal{X}_1` and :math:`\mathcal{X}_2`, a pointwise correspondence :math:`T_{12}` induces a pull-back operator (the functional map):

.. math::

    T^F_{21} : \mathcal{L}^2(\mathcal{X}_2) \rightarrow \mathcal{L}^2(\mathcal{X}_1), \quad T^F_{21}(g) = g \circ T_{12}

In a chosen basis, this operator is represented as a matrix :math:`C_{21}` mapping coefficients. If :math:`\Pi_{12}` is the pointwise correspondence matrix, then:

.. math::

    C_{21} = \Phi_1^\dagger \Pi_{12} \Phi_2

where :math:`\dagger` denotes the Moore–Penrose pseudoinverse.

.. figure:: ../_static/FunctionalMaps.png
    :width: 110%
    :align: center

    **Figure:** Example of a Functional Map calculation, highlighting the correspondences between the Laplacian eigenfunctions.


Pointwise Map Recovery
----------------------

To recover pointwise correspondences from functional maps, we use the nearest search in the embedding space.

.. math::

    T_{12} = \mathrm{NS}(\Phi_1, \Phi_2 C_{21}^\top)

Here, :math:`\mathrm{NS}` denotes nearest search in the embedding space.


.. figure:: ../_static/p2p.png
    :width: 110%
    :align: center

    **Figure:** Example of resulting pointwise correspondences.

Truncated Basis and Approximations
----------------------------------

Using a truncated basis (:math:`k \ll n_1, n_2`) enables efficient computation but introduces approximations in delta function representation and pointwise recovery. The row :math:`\Phi_1^k(x)` is the spectral embedding of :math:`x` in dimension :math:`k`. Linear operators cannot perfectly align these embeddings without additional priors.


Key Papers
----------

1. `Functional Maps: A Flexible Representation of Maps Between Shapes <http://www.lix.polytechnique.fr/~maks/papers/obsbg_fmaps.pdf>`_
2. `ZoomOut: Spectral Upsampling for Efficient Shape Correspondence <https://arxiv.org/abs/1904.07865>`_
3. `Deep Geometric Functional Maps: Robust Feature Learning for Shape Correspondence <https://arxiv.org/abs/2003.14286>`_
4. `Fast Sinkhorn Filters: Using Matrix Scaling for Non-Rigid Shape Correspondence <https://openaccess.thecvf.com/content/CVPR2021/html/Pai_Fast_Sinkhorn_Filters_Using_Matrix_Scaling_for_Non-Rigid_Shape_Correspondence_CVPR_2021_paper.html>`_
5. `Elastic Functional Maps <https://arxiv.org/abs/2307.12913>`_

For more detailed examples and tutorials, see the :doc:`tutorials <../tutorials/index>` section. 