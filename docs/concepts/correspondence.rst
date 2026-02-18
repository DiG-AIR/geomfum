Shape Correspondence
====================

Shape correspondence is the problem of finding meaningful point-to-point mappings between geometric shapes. This is a fundamental task in geometry processing, computer graphics, and shape analysis.

Pointwise Correspondences
-------------------------

Given two shapes :math:`\mathcal{X}_1` and :math:`\mathcal{X}_2`, modeled as compact two-dimensional manifolds embedded in :math:`\mathbb{R}^3`, a pointwise correspondence is a map:

.. math::

    T_{12}: \mathcal{X}_1 \rightarrow \mathcal{X}_2

where :math:`y = T_{12}(x)` is the corresponding point of :math:`x` on shape :math:`\mathcal{X}_2`.

Discrete Representation
-----------------------

In practice, shapes are discretized as point clouds or meshes with :math:`n_1` and :math:`n_2` points, respectively. The correspondence can be represented as a (soft or hard) permutation matrix :math:`\Pi_{12} \in \mathbb{R}^{n_1 \times n_2}` such that:

- :math:`\Pi_{12}(i, j) = 1` if :math:`T_{12}(x_i) = y_j`, and :math:`0` otherwise.
- If the correspondence is bijective, :math:`\Pi_{12}` is a permutation matrix.

This allows us to write, for example, the relationship between vertex matrices :math:`V_1` and :math:`V_2` as:

.. math::

    V_1 = \Pi_{12} V_2

where :math:`V_1 \in \mathbb{R}^{n_1 \times 3}` and :math:`V_2 \in \mathbb{R}^{n_2 \times 3}` are the matrices of 3D coordinates.

.. figure:: ../_static/ShapeMatching.png
    :width: 110%
    :align: center

    **Figure:** Example of pointwise correspondence between two meshes.


Functional Representation
-------------------------

A pointwise correspondence induces a linear operator between function spaces (see :doc:`functional_maps`):

.. math::

    T^F_{21}: \mathcal{L}^2(\mathcal{X}_2) \rightarrow \mathcal{L}^2(\mathcal{X}_1), \quad T^F_{21}(g) = g \circ T_{12}

This operator, known as the functional map, enables efficient and robust computation of correspondences by working in the space of functions rather than directly with points.

For more details on functional maps and their properties, see the :doc:`functional_maps` section.

.. figure:: ../_static/ShapeMatchingDense.png
    :width: 110%
    :align: center

    **Figure:** Example of dense shape correspondence between two meshes. Points with the same color indicate corresponding locations.

