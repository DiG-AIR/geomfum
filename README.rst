GEOMFUM
=======
![alt text](GeomFuMbianco.png)

A `pyFM <https://pypi.org/project/pyfmaps/>`_-inspired package for geometric processing with `functional maps <https://dl.acm.org/doi/10.1145/2185520.2185526>`_.


Installation
------------

::

    pip install geomfum@git+https://github.com/luisfpereira/geomfum.git@main

Or the classic pipeline ``clone + pip install .``.


Installation issues may likely arise from one of the dependencies that relies on ``C++``
(in particular, `robust_laplacian <https://pypi.org/project/robust-laplacian/>`_).
Make sure you have installed everything they require.

For ``pyRMT`` follow the instructions `here <https://github.com/filthynobleman/rematching/tree/python-binding>`_.


How to use
----------

The `how-to notebooks <./notebooks/how_to>`_ are designed to safely let you dive in the package.
Why not starting from the `beginning <./notebooks/how_to/load_mesh_from_file.ipynb>`_ and simply follow the links that inspire you the most?


Have fun!
