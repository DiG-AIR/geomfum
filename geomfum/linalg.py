"""Linear algebra utils."""

import gsops.backend as gs


def _prefix_with_ellipsis(string):
    return f"...{string}"


def normalize(array, axis=-1):
    """Normalize array along axis.

    Parameters
    ----------
    array : array-like, shape=[..., n, ...]
        Array to normalize.
    axis : int
        Axis to use for normalization.

    Returns
    -------
    array : array-like, shape=[..., n, ...]
        Normalized array.
    """
    # TODO: handle norm zero?
    return array / gs.linalg.norm(array, axis=axis, keepdims=True)


def scale_to_unit_sum(array, axis=-1):
    """Scale array to sum one along axis.

    Parameters
    ----------
    array : array-like, shape=[..., n, ...]
        Array to normalize.
    axis : int
        Axis to use for normalization.

    Returns
    -------
    array : array-like, shape=[..., n, ...]
        Scaled array.
    """
    return array / gs.sum(array, axis=axis, keepdims=True)


def _axiswise_scaling(vec, mat, axis=0):
    """Axis-wise scaling.

    Generalizaation of column- and row-wise scalings.

    Parameters
    ----------
    vec : array-like, shape=[..., {n, k}]
        Vector of scalings.
    mat :array-like, shape=[..., n, k]
        Matrix.
    axis : int
        Axis to use for normalization.

    Returns
    -------
    scaled_mat : array-like, shape=[..., n, k]
    """
    rhs = second_term = "nk"
    first_term = second_term[axis]

    if vec.ndim > 1:
        first_term = _prefix_with_ellipsis(first_term)
        rhs = _prefix_with_ellipsis(rhs)
    if mat.ndim > 2:
        second_term = _prefix_with_ellipsis(second_term)
        rhs = _prefix_with_ellipsis(rhs)

    return gs.einsum(f"{first_term},{second_term}->{rhs}", vec, mat)


def columnwise_scaling(vec, mat):
    """Columnwise scaling.

    Parameters
    ----------
    vec : array-like, shape=[..., k]
        Vector of scalings.
    mat :array-like, shape=[..., n, k]
        Matrix.

    Returns
    -------
    scaled_mat : array-like, shape=[..., n, k]
    """
    return _axiswise_scaling(vec, mat, axis=1)


def rowwise_scaling(vec, mat):
    """Columnwise scaling.

    Parameters
    ----------
    vec : array-like, shape=[..., n]
        Vector of scalings.
    mat :array-like, shape=[..., n, k]
        Matrix.

    Returns
    -------
    scaled_mat : array-like, shape=[..., n, k]
    """
    return _axiswise_scaling(vec, mat, axis=0)


def scalarvecmul(scalar, vec):
    """Scalar vector multiplication.

    Parameters
    ----------
    scalar : array-like, shape=[....]
        Scalar.
    vec : array-like, shape=[..., n]
        Vector.

    Returns
    -------
    scaled_vec : array-like, shape=[..., n]
        Scaled vector.
    """
    return gs.einsum("...,...i->...i", scalar, vec)


def matvecmul(mat, vec):
    """Matrix vector multiplication.

    Parameters
    ----------
    mat : array-like, shape=[..., m, n]
        Matrix.
    vec : array-like, shape=[..., n]
        Vector.

    Returns
    -------
    matvec : array-like, shape=[..., m]
        Matrix vector multiplication.
    """
    if vec.ndim == 1:
        return mat @ vec

    if mat.ndim == 2:
        reshape_out = False
        if vec.ndim > 2:  # to handle sparse matrices
            reshape_out = True
            batch_shape = vec.shape[:-1]
            vec = vec.reshape(-1, vec.shape[-1])

        out = (mat @ vec.T).T
        if reshape_out:
            return out.reshape(batch_shape + mat.shape[:1])

        return out

    return gs.einsum("...ij,...j->...i", mat, vec)
