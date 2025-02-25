"""Linear algebra utils."""

import numpy as np


def _prefix_with_ellipsis(string):
    return f"...{string}"


def normalize(array, axis=-1):
    # TODO: handle norm zero?
    return array / np.linalg.norm(array, axis=axis, keepdims=True)


def columnwise_scaling(vec, mat):
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
    first_term = "n"
    second_term = "nk"
    rhs = "nk"

    if vec.ndim > 1:
        first_term = _prefix_with_ellipsis(first_term)
        rhs = _prefix_with_ellipsis(rhs)
    if mat.ndim > 2:
        second_term = _prefix_with_ellipsis(second_term)
        rhs = _prefix_with_ellipsis(rhs)

    return np.einsum(f"{first_term},{second_term}->{rhs}", vec, mat)


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
    return np.einsum("...,...i->...i", scalar, vec)


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
    if mat.ndim == 2 and vec.ndim == 2:
        return (mat @ vec.T).T

    return np.einsum("...ij,...j->...i", mat, vec)


def outer(vec_a, vec_b):
    """Outer product of two vectors.

    Parameters
    ----------
    vec_a : array-like, shape=[..., n]
        Vector.
    vec_b : array-like, shape=[..., m]
        Vector.

    Returns
    -------
    mat : array-like, shape=[..., n, m]
        Matrix.
    """
    if vec_a.ndim > 1 and vec_b.ndim > 1:
        return np.einsum("...i,...j->...ij", vec_a, vec_b)

    out = np.multiply.outer(vec_a, vec_b)
    if vec_b.ndim > 1:
        out = out.swapaxes(0, -2)

    return out
