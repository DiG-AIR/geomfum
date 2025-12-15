"""Optimization routines."""

import logging

import gsops.backend as gs
import scipy
from geomstats.numerics._common import result_to_backend_type


# TODO: homogenize with geomstats
class ScipyMinimize:
    """Backend-agnostic wrapper for SciPy's optimization routines.

    Parameters
    ----------
    method : str, optional
        Optimization algorithm (default: 'L-BFGS-B').
    bounds : sequence, optional
        Bounds on variables for constrained methods.
    constraints : dict or sequence of dict, optional
        Constraints definition.
    tol : float, optional
        Tolerance for termination.
    callback : callable, optional
        Function called after each iteration.
    options : dict, optional
        Solver-specific options.
    save_result : bool, optional
        Whether to save the optimization result (default: False).
    """

    def __init__(
        self,
        method="L-BFGS-B",
        bounds=None,
        constraints=(),
        tol=None,
        callback=None,
        options=None,
        save_result=False,
    ):
        self.method = method
        self.bounds = bounds
        self.constraints = constraints
        self.tol = tol
        self.callback = callback
        self.options = options

        self.save_result = save_result
        self.result_ = None

    def minimize(self, fun, x0, fun_jac=None, fun_hess=None, hessp=None):
        """Minimize objective function.

        Parameters
        ----------
        fun : callable
            The objective function to be minimized.
        x0 : array-like
            Initial guess.
        fun_jac : callable
            Jacobian of fun.
        fun_hess : callable
            Hessian of fun.
        hessp : callable
        """
        fun_ = lambda x: fun(gs.from_numpy(x))
        fun_jac_ = lambda x: fun_jac(gs.from_numpy(x))
        if x0.ndim > 1:
            # TODO: consider value_and_jac
            # TODO: consider hessian
            shape = x0.shape
            fun_ = lambda x: fun(gs.reshape(gs.from_numpy(x), shape))
            fun_jac_ = lambda x: fun_jac(gs.from_numpy(x).reshape(shape)).flatten()

            x0 = x0.flatten()

        result = scipy.optimize.minimize(
            fun_,
            x0,
            method=self.method,
            jac=fun_jac_,
            hess=fun_hess,
            hessp=hessp,
            bounds=self.bounds,
            tol=self.tol,
            constraints=self.constraints,
            callback=self.callback,
            options=self.options,
        )

        result = result_to_backend_type(result)

        if not result.success:
            logging.warning(result.message)

        if self.save_result:
            self.result_ = result

        return result
