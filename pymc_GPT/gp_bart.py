import warnings
from multiprocessing import Manager
from typing import Optional

import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt
from pandas import DataFrame, Series
from pymc.distributions.distribution import Distribution, _support_point
from pymc.logprob.abstract import _logprob
from pytensor.tensor.random.op import RandomVariable

# from .split_rules import SplitRule
from .tree import Tree
from .utils import TensorLike, _sample_posterior

__all__ = ["GP_BART"]

class GP_BARTRV(RandomVariable):
    """Base class for GP_BART."""

    name: str = "GP_BART"
    signature = "(m,n),(m),(),(),() -> (m)"
    dtype: str = "floatX"
    _print_name: tuple[str, str] = ("GP_BART")
    all_trees = list[list[list[Tree]]]

    def _supp_shape_from_params(self, dist_params, rep_param_idx=1, param_shapes=None):
        idx = dist_params[0].ndim - 2
        return [dist_params[0].shape[idx]]
    
    @classmethod
    def rng_fn(
        cls, rng=None, X=None, Y=None, m=None, alpha=None, beta=None, size=None
    ):
        if not size:
            size = None

        if not cls.all_trees:
            if size is not None:
                return np.full((size[0], cls.Y.shape[0]), cls.Y.mean())
            else:
                return np.full(cls.Y.shape[0], cls.Y.mean())
        else:
            if size is not None:
                shape = size[0]
            else:
                shape = 1
            return _sample_posterior(cls.all_trees, cls.X, rng=rng, shape=shape).squeeze().T


gp_bart = GP_BARTRV()


class GP_BART(Distribution):
    r"""
    Gaussian Processes Bayesian Additive Regression Tree distribution 
    
    Distribution representing sum over trees with Gaussian prior

    X : PyTensor Variable, Pandas/Polars DataFrame or Numpy array
        The covariate matrix.
    Y : PyTensor Variable, Pandas/Polar DataFrame/Series,or Numpy array
        The response vector.
    m : int
        Number of trees.
    alpha : float
        Controls the prior probability over the depth of the trees.
        Should be in the (0, 1) interval.
    beta : float
        Controls the prior probability over the number of leaves of the trees.
        Should be positive.

    Notes
    -----
    The parameters ``alpha`` and ``beta`` parametrize the probability that a node at
    depth :math:`d \: (= 0, 1, 2,...)` is non-terminal, given by :math:`\alpha(1 + d)^{-\beta}`.
    The default values are :math:`\alpha = 0.95` and :math:`\beta = 2`.

    This is the recommend prior by Chipman Et al. BART: Bayesian additive regression trees,
    `link <https://doi.org/10.1214/09-AOAS285>`__
    """

    def __new__(
        cls,
        name: str,
        X: TensorLike,
        Y: TensorLike,
        m: int = 50,
        alpha: float = 0.95,
        beta: float = 2.0,
        **kwargs,
    ):
        manager = Manager()
        cls.all_trees = manager.list()

        # Preprocess X and Y
        X, Y = preprocess_xy(X, Y)

        gp_bart_op = type(
            f"GP_BART_{name}",
            (GP_BARTRV,),
            {
                "name": "GP_BART",
                "all_trees": cls.all_trees,
                "inplace": False,
                "initval": Y.mean(),
                "X": X,
                "Y": Y,
                "m": m,
                "alpha": alpha,
                "beta": beta,
            },
        )

        Distribution.register(GP_BARTRV)

        @_support_point.register(GP_BARTRV)
        def get_moment(rv, size, *rv_inputs):
            return cls.get_moment(rv, size, *rv_inputs)
        
        cls.rv_op = gp_bart_op
        params = [X, Y, m, alpha, beta]
        return super().__new__(cls, name, *params, **kwargs)
    

    @classmethod
    def dist(cls, *params, **kwargs):
        return super().dist(params, **kwargs)
    
    def logp(self, x, *inputs):
        """Calculate log probability.

        Parameters
        ----------
        x: numeric, TensorVariable
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        return pt.zeros_like(x)

    @classmethod
    def get_moment(cls, rv, size, *rv_inputs):
        mean = pt.fill(size, rv.Y.mean())
        return mean
    

def preprocess_xy(
    X: TensorLike, Y: TensorLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if isinstance(Y, (Series, DataFrame)):
        Y = Y.to_numpy()
    if isinstance(X, (Series, DataFrame)):
        X = X.to_numpy()

    try:
        import polars as pl

        if isinstance(X, (pl.Series, pl.DataFrame)):
            X = X.to_numpy()
        if isinstance(Y, (pl.Series, pl.DataFrame)):
            Y = Y.to_numpy()
    except ImportError:
        pass

    Y = Y.astype(float)
    X = X.astype(float)

    return X, Y


@_logprob.register(GP_BARTRV)
def logp(op, value_var, *dist_params, **kwargs):
    _dist_params = dist_params[3:]
    value_var = value_var[0]
    return GP_BART.logp(value_var, *_dist_params)  # pylint: disable=no-value-for-parameter
