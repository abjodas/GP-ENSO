from __future__ import annotations

import numpy as np
import pymc as pm

def build_quasiperiodic_gp_model(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[pm.Model, pm.gp.Marginal]:
    """
    Multi-input GP model matching the notebook:

      X: shape (N, 2) where columns are [t_years, soi] (scaled)
      y: shape (N,) scaled target (nino34)

    Kernel:
      cov_ard = ExpQuad(2, ls=l) with ARD lengthscales (shape=2)
      cov_per = Periodic(2, period, ls=l_per, active_dims=[0])  # periodic in time only
      cov_total = n^2 * (cov_per + cov_ard)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"Expected X to have shape (N, 2) [t_years, soi], got {X.shape}")

    with pm.Model() as model:
        # ARD lengthscales for both dims
        l = pm.Gamma("l", alpha=2, beta=1, shape=2)
        cov_ard = pm.gp.cov.ExpQuad(2, ls=l)

        # Periodic component (only depends on time dim 0)
        period = pm.Gamma("period", alpha=8, beta=2, initval=3.5)
        l_per = pm.Gamma("l_per", alpha=2, beta=1)
        cov_per = pm.gp.cov.Periodic(2, period=period, ls=l_per, active_dims=[0])

        # Overall amplitude
        n = pm.HalfCauchy("n", beta=2)
        cov_total = n**2 * (cov_per + cov_ard)

        sigma = pm.HalfNormal("sigma", sigma=0.5)

        gp = pm.gp.Marginal(cov_func=cov_total)
        gp.marginal_likelihood("y", X=X, y=y, sigma=sigma)

    return model, gp


def fit_map(model: pm.Model) -> dict:
    """
    Find MAP estimate
    """
    with model:
        mp = pm.find_MAP(include_transformed=True)

    return mp