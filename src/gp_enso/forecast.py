from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import pymc as pm

@dataclass(frozen=True)
class GPPredictionX:
    """GP prediction for arbitrary design matrix X_new."""
    X_new: np.ndarray           # (M,D)
    mu: np.ndarray              # (M,)
    cov: np.ndarray             # (M,M)

def draw_paths(
    mu: np.ndarray,
    cov: np.ndarray,
    draws: int = 5,
    seed: int = -1,
) -> np.ndarray:
    """
    Draw sample functions from N(mu, cov).
    Returns shape (draws, M).
    """
    rv = pm.MvNormal.dist(mu=mu, cov=cov)
    return pm.draw(rv, draws=draws, random_seed=seed)

def predict_gp_X(
    model: pm.Model,
    gp: pm.gp.Marginal,
    mp: dict,
    X_new: np.ndarray,
) -> GPPredictionX:
    """
    Predict GP mean/cov at an arbitrary design matrix.

    Typical usage for the multi-input notebook:
      pred = predict_gp_X(model, gp, mp, X_test_scaled)
    """
    X_new = np.asarray(X_new, dtype=float)
    if X_new.ndim != 2:
        raise ValueError(f"X_new must be 2D (M,D). Got shape {X_new.shape}.")

    with model:
        mu, cov = gp.predict(X_new, point=mp)

    mu = np.asarray(mu, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    return GPPredictionX(X_new=X_new, mu=mu, cov=cov)