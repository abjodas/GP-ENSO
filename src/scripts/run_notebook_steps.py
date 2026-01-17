from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
PLOT_DIR = REPO_ROOT / "plots"
sys.path.insert(0, str(SRC_DIR))

from gp_enso import *

set_plot_dir(PLOT_DIR)



def main():

    df_new = build_df(indices=("nino3", "soi", "nino4", "nino34", "nino12"))
    data = prepare_train_test(df_new, target="nino34", features=("t_years", "soi"))

    model, gp = build_quasiperiodic_gp_model(
        data.X_train_scaled,      # shape (700, 2): [t_years, soi]
        data.y_train_scaled       # shape (700,)
    )
    mp = fit_map(model)

    pred = predict_gp_X(model, gp, mp, data.X_test_scaled)

    samples_scaled = draw_paths(pred.mu, pred.cov, draws=10, seed=1)

    samples_rescaled = data.scaler_y.inverse_transform(samples_scaled.T).T  # (10, M)
    mean_pred = samples_rescaled.mean(axis=0)
    std_pred = samples_rescaled.std(axis=0)

    split_at = len(data.df) - len(data.X_test)  # == 600 if you used default prepare_train_test
    t_train = data.t.iloc[:split_at].to_numpy()
    y_train = data.df["nino34"].iloc[:split_at].to_numpy()

    t_test = data.t.iloc[split_at:].to_numpy()
    y_test = data.df["nino34"].iloc[split_at:].to_numpy()

    plot_multi_input_gp(
        t_train=t_train,
        y_train=y_train,
        t_test=t_test,
        y_test=y_test,
        mean_pred=mean_pred,
        std_pred=std_pred,
    )



if __name__ == "__main__":
    main()