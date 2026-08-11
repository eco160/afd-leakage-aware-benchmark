#!/usr/bin/env python3
"""Run one model family over both split regimes x 5 seeds, real + one permutation control each.

Checkpoints to results/<model>.json after every run; safe to re-invoke until it prints
ALL DONE. Thread use is capped at 2 so several model families can run concurrently.

Models:
  median_global   predict the training-pool median (chance-level reference)
  median_category predict the training-pool per-category median (the honest naive baseline)
  knn             k-NN on TF-IDF cosine -- automates the literature's manual
                  "assign the value of the nearest comparable food"
  ridge           L2 linear model on TF-IDF + category (alpha picked on the val split)
  rf              RandomForest, fixed a priori hyperparameters
  xgb / lgbm      gradient boosting, tree count by early stopping on the val split
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import argparse, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import load_data, load_splits, featurize, evaluate, checkpoint, save, save_preds

MODELS = ("median_global", "median_category", "knn", "ridge", "rf", "xgb", "lgbm", "mlp")


def predict_median(name, d, y, pool_ix, test_ix):
    if name == "median_global":
        return np.full(len(test_ix), float(np.median(y[pool_ix])))
    med, g = {}, float(np.median(y[pool_ix]))
    cats = d["category"][pool_ix]
    for c in np.unique(cats):
        med[c] = float(np.median(y[pool_ix][cats == c]))
    return np.array([med.get(c, g) for c in d["category"][test_ix]])


def fit_predict(name, seed, Xf, yf, Xv, yv, Xp, yp, Xt):
    if name == "ridge":
        from sklearn.linear_model import Ridge
        best = None
        # solver='lsqr': the default sparse_cg solver crashes under scipy>=1.14
        # (sklearn 1.2 passes the removed tol= kwarg to scipy.sparse.linalg.cg)
        for a in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
            m = Ridge(alpha=a, solver="lsqr").fit(Xf, yf)
            r = float(np.sqrt(np.mean((yv - m.predict(Xv)) ** 2)))
            if best is None or r < best[0]:
                best = (r, a)
        m = Ridge(alpha=best[1], solver="lsqr").fit(Xp, yp)
        return m.predict(Xt), {"alpha": best[1]}
    if name == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        best = None
        for k in (1, 3, 5, 10):
            m = KNeighborsRegressor(n_neighbors=k, metric="cosine",
                                    algorithm="brute").fit(Xf, yf)
            r = float(np.sqrt(np.mean((yv - m.predict(Xv)) ** 2)))
            if best is None or r < best[0]:
                best = (r, k)
        m = KNeighborsRegressor(n_neighbors=best[1], metric="cosine",
                                algorithm="brute").fit(Xp, yp)
        return m.predict(Xt), {"k": best[1]}
    if name == "mlp":
        # deep-vs-representation control: a neural net fed the SAME TF-IDF features as
        # the classical models, early-stopped on the same val split. partial_fit with a
        # RandomState INSTANCE keeps Adam's moments and the shuffle stream across epochs
        # (sklearn 1.2 re-seeds the rng and rebuilds the optimizer on every plain fit()).
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.neural_network import MLPRegressor
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        m = MLPRegressor(hidden_layer_sizes=(256,), alpha=1e-4, batch_size=128,
                         learning_rate_init=1e-3,
                         random_state=np.random.RandomState(seed))
        best_rmse, best_pred, best_epoch, bad = None, None, 0, 0
        for epoch in range(200):
            m.partial_fit(Xf, yf)               # one true epoch; optimizer state persists
            r = float(np.sqrt(np.mean((yv - m.predict(Xv)) ** 2)))
            if best_rmse is None or r < best_rmse - 1e-4:
                best_rmse, best_epoch, bad = r, epoch + 1, 0
                best_pred = m.predict(Xt)
            else:
                bad += 1
                if bad >= 12:
                    break
        return best_pred, {"epochs": best_epoch, "val_rmse": round(best_rmse, 4)}
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=300, max_features="sqrt", min_samples_leaf=2,
                                  n_jobs=2, random_state=seed).fit(Xp, yp)
        return m.predict(Xt), {"n_estimators": 300}
    if name == "xgb":
        from xgboost import XGBRegressor
        m = XGBRegressor(tree_method="hist", n_estimators=3000, learning_rate=0.05,
                         max_depth=6, subsample=0.8, colsample_bytree=0.8,
                         early_stopping_rounds=50, eval_metric="rmse",
                         n_jobs=2, random_state=seed)
        m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
        return m.predict(Xt), {"best_iteration": int(m.best_iteration)}
    if name == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.05, num_leaves=63,
                              subsample=0.8, colsample_bytree=0.8, n_jobs=2,
                              random_state=seed, verbosity=-1)
        m.fit(Xf, yf, eval_set=[(Xv, yv)], eval_metric="rmse",
              callbacks=[lgb.early_stopping(50, verbose=False)])
        return m.predict(Xt), {"best_iteration": int(m.best_iteration_)}
    raise ValueError(name)


def run_model(name):
    d, S, res = load_data(), load_splits(), checkpoint(name)
    for regime in ("grouped", "random"):
        for seed in range(5):
            sp = S[regime][str(seed)]
            fit_ix, val_ix, test_ix = (np.array(sp["fit"]), np.array(sp["val"]),
                                       np.array(sp["test"]))
            pool_ix = np.concatenate([fit_ix, val_ix])
            feats = None
            for mode in ("real", "perm"):
                key = f"{regime}_s{seed}_{mode}"
                if key in res["runs"]:
                    continue
                t0 = time.time()
                y = d["y"].copy()
                if mode == "perm":     # destroy the training signal, keep test labels real
                    rng = np.random.default_rng(1000 + seed)
                    y[pool_ix] = rng.permutation(y[pool_ix])
                if name in ("median_global", "median_category"):
                    yp, hp = predict_median(name, d, y, pool_ix, test_ix), {}
                else:
                    if feats is None:  # features are y-independent; share across real/perm
                        feats = featurize(d, fit_ix, [val_ix, pool_ix, test_ix])
                    Xf, Xv, Xp, Xt = feats
                    yp, hp = fit_predict(name, seed, Xf, y[fit_ix], Xv, y[val_ix],
                                         Xp, y[pool_ix], Xt)
                m = evaluate(d["y"][test_ix], yp)
                if mode == "real":   # artifacts before the checkpoint key (see dl.py)
                    save_preds(name, f"{regime}_s{seed}", test_ix, d["y"][test_ix], yp)
                save(res, name, key, m, dict(hp, mode=mode, seconds=round(time.time() - t0, 1)))
                print(f"{name} {key}: R2={m['r2']:.3f} rmse={m['rmse']:.3f} "
                      f"rho={m['spearman']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    run_model(ap.parse_args().model)
