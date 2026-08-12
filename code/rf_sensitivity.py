#!/usr/bin/env python3
"""Random-forest hyperparameter sensitivity check (post-hoc, both regimes, 5 seeds).

The benchmark fixed the forest a priori (300 trees, max_features="sqrt",
min_samples_leaf=2). This script asks whether validation-based tuning, as used for
ridge and the boosted models, would have changed the reported result.

Protocol per (regime, seed):
  1. fit every grid configuration on the FIT partition, score RMSE on the VAL split;
  2. select the configuration with the lowest val RMSE;
  3. refit the selected and the fixed configuration on the POOL, score on TEST.
The fixed-config pool refit must reproduce results/rf.json to within float tolerance
1e-9 (same seed, same params; parallel predict carries 1-ulp accumulation noise):
a built-in correctness check, asserted before the run is checkpointed.

Grid: max_features in {"sqrt", 1000, 3000} x min_samples_leaf in {1, 2, 5}.
Checkpoints to results/rf_sensitivity.json; safe to interrupt and resume.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import load_data, load_splits, evaluate, checkpoint, save, save_preds, featurize
from sklearn.ensemble import RandomForestRegressor

GRID = [(mf, msl) for mf in ("sqrt", 1000, 3000) for msl in (1, 2, 5)]
FIXED = ("sqrt", 2)
NAME = "rf_sensitivity"


def rf(seed, mf, msl):
    return RandomForestRegressor(n_estimators=300, max_features=mf,
                                 min_samples_leaf=msl, n_jobs=2, random_state=seed)


def main():
    d, S, res = load_data(), load_splits(), checkpoint(NAME)
    for regime in ("grouped", "random"):
        for seed in range(5):
            sp = S[regime][str(seed)]
            fit_ix, val_ix, test_ix = (np.array(sp["fit"]), np.array(sp["val"]),
                                       np.array(sp["test"]))
            pool_ix = np.concatenate([fit_ix, val_ix])
            feats = None
            # stage 1: validation scores for the whole grid
            for mf, msl in GRID:
                key = f"{regime}_s{seed}_val_mf{mf}_leaf{msl}"
                if key in res["runs"]:
                    continue
                if feats is None:
                    feats = featurize(d, fit_ix, [val_ix, pool_ix, test_ix])
                Xf, Xv, Xp, Xt = feats
                t0 = time.time()
                m = rf(seed, mf, msl).fit(Xf, d["y"][fit_ix])
                rmse = float(np.sqrt(np.mean((d["y"][val_ix] - m.predict(Xv)) ** 2)))
                save(res, NAME, key, {"val_rmse": rmse},
                     dict(max_features=str(mf), min_samples_leaf=msl, stage="val",
                          seconds=round(time.time() - t0, 1)))
                print(f"{key}: val_rmse={rmse:.4f} ({time.time() - t0:.0f}s)", flush=True)
            # stage 2: pool refit + test for the val-selected and the fixed config
            vals = {(r["max_features"], r["min_samples_leaf"]): r["val_rmse"]
                    for k, r in res["runs"].items()
                    if k.startswith(f"{regime}_s{seed}_val_")}
            sel = min(vals, key=vals.get)
            for tag, (mf, msl) in (("selected", sel), ("fixed", (str(FIXED[0]), FIXED[1]))):
                key = f"{regime}_s{seed}_test_{tag}"
                if key in res["runs"]:
                    continue
                if feats is None:
                    feats = featurize(d, fit_ix, [val_ix, pool_ix, test_ix])
                Xf, Xv, Xp, Xt = feats
                t0 = time.time()
                mf_arg = mf if mf == "sqrt" else int(mf)
                m = rf(seed, mf_arg, int(msl)).fit(Xp, d["y"][pool_ix])
                yp = m.predict(Xt)
                met = evaluate(d["y"][test_ix], yp)
                if tag == "fixed":   # must reproduce the frozen Day-3 run
                    ref = json.load(open("results/rf.json"))["runs"][f"{regime}_s{seed}_real"]
                    diff = max(abs(met[k2] - ref[k2])
                               for k2 in ("r2", "rmse", "mae", "spearman"))
                    assert diff < 1e-9, f"{key}: does not reproduce rf.json ({diff:.3e})"
                    print(f"{key}: reproduces rf.json, max metric diff {diff:.2e}", flush=True)
                save_preds(NAME, f"{regime}_s{seed}_{tag}", test_ix, d["y"][test_ix], yp)
                save(res, NAME, key, met,
                     dict(max_features=str(mf), min_samples_leaf=int(msl), stage="test",
                          tag=tag, seconds=round(time.time() - t0, 1)))
                print(f"{key}: mf={mf} leaf={msl} R2={met['r2']:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
