#!/usr/bin/env python3
"""Day-4 models: DL/ML hybrids, ablations, the memorization diagnostic, prediction fusion.

Same frozen splits, metrics and checkpointing as runner.py / dl.py.

  copypaste        memorization ceiling: test records whose normalised name exists in the
                   training pool get that group's mean y; all others get the category
                   median. Under the grouped regime there are no twins by construction,
                   so it must EXACTLY equal the median_category baseline (cross-check).
  abl_ridge_text   ridge on TF-IDF only (no category one-hot)
  abl_ridge_cat    ridge on category one-hot only
  hybrid_xgb_emb   XGBoost on concatenated cnn+bilstm embeddings (384 dense dims)
  hybrid_xgb_full  XGBoost on TF-IDF + category + embeddings
  hybrid_ridge_full ridge on TF-IDF + category + standardized embeddings
  fusion_all       equal-weight average of the 8 learned models' test predictions
  fusion_ltd       equal-weight average of ridge + lgbm + cnn (best linear/tree/deep)

Embeddings are reused from the Day-3 real runs of the SAME (regime, seed): they are
functions of text+category computed by models that never saw test labels (val was used
only for early stopping). Perm controls retrain the downstream learner on shuffled pool
y with the same fixed embeddings; fusion has no training step, so no perm mode.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse, csv, sys, time
import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import load_data, load_splits, evaluate, checkpoint, save, save_preds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

MODELS = ("copypaste", "abl_ridge_text", "abl_ridge_cat", "hybrid_xgb_emb",
          "hybrid_xgb_full", "hybrid_ridge_full", "fusion_all", "fusion_ltd")
FUSIONS = {"fusion_all": ["ridge", "lgbm", "xgb", "rf", "knn", "mlp", "cnn", "bilstm"],
           "fusion_ltd": ["ridge", "lgbm", "cnn"]}


def load_embs(regime, seed):
    e = [np.load(f"results/embeds/{a}_{regime}_s{seed}.npz")["emb"]
         for a in ("cnn", "bilstm")]
    return np.hstack(e).astype(np.float64)          # (3135, 384)


def featurize_variant(d, fit_ix, other_ixs, variant):
    """variant: 'text' | 'cat' | 'full'; all fitting on fit_ix only."""
    blocks = []
    if variant in ("text", "full"):
        word = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=20000,
                               sublinear_tf=True)
        word.fit(d["product"][fit_ix])
        char.fit(d["product"][fit_ix])
        blocks.append(lambda ix: sparse.hstack([word.transform(d["product"][ix]),
                                                char.transform(d["product"][ix])]))
    if variant in ("cat", "full"):
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        ohe.fit(d["category"][fit_ix].reshape(-1, 1))
        blocks.append(lambda ix: ohe.transform(d["category"][ix].reshape(-1, 1)))

    def tf(ix):
        return sparse.hstack([b(ix) for b in blocks]).tocsr()

    return [tf(ix) for ix in [fit_ix] + list(other_ixs)]


def ridge_fit(Xf, yf, Xv, yv, Xp, yp, Xt):
    from sklearn.linear_model import Ridge
    best = None
    for a in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
        m = Ridge(alpha=a, solver="lsqr").fit(Xf, yf)
        r = float(np.sqrt(np.mean((yv - m.predict(Xv)) ** 2)))
        if best is None or r < best[0]:
            best = (r, a)
    m = Ridge(alpha=best[1], solver="lsqr").fit(Xp, yp)
    return m.predict(Xt), {"alpha": best[1]}


def xgb_fit(seed, Xf, yf, Xv, yv, Xt):
    from xgboost import XGBRegressor
    m = XGBRegressor(tree_method="hist", n_estimators=3000, learning_rate=0.05,
                     max_depth=6, subsample=0.8, colsample_bytree=0.8,
                     early_stopping_rounds=50, eval_metric="rmse",
                     n_jobs=2, random_state=seed)
    m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
    return m.predict(Xt), {"best_iteration": int(m.best_iteration)}


def copypaste(d, y, pool_ix, test_ix):
    gmean = {}
    for i in pool_ix:
        gmean.setdefault(d["group"][i], []).append(y[i])
    gmean = {g: float(np.mean(v)) for g, v in gmean.items()}
    cats = d["category"][pool_ix]
    cmed = {c: float(np.median(y[pool_ix][cats == c])) for c in np.unique(cats)}
    gmed = float(np.median(y[pool_ix]))
    out, twins = [], 0
    for i in test_ix:
        g = d["group"][i]
        if g in gmean:
            out.append(gmean[g])
            twins += 1
        else:
            out.append(cmed.get(d["category"][i], gmed))
    return np.array(out), {"twin_fraction": round(twins / len(test_ix), 4)}


def fusion_preds(members, regime, seed):
    ys, ps, idx = None, [], None
    for m in members:
        rows = sorted(csv.DictReader(open(f"results/preds/{m}_{regime}_s{seed}.csv")),
                      key=lambda r: int(r["idx"]))
        p = np.array([float(r["y_pred_log"]) for r in rows])
        yt = np.array([float(r["y_true_log"]) for r in rows])
        if ys is None:
            ys, idx = yt, [int(r["idx"]) for r in rows]
        else:
            assert np.allclose(ys, yt), f"prediction files misaligned for {m}"
        ps.append(p)
    return idx, ys, np.mean(ps, axis=0)


def run_model(name, limit=None):
    d, S, res = load_data(), load_splits(), checkpoint(name)
    done = 0
    for regime in ("grouped", "random"):
        for seed in range(5):
            sp = S[regime][str(seed)]
            fit_ix, val_ix, test_ix = (np.array(sp["fit"]), np.array(sp["val"]),
                                       np.array(sp["test"]))
            pool_ix = np.concatenate([fit_ix, val_ix])
            modes = ("real",) if name in FUSIONS else ("real", "perm")
            feats = embs = None
            for mode in modes:
                key = f"{regime}_s{seed}_{mode}"
                if key in res["runs"]:
                    continue
                t0 = time.time()
                y = d["y"].copy()
                if mode == "perm":
                    prng = np.random.default_rng(1000 + seed)
                    y[pool_ix] = prng.permutation(y[pool_ix])

                if name in FUSIONS:
                    idx, ys, yp = fusion_preds(FUSIONS[name], regime, seed)
                    m = evaluate(ys, yp)
                    with open(f"results/preds/{name}_{regime}_s{seed}.csv", "w",
                              newline="") as fh:
                        w = csv.writer(fh)
                        w.writerow(["idx", "y_true_log", "y_pred_log"])
                        w.writerows(zip(idx, ys, yp))
                    save(res, name, key, m, {"mode": mode,
                                             "members": ",".join(FUSIONS[name]),
                                             "seconds": round(time.time() - t0, 1)})
                    print(f"{name} {key}: R2={m['r2']:.3f}", flush=True)
                    done += 1
                    continue

                if name == "copypaste":
                    yp, hp = copypaste(d, y, pool_ix, test_ix)
                elif name == "abl_ridge_text":
                    if feats is None:
                        feats = featurize_variant(d, fit_ix, [val_ix, pool_ix, test_ix],
                                                  "text")
                    Xf, Xv, Xp, Xt = feats
                    yp, hp = ridge_fit(Xf, y[fit_ix], Xv, y[val_ix], Xp, y[pool_ix], Xt)
                elif name == "abl_ridge_cat":
                    if feats is None:
                        feats = featurize_variant(d, fit_ix, [val_ix, pool_ix, test_ix],
                                                  "cat")
                    Xf, Xv, Xp, Xt = feats
                    yp, hp = ridge_fit(Xf, y[fit_ix], Xv, y[val_ix], Xp, y[pool_ix], Xt)
                elif name == "hybrid_xgb_emb":
                    if embs is None:
                        embs = load_embs(regime, seed)
                    yp, hp = xgb_fit(seed, embs[fit_ix], y[fit_ix],
                                     embs[val_ix], y[val_ix], embs[test_ix])
                elif name in ("hybrid_xgb_full", "hybrid_ridge_full"):
                    if embs is None:
                        embs = load_embs(regime, seed)
                    if feats is None:
                        base = featurize_variant(d, fit_ix, [val_ix, pool_ix, test_ix],
                                                 "full")
                        # standardize embeddings on fit-partition statistics only.
                        # dims that are (near-)constant on fit — dead ReLU units — get
                        # divisor 1, NOT sd+eps: dividing by ~1e-8 explodes the rare
                        # rows where such a dim activates outside fit (observed 5e6
                        # entries that collapsed the ridge pool refit to zero coefs)
                        mu = embs[fit_ix].mean(0)
                        sd = embs[fit_ix].std(0)
                        sd[sd < 1e-6] = 1.0
                        E = (embs - mu) / sd
                        feats = [sparse.hstack([B, sparse.csr_matrix(E[ix])]).tocsr()
                                 for B, ix in zip(base, [fit_ix, val_ix, pool_ix, test_ix])]
                    Xf, Xv, Xp, Xt = feats
                    if name == "hybrid_xgb_full":
                        yp, hp = xgb_fit(seed, Xf, y[fit_ix], Xv, y[val_ix], Xt)
                    else:
                        yp, hp = ridge_fit(Xf, y[fit_ix], Xv, y[val_ix],
                                           Xp, y[pool_ix], Xt)
                else:
                    raise ValueError(name)

                m = evaluate(d["y"][test_ix], yp)
                if mode == "real":
                    save_preds(name, f"{regime}_s{seed}", test_ix, d["y"][test_ix], yp)
                save(res, name, key, m, dict(hp, mode=mode,
                                             seconds=round(time.time() - t0, 1)))
                print(f"{name} {key}: R2={m['r2']:.3f} rmse={m['rmse']:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                done += 1
                if limit and done >= limit:
                    print("LIMIT REACHED", flush=True)
                    return
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run_model(a.model, a.limit)
