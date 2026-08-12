#!/usr/bin/env python3
"""Pretrained-embedding control: are the deep nets' losses due to missing pre-training,
or to the character-level architectures themselves?

Representation: mean-pooled pretrained fastText vectors (wiki-news-300d-1M.vec,
Mikolov et al. 2018), an external constant never fitted on any partition, concatenated
with the category one-hot (fitted on the fit partition only, as everywhere else).
Standardisation statistics (mu/sd) come from the fit partition only, with the same
dead-dimension guard as the Day-4 hybrids.

  ft_ridge   pretrained embedding + category OHE -> ridge (alpha on val, refit on pool)
  ft_xgb     pretrained embedding + category OHE -> XGBoost (early stopping on val)

Same frozen splits, metrics, perm convention and checkpointing as runner.py/day4.py.
Vectors are NOT redistributed; download wiki-news-300d-1M.vec.zip from
https://dl.fbaipublicfiles.com/fasttext/vectors-english/ into raw/fasttext/.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse, io, re, sys, time, zipfile
import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import load_data, load_splits, evaluate, checkpoint, save, save_preds
from sklearn.preprocessing import OneHotEncoder

MODELS = ("ft_ridge", "ft_xgb")
VEC_ZIP = "raw/fasttext/wiki-news-300d-1M.vec.zip"
VEC_NAME = "wiki-news-300d-1M.vec"
TOKEN_RE = re.compile(r"[^\W_]+")   # Unicode word chars: accented tokens stay whole


def tokens(s):
    return TOKEN_RE.findall(s)


def load_vectors(products):
    """Stream the .vec file, keeping only case variants of tokens that occur in the corpus.

    The corpus-wide filter is a memory optimisation with row-independent lookups:
    each row's embedding equals what loading the entire vector file would produce."""
    wanted = set()
    for p in products:
        for t in tokens(p):
            wanted.update((t, t.lower(), t.title()))
    vec = {}
    with zipfile.ZipFile(VEC_ZIP) as z, z.open(VEC_NAME) as raw:
        fh = io.TextIOWrapper(raw, encoding="utf-8", newline="\n")
        header = fh.readline().split()
        dim = int(header[1])
        for line in fh:
            sp = line.rstrip().split(" ")
            if sp[0] in wanted:
                vec[sp[0]] = np.array(sp[1:], dtype=np.float64)
    assert all(v.shape == (dim,) for v in vec.values())
    return vec, dim


def embed(products, vec, dim):
    """Mean-pool pretrained vectors; exact -> lower -> title lookup; zeros if no hit."""
    E = np.zeros((len(products), dim))
    oov = hit = empty = 0
    for i, p in enumerate(products):
        found = []
        for t in tokens(p):
            v = vec.get(t)
            if v is None:
                v = vec.get(t.lower())
            if v is None:
                v = vec.get(t.title())
            if v is None:
                oov += 1
            else:
                hit += 1
                found.append(v)
        if found:
            E[i] = np.mean(found, axis=0)
        else:
            empty += 1
    return E, dict(oov_token_rate=round(oov / max(oov + hit, 1), 4), empty_rows=empty)


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


def run_model(name):
    d, S, res = load_data(), load_splits(), checkpoint(name)
    vec, dim = load_vectors(d["product"])
    E_all, cov = embed(d["product"], vec, dim)
    print(f"{name}: {len(vec)} vectors kept, OOV token rate {cov['oov_token_rate']}, "
          f"{cov['empty_rows']} rows without any hit", flush=True)
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
                if feats is None:      # features are y-independent; share across real/perm
                    mu = E_all[fit_ix].mean(axis=0)
                    sd = E_all[fit_ix].std(axis=0)
                    sd[sd < 1e-6] = 1.0
                    E = (E_all - mu) / sd
                    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
                    ohe.fit(d["category"][fit_ix].reshape(-1, 1))
                    feats = [sparse.hstack([sparse.csr_matrix(E[ix]),
                                            ohe.transform(d["category"][ix].reshape(-1, 1))
                                            ]).tocsr()
                             for ix in (fit_ix, val_ix, pool_ix, test_ix)]
                Xf, Xv, Xp, Xt = feats
                if name == "ft_ridge":
                    yp, hp = ridge_fit(Xf, y[fit_ix], Xv, y[val_ix], Xp, y[pool_ix], Xt)
                else:
                    yp, hp = xgb_fit(seed, Xf, y[fit_ix], Xv, y[val_ix], Xt)
                m = evaluate(d["y"][test_ix], yp)
                if mode == "real":   # artifacts before the checkpoint key (see dl.py)
                    save_preds(name, f"{regime}_s{seed}", test_ix, d["y"][test_ix], yp)
                save(res, name, key, m, dict(hp, mode=mode, **cov,
                                             seconds=round(time.time() - t0, 1)))
                print(f"{name} {key}: R2={m['r2']:.3f} rmse={m['rmse']:.3f} "
                      f"rho={m['spearman']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    run_model(ap.parse_args().model)
