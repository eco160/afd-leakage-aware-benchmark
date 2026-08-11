"""Shared evaluation harness for the AFD benchmark.

Rules enforced here:
  - every model loads the SAME frozen splits (processed/splits.json); nothing re-splits
  - all feature fitting (TF-IDF vocabularies, category encoder) happens on the fit
    partition only; val/pool/test are only ever transformed
  - metrics are computed on log1p(FRAP); Spearman is scale-free
  - every run checkpoints to results/<model>.json so interrupted runs resume
"""
import csv, json, os
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


def load_data():
    rows = list(csv.DictReader(open("processed/model_data.csv")))
    return dict(
        product=np.array([r["product"] for r in rows]),
        category=np.array([r["category"] for r in rows]),
        group=np.array([r["group"] for r in rows]),
        y=np.array([float(r["y_log"]) for r in rows]),
        y_raw=np.array([float(r["y_raw"]) for r in rows]),
    )


def load_splits():
    return json.load(open("processed/splits.json"))


def featurize(d, fit_ix, other_ixs):
    """Fit text+category features on fit_ix ONLY; transform fit_ix and every other index set."""
    word = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=20000,
                           sublinear_tf=True)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    P = d["product"][fit_ix]
    word.fit(P)
    char.fit(P)
    ohe.fit(d["category"][fit_ix].reshape(-1, 1))

    def tf(ix):
        p = d["product"][ix]
        return sparse.hstack([word.transform(p), char.transform(p),
                              ohe.transform(d["category"][ix].reshape(-1, 1))]).tocsr()

    return [tf(ix) for ix in [fit_ix] + list(other_ixs)]


def evaluate(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return dict(r2=1.0 - ss_res / ss_tot,
                rmse=float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                mae=float(np.mean(np.abs(y_true - y_pred))),
                spearman=float(spearmanr(y_true, y_pred).statistic))


def checkpoint(model):
    p = f"results/{model}.json"
    return json.load(open(p)) if os.path.exists(p) else {"model": model, "runs": {}}


def save(res, model, key, metrics, extra=None):
    res["runs"][key] = dict(metrics, **(extra or {}))
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/{model}.json", "w"), indent=1)


def save_preds(model, key, idx, y_true, y_pred):
    os.makedirs("results/preds", exist_ok=True)
    with open(f"results/preds/{model}_{key}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "y_true_log", "y_pred_log"])
        for a, b, c in zip(idx, y_true, y_pred):
            w.writerow([int(a), float(b), float(c)])
