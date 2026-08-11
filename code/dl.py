#!/usr/bin/env python3
"""Character-level deep models for the AFD benchmark: char-CNN and BiLSTM.

Protocol identical to runner.py: same frozen splits, same metrics, checkpointed runs,
one y-permutation control per (regime, seed). The char vocabulary and category map are
built on the fit partition ONLY; val (group-aware in the grouped regime) drives early
stopping; like xgb/lgbm, the deep models train on fit only. After each real run the
penultimate-layer embedding of ALL rows is saved for the Day-4 hybrid (text+category
only — embeddings never see y at inference time; test rows never influence training).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "3")
import argparse, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import load_data, load_splits, evaluate, checkpoint, save, save_preds

torch.set_num_threads(3)
MAXLEN = 80


def build_vocab(texts):
    chars = sorted({c for t in texts for c in t.lower()})
    return {c: i + 2 for i, c in enumerate(chars)}          # 0 = pad, 1 = unk


def encode(texts, vocab):
    X = np.zeros((len(texts), MAXLEN), dtype=np.int64)
    for i, t in enumerate(texts):
        for j, c in enumerate(t.lower()[:MAXLEN]):
            X[i, j] = vocab.get(c, 1)
    return X


class CharCNN(nn.Module):
    def __init__(self, vocab_size, n_cat, emb=48, filters=96, widths=(2, 3, 4, 5),
                 cat_emb=16, hidden=192, p=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, filters, w, padding=w // 2)
                                    for w in widths])
        self.cat = nn.Embedding(n_cat, cat_emb)
        self.head = nn.Sequential(nn.Linear(filters * len(widths) + cat_emb, hidden),
                                  nn.ReLU(), nn.Dropout(p))
        self.out = nn.Linear(hidden, 1)

    def features(self, x, c):
        e = self.emb(x).transpose(1, 2)
        h = torch.cat([torch.relu(cv(e)).amax(dim=2) for cv in self.convs], dim=1)
        return self.head(torch.cat([h, self.cat(c)], dim=1))

    def forward(self, x, c):
        return self.out(self.features(x, c)).squeeze(-1)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, n_cat, emb=48, hidden_rnn=96, cat_emb=16,
                 hidden=192, p=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.rnn = nn.LSTM(emb, hidden_rnn, batch_first=True, bidirectional=True)
        self.cat = nn.Embedding(n_cat, cat_emb)
        self.head = nn.Sequential(nn.Linear(2 * hidden_rnn + cat_emb, hidden),
                                  nn.ReLU(), nn.Dropout(p))
        self.out = nn.Linear(hidden, 1)

    def features(self, x, c):
        h, _ = self.rnn(self.emb(x))
        # mask pad timesteps so max-pooling never selects recurrent output over padding
        h = h.masked_fill((x == 0).unsqueeze(-1), float("-inf")).amax(dim=1)
        return self.head(torch.cat([h, self.cat(c)], dim=1))

    def forward(self, x, c):
        return self.out(self.features(x, c)).squeeze(-1)


def batches(n, size, rng=None):
    ix = np.arange(n) if rng is None else rng.permutation(n)
    for i in range(0, n, size):
        yield ix[i:i + size]


def train_one(arch, seed, Xall, Call, y, fit_ix, val_ix, test_ix, n_vocab, n_cat,
              max_epochs=150, patience=12, batch=64):
    torch.manual_seed(seed)
    model = (CharCNN if arch == "cnn" else BiLSTM)(n_vocab, n_cat)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    Xt = torch.from_numpy(Xall)
    Ct = torch.from_numpy(Call)
    yt = torch.from_numpy(y.astype(np.float32))
    rng = np.random.default_rng(seed)
    best_rmse, best_state, best_epoch, bad = None, None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        for b in batches(len(fit_ix), batch, rng):
            ix = fit_ix[b]
            opt.zero_grad()
            loss = lossf(model(Xt[ix], Ct[ix]), yt[ix])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xt[val_ix], Ct[val_ix]).numpy()
        rmse = float(np.sqrt(np.mean((y[val_ix] - pv) ** 2)))
        if best_rmse is None or rmse < best_rmse - 1e-4:
            best_rmse, best_epoch, bad = rmse, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pt = model(Xt[test_ix], Ct[test_ix]).numpy()
    return model, pt, {"best_epoch": best_epoch, "val_rmse": round(best_rmse, 4)}


def embed_all(model, Xall, Call, batch=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xall), batch):
            out.append(model.features(torch.from_numpy(Xall[i:i + batch]),
                                      torch.from_numpy(Call[i:i + batch])).numpy())
    return np.vstack(out)


def run(arch, limit=None):
    d, S, res = load_data(), load_splits(), checkpoint(arch)
    done = 0
    for regime in ("grouped", "random"):
        for seed in range(5):
            sp = S[regime][str(seed)]
            fit_ix, val_ix, test_ix = (np.array(sp["fit"]), np.array(sp["val"]),
                                       np.array(sp["test"]))
            pool_ix = np.concatenate([fit_ix, val_ix])
            vocab = build_vocab(d["product"][fit_ix])
            cats = sorted(set(d["category"][fit_ix]))
            cmap = {c: i + 1 for i, c in enumerate(cats)}   # 0 = unseen category
            Xall = encode(d["product"], vocab)
            Call = np.array([cmap.get(c, 0) for c in d["category"]])
            for mode in ("real", "perm"):
                key = f"{regime}_s{seed}_{mode}"
                if key in res["runs"]:
                    continue
                t0 = time.time()
                y = d["y"].copy()
                if mode == "perm":
                    prng = np.random.default_rng(1000 + seed)
                    y[pool_ix] = prng.permutation(y[pool_ix])
                model, yp, hp = train_one(arch, seed, Xall, Call, y,
                                          fit_ix, val_ix, test_ix,
                                          len(vocab) + 2, len(cats) + 1)
                m = evaluate(d["y"][test_ix], yp)
                if mode == "real":
                    # artifacts BEFORE the checkpoint key, so an interrupted run
                    # redoes the whole (regime, seed) on resume instead of leaving
                    # a silent gap in preds/embeds that Day-4 depends on
                    save_preds(arch, f"{regime}_s{seed}", test_ix, d["y"][test_ix], yp)
                    os.makedirs("results/embeds", exist_ok=True)
                    np.savez_compressed(f"results/embeds/{arch}_{regime}_s{seed}.npz",
                                        emb=embed_all(model, Xall, Call))
                save(res, arch, key, m, dict(hp, mode=mode,
                                             seconds=round(time.time() - t0, 1)))
                print(f"{arch} {key}: R2={m['r2']:.3f} rmse={m['rmse']:.3f} "
                      f"rho={m['spearman']:.3f} ep={hp['best_epoch']} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                done += 1
                if limit and done >= limit:
                    print("LIMIT REACHED", flush=True)
                    return
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=("cnn", "bilstm"))
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N runs (smoke test)")
    run(ap.parse_args().arch, ap.parse_args().limit)
