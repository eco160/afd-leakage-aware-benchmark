#!/usr/bin/env python3
"""Build the modelling table and frozen train/val/test splits for the AFD benchmark.

Regimes:
  grouped : GroupShuffleSplit by normalised product name -- no name straddles train/test
  random  : ShuffleSplit with the same sizes -- the literature's default, deliberately leaky
5 seeds each. Inside every training pool a further 15% validation carve-out (group-aware in
the grouped regime) provides early stopping / hyperparameter selection without touching test.
All downstream models MUST load these frozen splits; nothing may re-split.
"""
import csv, json, re, unicodedata
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit


def norm_name(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


rows = list(csv.DictReader(open("processed/antioxidant_food_table.csv")))
recs = []
for i, r in enumerate(rows):
    y = float(r["antioxidant_mmol_per_100g"])
    recs.append(dict(idx=i, product=r["product"], category=r["category"],
                     category_no=int(r["category_no"]), group=norm_name(r["product"]),
                     y_raw=y, y_log=float(np.log1p(y))))

with open("processed/model_data.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
    w.writeheader()
    w.writerows(recs)

groups = np.array([r["group"] for r in recs])
idx = np.arange(len(recs))
splits = {}
for regime in ("grouped", "random"):
    splits[regime] = {}
    for seed in range(5):
        if regime == "grouped":
            (pool, test), = GroupShuffleSplit(n_splits=1, test_size=0.20,
                                              random_state=seed).split(idx, groups=groups)
            (f, v), = GroupShuffleSplit(n_splits=1, test_size=0.15,
                                        random_state=100 + seed).split(pool, groups=groups[pool])
        else:
            (pool, test), = ShuffleSplit(n_splits=1, test_size=0.20,
                                         random_state=seed).split(idx)
            (f, v), = ShuffleSplit(n_splits=1, test_size=0.15,
                                   random_state=100 + seed).split(pool)
        fit, val = pool[f], pool[v]
        # integrity: exact partition, no index reuse
        assert len(set(fit) | set(val) | set(test)) == len(idx)
        assert not (set(fit) | set(val)) & set(test) and not set(fit) & set(val)
        pool_groups = set(groups[np.concatenate([fit, val])])
        leak = float(np.mean([g in pool_groups for g in groups[test]]))
        if regime == "grouped":
            assert leak == 0.0, f"group leak in grouped split seed {seed}"
        splits[regime][str(seed)] = dict(
            fit=fit.tolist(), val=val.tolist(), test=test.tolist(),
            test_leak_fraction=round(leak, 4))
        print(f"{regime} s{seed}: fit {len(fit)}  val {len(val)}  test {len(test)}  "
              f"| test records whose name also occurs in training pool: {leak:.1%}")

json.dump(splits, open("processed/splits.json", "w"))
print("\nwrote processed/model_data.csv and processed/splits.json")
