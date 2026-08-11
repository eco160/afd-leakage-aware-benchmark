#!/usr/bin/env python3
"""Extract the Antioxidant Food Table (Carlsen et al. 2010, Additional file 1) to CSV.

Uses word-level bounding boxes (`pdftotext -bbox-layout`) rather than the flattened text
layout. Table cells are vertically centred and wrap over several lines, so a plain-text parse
mis-assigns wrapped product names to the neighbouring record. Here each numeric value in the
antioxidant column anchors a row band (bounded by the midpoints to the values above and below),
and every word is assigned to a band by its y-centre and to a column by its x-centre.
"""
import csv, html, os, re, subprocess, sys
from collections import Counter

PDF = "raw/antioxidant_food_table/antioxidant_food_table_carlsen2010.pdf"
XML = "/tmp/afd_bbox.xml"
if not os.path.exists(XML):
    subprocess.run(["pdftotext", "-bbox-layout", PDF, XML], check=True)
doc = open(XML, encoding="utf-8", errors="replace").read()

WORD = re.compile(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">'
                  r'(.*?)</word>', re.S)
FLOAT = re.compile(r"^\d+(?:\.\d+)?$")
CATLINE = re.compile(r"^Category\s+(\d+)\s*(.*)$")

# header anchor -> column key; x positions are read per page from these words
HEADERS = [("Product", "product"), ("Manufacturer", "manufacturer"),
           ("Procured", "procured"), ("Antioxidant", "value"), ("Comment", "comment")]

rows, skipped = [], []
cat_no = cat_name = None

for pageno, page in enumerate(doc.split("<page ")[1:]):
    words = [(float(a), float(b), float(c), float(d), html.unescape(t).strip())
             for a, b, c, d, t in WORD.findall(page)]
    words = [w for w in words if w[4]]
    if not words:
        continue

    # drop the page footer ("Page N of 138  The Antioxidant Food Table, Carlsen et al. 2010"),
    # whose trailing year otherwise lands in the antioxidant-value column
    words = [w for w in words if w[1] < 780]

    # ---- category heading (top-left of the page) -------------------------------
    top = sorted([w for w in words if w[1] < 95 and w[0] < 300],
                 key=lambda w: (round(w[1], 1), w[0]))
    heading = " ".join(w[4] for w in top)
    heading = re.split(r"\b(?:Antioxidant|Manufacturer|Procured|Comment)\b", heading)[0]
    heading = re.sub(r"\s*continued\s*", " ", heading).strip()
    m = CATLINE.match(heading)
    if m:
        cat_no = int(m.group(1))
        name = re.sub(r"\s*continued\s*$", "", m.group(2)).strip()
        if name:
            cat_name = name
    if cat_no is None:
        continue                                    # cover / contents pages

    # ---- column x-boundaries from this page's header words ----------------------
    xs = {}
    for token, key in HEADERS:
        cand = [w for w in words if w[4].startswith(token) and w[1] < 140]
        if cand:
            xs[key] = min(c[0] for c in cand)
    if "value" not in xs or "manufacturer" not in xs:
        continue
    xs.setdefault("product", 60.0)
    xs.setdefault("procured", (xs["manufacturer"] + xs["value"]) / 2)
    xs.setdefault("comment", xs["value"] + 60)
    order = ["product", "manufacturer", "procured", "value", "comment"]
    bounds = []
    for i, k in enumerate(order[:-1]):
        bounds.append((k, xs[k] - 6, xs[order[i + 1]] - 6))
    bounds.append(("comment", xs["comment"] - 6, 10_000.0))

    def column(w):
        xc = (w[0] + w[2]) / 2
        for k, lo, hi in bounds:
            if lo <= xc < hi:
                return k
        return None

    header_bottom = max([w[3] for w in words if w[1] < 140 and
                         re.search(r"mmol/100g|Comment|origin", w[4])] or [140.0])
    body = [w for w in words if w[1] > header_bottom - 1]

    # ---- row bands, anchored on values ------------------------------------------
    anchors = sorted([w for w in body if column(w) == "value" and FLOAT.match(w[4])],
                     key=lambda w: w[1])
    if not anchors:
        continue
    centres = [(a[1] + a[3]) / 2 for a in anchors]
    edges = [header_bottom - 1]
    edges += [(centres[i] + centres[i + 1]) / 2 for i in range(len(centres) - 1)]
    edges.append(10_000.0)

    for i, anchor in enumerate(anchors):
        lo, hi = edges[i], edges[i + 1]
        cells = {k: [] for k in order}
        for w in body:
            yc = (w[1] + w[3]) / 2
            if not (lo <= yc < hi):
                continue
            k = column(w)
            if k:
                cells[k].append(w)
        def join(k):
            ws = sorted(cells[k], key=lambda w: (round(w[1], 1), w[0]))
            return re.sub(r"\s+", " ", " ".join(w[4] for w in ws)).strip()
        # strip bullet/marker glyphs the PDF uses in front of some product names
        product = re.sub(r"^[^\w(]+", "", join("product")).strip()
        if not product:
            skipped.append((pageno, anchor[4]))
            continue
        rows.append({"category_no": cat_no, "category": cat_name, "product": product,
                     "manufacturer_origin": join("manufacturer"),
                     "procured_in": join("procured"),
                     "antioxidant_mmol_per_100g": float(anchor[4]),
                     "comment_ref": join("comment")})

FIX = {"Diary and diary products": "Dairy and dairy products",
       "Breakfast cereal": "Breakfast cereals",
       "Herbal / traditional plant medicine": "Herbal/traditional plant medicine"}
for r in rows:
    r["category"] = FIX.get(r["category"], r["category"])
# a category title that wraps onto a third line is truncated on some pages; adopt the
# longest name seen for each category number as the canonical one
canon = {}
for r in rows:
    n = r["category_no"]
    if len(r["category"]) > len(canon.get(n, "")):
        canon[n] = r["category"]
for r in rows:
    r["category"] = canon[r["category_no"]]

os.makedirs("processed", exist_ok=True)
out = "processed/antioxidant_food_table.csv"
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"extracted {len(rows)} records -> {out}")
print(f"published total 3139; difference {len(rows)-3139:+d} ({100*len(rows)/3139:.1f}%)")
if skipped:
    print(f"{len(skipped)} values dropped for having no product text (first 5: {skipped[:5]})")
c = Counter((r["category_no"], r["category"]) for r in rows)
print(f"{len(c)} categories (published 24)")
