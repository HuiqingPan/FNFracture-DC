import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
import argparse

REQUIRED_POINTS = ["horizon_p1", "horizon_p2", "fracture_p1", "fracture_p2"]
IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
JSON_SUFFIXES = [".json", ".JSON"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def md5_file(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def point_xy(shape):
    points = shape.get("points") or []
    if not points:
        return None
    return float(points[0][0]), float(points[0][1])


def line_angle(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


def acute_angle_between(a, b):
    d = abs((a - b + 180) % 360 - 180)
    return min(d, 180 - d)


def pauwels_class(angle):
    if angle < 30:
        return 0
    if angle <= 50:
        return 1
    return 2


def _norm_stem_variants(stem):
    """
    Return possible filename stems for an ID.

    This handles both:
    - Excel ID read as 322, file named 322.JPG
    - Excel ID shown as 0322 or 00322, file named 0322.JPG
    """
    s = str(stem).strip()
    variants = [s]

    # If the ID can be parsed as int, also allow the non-leading-zero version.
    try:
        variants.append(str(int(float(s))))
    except Exception:
        pass

    # Deduplicate while preserving order.
    return list(dict.fromkeys(variants))


def find_file_by_stem(parent, stem, suffixes):
    """
    Find a file by stem and suffix, case-insensitively.

    Examples:
    - stem=322 can match 322.jpg / 322.JPG / 322.jpeg / 322.PNG
    - stem=00322 can match 00322.JPG and, if Excel converted it, 322.JPG

    Returns:
        pathlib.Path or None

    Raises:
        RuntimeError if multiple files match the same ID.
    """
    parent = Path(parent)
    suffixes_lower = {s.lower() for s in suffixes}
    stems = _norm_stem_variants(stem)

    # First try exact common paths quickly.
    for one_stem in stems:
        for suffix in suffixes:
            p = parent / f"{one_stem}{suffix}"
            if p.exists():
                return p

    if not parent.exists():
        return None

    # Then scan the directory and compare case-insensitively.
    stem_set = {x.lower() for x in stems}
    matches = [
        p for p in parent.iterdir()
        if p.is_file()
        and p.stem.lower() in stem_set
        and p.suffix.lower() in suffixes_lower
    ]

    # Deduplicate paths and sort for deterministic behavior.
    matches = sorted(set(matches), key=lambda x: str(x).lower())

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple files matched stem={stem!r} under {parent}: "
            + ", ".join(str(p) for p in matches)
        )

    return None


def count_json_files(path):
    path = Path(path)
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".json")


def find_annotation_dir(root):
    """
    Find the directory containing annotation JSON files.

    Fixed:
    - The old version used glob("*.json"), which misses .JSON on case-sensitive systems.
    """
    root = Path(root)
    candidates = [root] + [p for p in root.iterdir() if p.is_dir()]
    scored = [(count_json_files(p), p) for p in candidates]
    scored = [x for x in scored if x[0] > 0]
    if not scored:
        raise FileNotFoundError(f"No annotation directory with JSON files found under {root}")

    return max(scored, key=lambda x: x[0])[1]


def crop_fracture_roi(image_path, fp1, fp2, out_path, margin_scale=2.2, min_side=96):
    img = Image.open(image_path).convert("L")
    w, h = img.size
    pts = np.array([fp1, fp2], dtype=np.float32)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    bw = max(float(min_side), float(x2 - x1))
    bh = max(float(min_side), float(y2 - y1))
    margin = max(bw, bh) * margin_scale
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    crop = (
        max(0, int(round(cx - bw / 2 - margin))),
        max(0, int(round(cy - bh / 2 - margin))),
        min(w, int(round(cx + bw / 2 + margin))),
        min(h, int(round(cy + bh / 2 + margin))),
    )
    roi = img.crop(crop)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    roi.save(out_path)
    return out_path


def build_rows(root, use_roi=True, roi_dir=None):
    root = Path(root)
    data_dir = root / "data"
    xlsx = data_dir / "data.xlsx"
    image_dir = data_dir / "原片"
    print("image_dir =", image_dir)
    ann_dir = find_annotation_dir(image_dir)
    print("ann_dir =", ann_dir)

    if not xlsx.exists():
        raise FileNotFoundError(xlsx)

    roi_dir = Path(roi_dir) if roi_dir is not None else root / "data" / "roi"

    # Use dtype={"ID": str} to avoid losing leading zeros in Excel IDs.
    table = pd.read_excel(xlsx, dtype={"ID": str})
    table["_source_row"] = np.arange(len(table)) + 2  # Excel row number, assuming row 1 is header.

    fracture_col = next((c for c in table.columns if str(c).strip().lower() == "fracture"), None)
    if fracture_col is None:
        raise KeyError("data.xlsx 中找不到 fracture 列")

    rows = []
    skipped = []

    for _, record in table.iterrows():
        raw_id = str(record["ID"]).strip()
        if raw_id.lower() in {"nan", ""}:
            skipped.append({
                "source_row": int(record["_source_row"]),
                "raw_id": raw_id,
                "id": np.nan,
                "reason": "missing_id",
            })
            continue

        # Numeric ID used in output; raw_id is also preserved for traceability.
        try:
            image_id = int(float(raw_id))
        except Exception:
            image_id = raw_id

        image_path = find_file_by_stem(image_dir, raw_id, IMAGE_SUFFIXES)
        json_path = find_file_by_stem(ann_dir, raw_id, JSON_SUFFIXES)

        fracture = int(record[fracture_col])
        split = "external" if str(record.get("data", "")).strip().lower() == "external" else "internal"

        common_meta = {
            "source_row": int(record["_source_row"]),
            "raw_id": raw_id,
            "id": image_id,
            "split": split,
            "fracture": fracture,
        }

        if image_path is None:
            skipped.append({
                **common_meta,
                "reason": "missing_image",
                "expected_path": str(image_dir / f"{raw_id}.jpg"),
            })
            continue

        # fracture=0 does not require JSON; keep the original image.
        if fracture == 0:
            rows.append(
                {
                    **common_meta,
                    "path": str(image_path),
                    "source_image": str(image_path),
                    "md5": md5_file(image_path),
                    "garden": np.nan,
                    "injury_side": record.get("injury_side", ""),
                    "angle": np.nan,
                    "pauwels": np.nan,
                }
            )
            continue

        # fracture=1 requires JSON.
        if json_path is None:
            skipped.append({
                **common_meta,
                "reason": "missing_json",
                "expected_path": str(ann_dir / f"{raw_id}.json"),
            })
            continue

        obj = json.loads(Path(json_path).read_text(encoding="utf-8"))

        by_label = {}
        for shape in obj.get("shapes", []):
            p = point_xy(shape)
            if p is not None:
                label = shape.get("label")
                by_label.setdefault(label, []).append(p)

        missing = [k for k in REQUIRED_POINTS if k not in by_label]
        if missing:
            skipped.append({
                **common_meta,
                "reason": "missing_points",
                "detail": ",".join(missing),
                "json_path": str(json_path),
            })
            continue

        hp1, hp2, fp1, fp2 = [by_label[k][0] for k in REQUIRED_POINTS]
        angle = acute_angle_between(line_angle(hp1, hp2), line_angle(fp1, fp2))

        path = image_path
        if use_roi:
            # Save ROI using the normalized output ID to avoid mixed suffixes in the roi folder.
            path = crop_fracture_roi(image_path, fp1, fp2, roi_dir / f"{image_id}.jpg")

        rows.append(
            {
                **common_meta,
                "path": str(path),
                "source_image": str(image_path),
                "md5": md5_file(image_path),
                "garden": int(record["Garden"]) if "Garden" in record and not pd.isna(record["Garden"]) else np.nan,
                "injury_side": record.get("injury_side", ""),
                "angle": float(angle),
                "pauwels": pauwels_class(angle),
            }
        )

    data = pd.DataFrame(rows)
    skipped = pd.DataFrame(skipped)
    return data, skipped


def group_exact_split(df, test_n, seed):
    rng = random.Random(seed)
    groups = [(md5, g.index.tolist()) for md5, g in df.groupby("md5", sort=False)]
    rng.shuffle(groups)
    test = []
    deferred = []
    for _, idxs in groups:
        if len(test) + len(idxs) <= test_n:
            test.extend(idxs)
        else:
            deferred.append(idxs)
        if len(test) == test_n:
            break
    if len(test) < test_n:
        for idxs in deferred:
            if len(idxs) == 1 and len(test) < test_n:
                test.extend(idxs)
            if len(test) == test_n:
                break
    if len(test) != test_n:
        raise RuntimeError(f"Could not create exact test split: requested {test_n}, got {len(test)}")
    test = set(test)
    train_df = df.loc[[i for i in df.index if i not in test]].copy()
    test_df = df.loc[sorted(test)].copy()
    overlap = set(train_df["md5"]) & set(test_df["md5"])
    if overlap:
        raise RuntimeError(f"Image leakage across train/test groups: {len(overlap)} duplicate hashes")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_internal_by_fracture(internal_df, fracture1_n=493, fracture0_n=239, seed=42):
    internal_df = internal_df.copy().reset_index(drop=True)
    internal_df["fracture"] = pd.to_numeric(internal_df["fracture"]).astype(int)

    fracture1_df = internal_df[internal_df["fracture"] == 1]
    fracture0_df = internal_df[internal_df["fracture"] == 0]

    if len(fracture1_df) < fracture1_n:
        raise RuntimeError(
            f"Not enough fracture=1 internal samples: requested {fracture1_n}, got {len(fracture1_df)}"
        )

    if len(fracture0_df) < fracture0_n:
        raise RuntimeError(
            f"Not enough fracture=0 internal samples: requested {fracture0_n}, got {len(fracture0_df)}"
        )

    test_f1 = fracture1_df.sample(n=fracture1_n, random_state=seed)
    test_f0 = fracture0_df.sample(n=fracture0_n, random_state=seed + 1000003)

    test_idx = test_f1.index.union(test_f0.index)

    internal_test_df = internal_df.loc[test_idx].copy()
    train_df = internal_df.drop(index=test_idx).copy()

    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    internal_test_df = internal_test_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    test_counts = internal_test_df["fracture"].value_counts().to_dict()

    if test_counts.get(1, 0) != fracture1_n:
        raise RuntimeError(
            f"test_internal fracture=1 count mismatch: requested {fracture1_n}, got {test_counts.get(1, 0)}"
        )

    if test_counts.get(0, 0) != fracture0_n:
        raise RuntimeError(
            f"test_internal fracture=0 count mismatch: requested {fracture0_n}, got {test_counts.get(0, 0)}"
        )

    return train_df, internal_test_df


def save_split_summary(train_df, internal_test_df, external_df, out_dir):
    rows = []
    for split_name, df in [("train_internal", train_df), ("test_internal", internal_test_df), ("test_external", external_df)]:
        desc = df["angle"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        row = {"split": split_name, "n": len(df), **{f"angle_{k}": v for k, v in desc.items()}}
        for cls_id, count in df["pauwels"].value_counts().sort_index().items():
            row[f"pauwels_{int(cls_id)}"] = int(count)
        for garden_id, count in df["garden"].value_counts().sort_index().items():
            row[f"garden_{int(garden_id)}"] = int(count)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "split_summary.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--internal-test-fracture1-n", type=int, default=493)
    parser.add_argument("--internal-test-fracture0-n", type=int, default=239)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    seed_everything(args.seed)
    out_dir = args.root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    rows, skipped = build_rows(args.root, use_roi=True, roi_dir=None)
    rows.to_csv(out_dir / "dataset_rows.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(out_dir / "skipped_rows.csv", index=False, encoding="utf-8-sig")

    internal = rows[rows["split"] == "internal"].reset_index(drop=True)
    external = rows[rows["split"] == "external"].reset_index(drop=True)
    train_df, internal_test_df = split_internal_by_fracture(
        internal,
        fracture1_n=args.internal_test_fracture1_n,
        fracture0_n=args.internal_test_fracture0_n,
        seed=args.seed,
    )
    train_df.to_csv(out_dir / "train.csv", index=False, encoding="utf-8-sig")
    internal_test_df.to_csv(out_dir / "internal.csv", index=False, encoding="utf-8-sig")
    external.to_csv(out_dir / "external.csv", index=False, encoding="utf-8-sig")
    save_split_summary(train_df, internal_test_df, external, out_dir)
    print(f"dataset train={len(train_df)} internal_test={len(internal_test_df)} external={len(external)} skipped={len(skipped)}")


if __name__ == "__main__":
    main()
