import filecmp
import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def source_id(stem):
    return stem.removeprefix("synth_").split("__", 1)[0]


def pair_paths(a_dir, b_dir, suffixes=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
    def collect(root):
        root = Path(root)
        if not root.is_dir():
            raise ValueError(f"目录不存在: {root}")
        return {p.stem: p for p in root.iterdir()
                if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in suffixes}
    a, b = collect(a_dir), collect(b_dir)
    if not a:
        raise ValueError(f"A 目录没有图像: {a_dir}")
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_a or only_b:
        raise ValueError(f"A/B 目录不配对，A 独有 {len(only_a)} 个 {only_a[:5]}，B 独有 {len(only_b)} 个 {only_b[:5]}")
    return [{"stem": stem, "a": str(a[stem]), "b": str(b[stem])} for stem in sorted(a)]


def discover(config):
    rows = []
    roots = [(s, Path(config["official_root"]) / s) for s in ("normal", "abnormal")]
    roots.append(("synth", Path(config["synth_root"])))
    for source, root in roots:
        sets = {k: {p.stem for p in (root / k).iterdir() if not p.name.startswith(".")}
                for k in ("A", "B", "label", "ignore")}
        if not sets["A"] or any(v != sets["A"] for v in sets.values()):
            raise ValueError(f"四元组不完整: {root}")
        for stem in sorted(sets["A"]):
            row = {"key": f"{source}/{stem}", "source": source, "id": source_id(stem), "stem": stem}
            row.update({k: str(root / k / (stem + (".jpg" if k in ("A", "B") else ".png")))
                        for k in sets})
            rows.append(row)
    normal = {r["stem"] for r in rows if r["source"] == "normal"}
    if any(r["stem"] not in normal for r in rows if r["source"] == "synth"):
        raise ValueError("存在无法映射到官方 normal 的 synth 样本")
    return rows


def assign_folds(rows, folds, seed):
    mapping = {}
    for source in ("normal", "abnormal"):
        ids = sorted({r["id"] for r in rows if r["source"] == source})
        if len(ids) < folds or any(i in mapping for i in ids):
            raise ValueError("官方源 ID 重叠或不足以分折")
        random.Random(seed + (source == "abnormal")).shuffle(ids)
        mapping.update({i: n % folds for n, i in enumerate(ids)})
    return [dict(r, fold=mapping[r["id"]]) for r in rows]


def inspect_row(row):
    images = []
    for key in ("A", "B", "label", "ignore"):
        with Image.open(row[key]) as image:
            image.load()
            if image.mode != ("RGB" if key in ("A", "B") else "L"):
                raise ValueError(f"图像模式错误: {row[key]} {image.mode}")
            images.append((image.size, np.array(image, copy=True) if key in ("label", "ignore") else None))
    if len({size for size, _ in images}) != 1:
        raise ValueError(f"图像和掩码尺寸不一致: {row['key']}")
    label, ignore = images[2][1], images[3][1]
    if any(np.any((a != 0) & (a != 255)) for a in (label, ignore)):
        raise ValueError(f"非二值掩码: {row['key']}")
    valid = ignore <= 127
    foreground = (label > 127) & valid
    if not valid.any() or np.any((label > 127) & ~valid):
        raise ValueError(f"无有效区域或 label/ignore 重叠: {row['key']}")
    indices = np.flatnonzero(foreground)
    same = filecmp.cmp(row["A"], row["B"], shallow=False) if row["source"] == "normal" else False
    if (row["source"] == "normal" and (len(indices) or not same)) or (row["source"] != "normal" and not len(indices)):
        raise ValueError(f"来源与实际标签或 normal 配对不符: {row['key']}")
    points = []
    if len(indices):
        selected = indices[np.linspace(0, len(indices) - 1, min(128, len(indices)), dtype=np.int64)]
        points = [[int(i // label.shape[1]), int(i % label.shape[1])] for i in selected]
    return dict(row, width=images[0][0][0], height=images[0][0][1],
                positive_pixels=len(indices), target_points=points, identical=same)


def prepare(config):
    rows = assign_folds(discover(config), config["folds"], config["seed"])
    inspected = []
    with ThreadPoolExecutor(max_workers=config["prepare_workers"]) as pool:
        for index, row in enumerate(pool.map(inspect_row, rows), 1):
            inspected.append(row)
            if index % 100 == 0:
                print(json.dumps({"event": "prepare", "done": index, "total": len(rows)}), flush=True)
    payload = {"seed": config["seed"], "folds": config["folds"], "rows": inspected}
    path = Path(config["run_root"]) / "manifest.json"
    write_json(path, payload)
    summary = {"samples": dict(Counter(r["source"] for r in inspected)),
               "folds": {str(f): dict(Counter(r["source"] for r in inspected if r["fold"] == f))
                         for f in range(config["folds"])},
               "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    write_json(path.with_name("data_summary.json"), summary)
    return summary


def split_rows(rows, fold):
    if fold == -1:
        return rows, []
    train = [r for r in rows if r["fold"] != fold]
    validation = [r for r in rows if r["fold"] == fold and r["source"] != "synth"]
    if {r["id"] for r in train} & {r["id"] for r in validation}:
        raise ValueError("训练与验证存在同源泄漏")
    return train, validation


def image_tensor(image):
    return torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1).contiguous()


def load_pair(row):
    with Image.open(row["A"]) as image:
        a = image.convert("RGB")
    if row["identical"]:
        b = a
    else:
        with Image.open(row["B"]) as image:
            b = image.convert("RGB")
    return image_tensor(a), image_tensor(b)


class EpochSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.epoch = 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter((self.epoch, i) for i in range(len(self)))


class TrainingCrops(Dataset):
    def __init__(self, rows, config):
        self.config = config
        self.rows = list(rows)
        self.sampling = config.get("sampling", "source_balanced")
        if self.sampling not in ("source_balanced", "natural"):
            raise ValueError(f"未知采样策略: {self.sampling}")
        if not self.rows:
            raise ValueError("训练样本为空")
        if self.sampling == "source_balanced":
            self.pools = {s: defaultdict(list) for s in config["source_weights"]}
            for row in self.rows:
                self.pools[row["source"]][row["id"]].append(row)
            self.ids = {s: sorted(pool) for s, pool in self.pools.items()}
            if not all(self.ids.values()):
                raise ValueError("训练来源为空")

    def __len__(self):
        return self.config["steps_per_epoch"] * self.config["batch_size"]

    def select_row(self, rng):
        if self.sampling == "natural":
            return rng.choice(self.rows)
        source = rng.choices(list(self.pools), weights=list(self.config["source_weights"].values()))[0]
        return rng.choice(self.pools[source][rng.choice(self.ids[source])])

    def __getitem__(self, key):
        epoch, index = key
        rng = random.Random(self.config["seed"] + epoch * 10000019 + index)
        row = self.select_row(rng)
        source = row["source"]
        size = self.config["crop_size"]
        if min(row["width"], row["height"]) < size:
            raise ValueError(f"训练窗口大于图像: {row['key']}")
        with Image.open(row["label"]) as image:
            label = image.copy()
        with Image.open(row["ignore"]) as image:
            ignore = image.copy()
        targeted = bool(row["target_points"]) and rng.random() < 0.5
        for _ in range(32):
            if targeted:
                py, px = rng.choice(row["target_points"])
                x = rng.randint(max(0, px - size + 1), min(px, row["width"] - size))
                y = rng.randint(max(0, py - size + 1), min(py, row["height"] - size))
            else:
                x, y = rng.randint(0, row["width"] - size), rng.randint(0, row["height"] - size)
            box = (x, y, x + size, y + size)
            valid = np.array(ignore.crop(box), copy=True) <= 127
            if valid.any():
                break
        else:
            raise ValueError(f"无法取到有效裁剪: {row['key']}")
        with Image.open(row["A"]) as image:
            a = image_tensor(image.convert("RGB").crop(box))
        if row["identical"]:
            b = a
        else:
            with Image.open(row["B"]) as image:
                b = image_tensor(image.convert("RGB").crop(box))
        target = torch.from_numpy(np.array(label.crop(box), copy=True) > 127).unsqueeze(0)
        return {"a": a, "b": b, "target": target,
                "valid": torch.from_numpy(valid).unsqueeze(0), "identical": row["identical"],
                "source": source, "id": row["id"], "key": row["key"], "targeted": targeted}
