import json
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from change_seg.data import TrainingCrops, assign_folds, inspect_row, source_id, split_rows
from change_seg.inference import sliding_predict, window_starts
from change_seg.metrics import Metrics
from change_seg.model import ChangeSegmenter, masked_loss


@pytest.fixture
def tiny_model():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    config = DINOv3ViTConfig(hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
                            intermediate_size=64, patch_size=16, image_size=32,
                            num_register_tokens=4, use_gated_mlp=True)
    return ChangeSegmenter(DINOv3ViTModel(config), (1, 2, 3, 4), 32)


@pytest.fixture
def rows(tmp_path):
    rows = []
    for index, source in enumerate(("abnormal", "synth", "normal")):
        root = tmp_path / source
        root.mkdir()
        stem = f"{index}__r000_f0"
        a = np.full((64, 80, 3), 90, dtype=np.uint8)
        b = a.copy()
        label, ignore = np.zeros((64, 80), np.uint8), np.zeros((64, 80), np.uint8)
        ignore[:2] = 255
        if source != "normal":
            b[20:30, 30:40] = 210
            label[20:30, 30:40] = 255
        row = {"source": source, "id": str(index), "key": source + "/" + stem, "stem": stem, "fold": index}
        for key, array in (("A", a), ("B", b), ("label", label), ("ignore", ignore)):
            row[key] = str(root / (key + (".jpg" if key in ("A", "B") else ".png")))
            Image.fromarray(array).save(row[key])
        if source == "normal":
            shutil.copy2(row["A"], row["B"])
        rows.append(inspect_row(row))
    return rows


def test_source_ids_and_grouped_folds():
    assert source_id("synth_100__r090_f1") == "100"
    rows = []
    for source, first, count in (("normal", 100, 10), ("abnormal", 1, 5)):
        for i in range(first, first + count):
            for variant in range(8):
                rows.append(dict(source=source, id=str(i), stem=f"{i}__r{variant}"))
            if source == "normal":
                rows.append(dict(source="synth", id=str(i), stem=f"{i}__r000"))
    folded = assign_folds(rows, 5, 42)
    assert folded == assign_folds(rows, 5, 42)
    for fold in range(5):
        train, validation = split_rows(folded, fold)
        assert not {r["id"] for r in train} & {r["id"] for r in validation}
        assert all(r["source"] != "synth" for r in validation)
        assert len({r["id"] for r in validation if r["source"] == "abnormal"}) == 1


def test_crops_match_and_keep_normal_unchanged(rows):
    config = dict(seed=42, crop_size=32, steps_per_epoch=200, batch_size=4,
                  source_weights={"abnormal": 2, "synth": 1, "normal": 1})
    dataset = TrainingCrops(rows, config)
    counts = {s: 0 for s in config["source_weights"]}
    targeted = 0
    for index in range(400):
        sample = dataset[(1, index)]
        counts[sample["source"]] += 1
        assert sample["a"].shape == sample["b"].shape == (3, 32, 32)
        assert sample["target"].shape == sample["valid"].shape == (1, 32, 32)
        assert sample["valid"].any()
        if sample["targeted"]:
            targeted += 1
            assert (sample["target"] & sample["valid"]).any()
        if sample["source"] == "normal":
            assert torch.equal(sample["a"], sample["b"])
            assert not sample["target"].any()
    assert 160 < counts["abnormal"] < 240
    assert 65 < counts["synth"] < 135
    assert 65 < counts["normal"] < 135
    assert 100 < targeted < 200
    assert torch.equal(dataset[(1, 10)]["a"], dataset[(1, 10)]["a"])


def test_masked_loss_has_no_ignore_gradient():
    logits = torch.zeros(2, 1, 4, 4, requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 0, 1, 1] = 1
    valid = torch.ones_like(logits, dtype=torch.bool)
    valid[:, :, 0] = False
    loss = masked_loss(logits, target, valid)
    loss.backward()
    assert torch.isfinite(loss)
    assert not logits.grad[:, :, 0].any()
    assert logits.grad[0, 0, 1, 1] < 0
    assert logits.grad[1, 0, 1, 1] > 0
    with pytest.raises(ValueError):
        masked_loss(logits, target, torch.zeros_like(valid))


def test_normal_only_loss_is_bce():
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    loss = masked_loss(logits, torch.zeros_like(logits), torch.ones_like(logits, dtype=torch.bool))
    assert float(loss.detach()) == pytest.approx(np.log(2))
    loss.backward()
    assert (logits.grad > 0).all()


def test_real_transformer_head_updates_and_backbone_stays_frozen(tiny_model):
    model = tiny_model.train()
    original = {k: value.clone() for k, value in model.backbone.state_dict().items()}
    original_head = {k: value.clone() for k, value in model.head.state_dict().items()}
    a = torch.randint(0, 256, (2, 3, 64, 80), dtype=torch.uint8)
    b = a.clone()
    b[0, :, 16:32, 16:32] = 255
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.001)
    logits = model(a, b, torch.tensor([False, True]))
    assert logits.shape == (2, 1, 64, 80)
    target = torch.zeros_like(logits)
    target[0, :, 16:32, 16:32] = 1
    masked_loss(logits, target, torch.ones_like(logits, dtype=torch.bool)).backward()
    optimizer.step()
    assert not model.backbone.training
    assert all(not p.requires_grad and p.grad is None for p in model.backbone.parameters())
    assert all(torch.equal(value, original[k]) for k, value in model.backbone.state_dict().items())
    assert any(not torch.equal(value, original_head[k]) for k, value in model.head.state_dict().items())


def test_sliding_actual_transformer_covers_edges_and_padding(tiny_model):
    model = tiny_model.eval()
    a = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
    with torch.inference_mode():
        expected = model(a[None], a[None], torch.tensor([True])).sigmoid()[0, 0]
    actual = sliding_predict(model, a, a, 32, 24, 2, "cpu", True)
    assert torch.allclose(actual, expected, atol=1e-6)
    a = torch.randint(0, 256, (3, 51, 70), dtype=torch.uint8)
    output = sliding_predict(model, a, a, 32, 24, 2, "cpu", True)
    assert output.shape == (51, 70) and torch.isfinite(output).all()
    small = a[:, :19, :21]
    assert sliding_predict(model, small, small, 32, 24, 2, "cpu", True).shape == (19, 21)
    assert window_starts(70, 32, 24) == [0, 24, 38]
    with pytest.raises(ValueError):
        window_starts(70, 32, 33)


def test_global_iou_counts_normal_false_positives_and_ignores():
    metrics = Metrics()
    metrics.update(torch.tensor([True, True, False]), torch.tensor([True, False, False]),
                   torch.tensor([True, False, True]))
    metrics.update(torch.tensor([True, False]), torch.tensor([False, False]),
                   torch.tensor([True, True]), normal=True)
    result = metrics.report()
    assert result["iou"] == 0.5
    assert result["dice"] == pytest.approx(2 / 3)
    assert result["normal_pixel_fpr"] == 0.5
    assert result["normal_image_fpr"] == 1


def test_checkpoint_roundtrip_is_head_only_and_records_fold(tiny_model, tmp_path):
    from change_seg.engine import save_checkpoint

    path = tmp_path / "head.pt"
    save_checkpoint(path, tiny_model, {"seed": 42}, 3, "digest", fold=2)
    restored = torch.load(path, map_location="cpu", weights_only=True)
    assert restored["fold"] == 2 and restored["epoch"] == 3
    assert restored["manifest_sha256"] == "digest"
    assert "backbone" not in restored
    assert all(torch.equal(value, restored["head"][key])
               for key, value in tiny_model.head.state_dict().items())


def test_project_folds_on_actual_file_names():
    from change_seg.data import discover

    config = json.loads(Path("configs/train.json").read_text())
    if not Path(config["official_root"]).is_dir() or not Path(config["synth_root"]).is_dir():
        pytest.skip("需要真实数据集，仓库内不含 data/")
    rows = assign_folds(discover(config), config["folds"], config["seed"])
    assert len(rows) == 3938
    for fold in range(5):
        train, validation = split_rows(rows, fold)
        assert len(validation) == 480
        assert sum(r["source"] == "abnormal" for r in validation) == 24
        assert len({r["id"] for r in validation if r["source"] == "abnormal"}) == 3
        assert len({r["id"] for r in train if r["source"] == "abnormal"}) == 12
        assert not {r["id"] for r in train} & {r["id"] for r in validation}


def test_natural_sampling_is_uniform_over_actual_pairs(tmp_path, rows):
    from collections import Counter

    expanded = []
    for template, variants_per_id in zip(rows, ((1,), (1, 4), (2, 7))):
        for group, variants in enumerate(variants_per_id):
            identity = str(1 if template["source"] == "abnormal" else 100 + group)
            for variant in range(variants):
                stem = f"{identity}__variant_{variant}"
                root = tmp_path / "natural" / template["source"] / stem
                root.mkdir(parents=True)
                row = dict(source=template["source"], id=identity, stem=stem,
                           key=f"{template['source']}/{stem}", fold=0)
                for key in ("A", "B", "label", "ignore"):
                    row[key] = str(root / (key + (".jpg" if key in ("A", "B") else ".png")))
                    shutil.copy2(template[key], row[key])
                expanded.append(inspect_row(row))
    config = dict(seed=42, crop_size=32, steps_per_epoch=200, batch_size=4, sampling="natural")
    dataset = TrainingCrops(expanded, config)
    assert len(dataset) == 800 and len(expanded) == 15
    draws = [dataset.select_row(random.Random(42 + i)) for i in range(6000)]
    key_counts = Counter(r["key"] for r in draws)
    assert set(key_counts) == {r["key"] for r in expanded}
    assert all(abs(n / 6000 - 1 / 15) < 0.015 for n in key_counts.values())
    source_counts = Counter(r["source"] for r in draws)
    assert abs(source_counts["abnormal"] / 6000 - 1 / 15) < 0.025
    assert abs(source_counts["synth"] / 6000 - 5 / 15) < 0.025
    assert abs(source_counts["normal"] / 6000 - 9 / 15) < 0.025
    for index in range(40):
        expected = random.Random(42 + 10000019 + index).choice(expanded)
        sample = dataset[(1, index)]
        assert sample["key"] == expected["key"]
        assert sample["source"] == expected["source"]
        assert sample["a"].shape == sample["b"].shape == (3, 32, 32)
        assert sample["valid"].any()
        if sample["targeted"]:
            assert (sample["target"] & sample["valid"]).any()
        if sample["source"] == "normal":
            assert torch.equal(sample["a"], sample["b"])
            assert not sample["target"].any()
    assert torch.equal(dataset[(1, 4)]["a"], dataset[(1, 4)]["a"])


def test_source_balanced_selection_preserves_existing_random_sequence(rows):
    config = dict(seed=42, crop_size=32, steps_per_epoch=200, batch_size=4,
                  source_weights={"abnormal": 2, "synth": 1, "normal": 1})
    dataset = TrainingCrops(rows, config)
    assert dataset.sampling == "source_balanced"
    for index in range(100):
        seed = 42 + 10000019 + index
        rng = random.Random(seed)
        source = rng.choices(list(dataset.pools), weights=list(config["source_weights"].values()))[0]
        expected = rng.choice(dataset.pools[source][rng.choice(dataset.ids[source])])
        assert dataset.select_row(random.Random(seed))["key"] == expected["key"]


def test_natural_sampling_configuration_keeps_comparison_budget():
    baseline = json.loads(Path("configs/train.json").read_text())
    natural = json.loads(Path("configs/train_natural.json").read_text())
    assert natural["sampling"] == "natural" and "source_weights" not in natural
    changed = {key for key in baseline.keys() | natural.keys() if baseline.get(key) != natural.get(key)}
    assert changed == {"sampling", "source_weights", "run_root"}
    assert natural["epochs"] * natural["steps_per_epoch"] * natural["batch_size"] == 16000


def test_pair_paths_requires_matching_stems(tmp_path):
    from change_seg.data import pair_paths

    a, b, empty = tmp_path / "A", tmp_path / "B", tmp_path / "empty"
    for root in (a, b, empty):
        root.mkdir()
    image = np.full((8, 8, 3), 90, dtype=np.uint8)
    Image.fromarray(image).save(a / "1__r000_f0.jpg")
    Image.fromarray(image).save(b / "1__r000_f0.jpg")
    Image.fromarray(image).save(b / "2__r000_f0.jpg")
    with pytest.raises(ValueError, match="不配对"):
        pair_paths(a, b)
    pairs = pair_paths(a, a)
    assert [p["stem"] for p in pairs] == ["1__r000_f0"]
    assert pairs[0] == {"stem": "1__r000_f0", "a": str(a / "1__r000_f0.jpg"),
                        "b": str(a / "1__r000_f0.jpg")}
    with pytest.raises(ValueError, match="目录不存在"):
        pair_paths(tmp_path / "missing", b)
    with pytest.raises(ValueError, match="没有图像"):
        pair_paths(empty, b)


def test_sampling_rejects_unknown_strategy_and_empty_dataset(rows):
    config = dict(seed=42, crop_size=32, steps_per_epoch=200, batch_size=4, sampling="unknown")
    with pytest.raises(ValueError, match="未知采样策略"):
        TrainingCrops(rows, config)
    with pytest.raises(ValueError, match="训练样本为空"):
        TrainingCrops([], dict(config, sampling="natural"))
