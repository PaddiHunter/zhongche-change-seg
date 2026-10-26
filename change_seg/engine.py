import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import EpochSampler, TrainingCrops, load_pair, split_rows, write_json
from .inference import sliding_predict
from .metrics import Metrics
from .model import ChangeSegmenter, masked_loss


def emit(event, **values):
    print(json.dumps(dict(event=event, time=time.strftime("%Y-%m-%dT%H:%M:%S"), **values), ensure_ascii=False), flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)


def read_manifest(config):
    path = Path(config["run_root"]) / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["seed"] != config["seed"] or manifest["folds"] != config["folds"]:
        raise ValueError("manifest 与配置中的分折约定不一致")
    return manifest["rows"], hashlib.sha256(path.read_bytes()).hexdigest()


def make_loader(rows, config):
    dataset = TrainingCrops(rows, config)
    sampler = EpochSampler(dataset)
    options = {"num_workers": config["workers"], "pin_memory": True,
               "batch_size": config["batch_size"], "sampler": sampler}
    if config["workers"]:
        options.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **options), sampler


def device_batch(batch, device):
    return {k: batch[k].to(device, non_blocking=True) for k in ("a", "b", "target", "valid", "identical")}


def evaluate(model, rows, config, device, output):
    if not rows:
        raise ValueError("验证集为空")
    model.eval()
    aggregate = Metrics()
    start = time.monotonic()
    with Path(output).open("w") as stream:
        for index, row in enumerate(rows, 1):
            a, b = load_pair(row)
            probability = sliding_predict(model, a, b, config["crop_size"], config["stride"],
                                          config["inference_batch_size"], device, row["identical"])
            with Image.open(row["label"]) as image:
                target = torch.from_numpy(np.array(image, copy=True) > 127)
            with Image.open(row["ignore"]) as image:
                valid = torch.from_numpy(np.array(image, copy=True) <= 127)
            current = Metrics()
            current.update(probability >= config["threshold"], target, valid, row["source"] == "normal")
            for field in fields(Metrics):
                setattr(aggregate, field.name, getattr(aggregate, field.name) + getattr(current, field.name))
            stream.write(json.dumps(dict(key=row["key"], **current.report())) + "\n")
            if index % 20 == 0 or index == len(rows):
                stream.flush()
                emit("validation", done=index, total=len(rows), seconds=round(time.monotonic() - start, 1),
                     **aggregate.report())
    return aggregate.report()


def save_checkpoint(path, model, config, epoch, manifest_hash, optimizer=None, scheduler=None, metrics=None, fold=None):
    state = {"head": {k: v.detach().cpu() for k, v in model.head.state_dict().items()},
             "config": config, "epoch": epoch, "fold": fold, "manifest_sha256": manifest_hash, "metrics": metrics}
    if optimizer is not None:
        state.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
    path = Path(path)
    temporary = path.with_suffix(".partial")
    torch.save(state, temporary)
    temporary.replace(path)


def train(config, fold, device="cuda", resume=None, epochs=None):
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("训练需要明确可用的 CUDA 设备")
    rows, manifest_hash = read_manifest(config)
    if fold not in range(config["folds"]) and fold != -1:
        raise ValueError("无效 fold")
    train_rows, validation_rows = split_rows(rows, fold)
    if fold != -1 and not validation_rows:
        raise ValueError("交叉验证 fold 没有官方验证样本")
    output = Path(config["run_root"]) / ("full" if fold == -1 else f"fold_{fold}")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and resume is None:
        raise FileExistsError(f"训练输出已存在，须显式 --resume: {output}")
    local_config = dict(config, seed=config["seed"] + (fold + 1) * 1000)
    seed_all(local_config["seed"])
    model = ChangeSegmenter.pretrained(config, device)
    loader, sampler = make_loader(train_rows, local_config)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    total_steps = config["epochs"] * config["steps_per_epoch"]
    warmup = config["warmup_steps"]

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    start_epoch, best_iou = 0, -1.0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        if checkpoint["manifest_sha256"] != manifest_hash or checkpoint["config"] != config or checkpoint["fold"] != fold:
            raise ValueError("恢复点的数据、训练配置或 fold 不同")
        model.head.load_state_dict(checkpoint["head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["rng"].cpu())
        torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu())
        start_epoch = checkpoint["epoch"]
        if (output / "best.json").exists():
            best_iou = json.loads((output / "best.json").read_text())["metrics"]["iou"]
    write_json(output / "config.json", config)
    training_sources = {source: sum(r["source"] == source for r in train_rows)
                        for source in sorted({r["source"] for r in train_rows})}
    write_json(output / "split.json", {"train_ids": sorted({r["id"] for r in train_rows}),
                                       "validation_ids": sorted({r["id"] for r in validation_rows}),
                                       "training_samples": len(train_rows), "validation_samples": len(validation_rows),
                                       "training_sources": training_sources,
                                       "sampling": config.get("sampling", "source_balanced"),
                                       "manifest_sha256": manifest_hash})
    emit("train_start", fold=fold, device=str(device), gpu=torch.cuda.get_device_name(),
         train_samples=len(train_rows), validation_samples=len(validation_rows),
         backbone_parameters=sum(p.numel() for p in model.backbone.parameters()),
         trainable_parameters=sum(p.numel() for p in model.head.parameters()),
         crop_size=config["crop_size"], batch_size=config["batch_size"],
         sampling=config.get("sampling", "source_balanced"), training_sources=training_sources)
    max_epochs = config["epochs"] if epochs is None else epochs
    if not 0 < max_epochs <= config["epochs"] or start_epoch >= max_epochs:
        raise ValueError("训练轮数无效，或恢复点已完成")
    for epoch in range(start_epoch + 1, max_epochs + 1):
        sampler.epoch = epoch
        model.train()
        running_loss = 0.0
        epoch_start = time.monotonic()
        sources = {key: 0 for key in training_sources}
        for step, batch in enumerate(loader, 1):
            for source in batch["source"]:
                sources[source] += 1
            batch = device_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch["a"], batch["b"], batch["identical"])
                loss = masked_loss(logits, batch["target"], batch["valid"])
            if not torch.isfinite(loss):
                raise FloatingPointError("训练损失非有限值")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.detach())
            if step == 1 or step % 20 == 0:
                emit("train_step", fold=fold, epoch=epoch, step=step, steps=len(loader),
                     loss=running_loss / step, lr=optimizer.param_groups[0]["lr"],
                     seconds=round(time.monotonic() - epoch_start, 1))
        if any(p.grad is not None or p.requires_grad for p in model.backbone.parameters()) or model.backbone.training:
            raise RuntimeError("骨干冻结约束被破坏")
        metrics = None
        if validation_rows and (epoch == 1 or epoch % config["eval_every"] == 0 or epoch == max_epochs):
            metrics = evaluate(model, validation_rows, config, device, output / f"validation_{epoch:03d}.jsonl")
            if metrics["iou"] > best_iou:
                best_iou = metrics["iou"]
                save_checkpoint(output / "best.pt", model, config, epoch, manifest_hash, metrics=metrics, fold=fold)
                write_json(output / "best.json", {"epoch": epoch, "metrics": metrics})
        save_checkpoint(output / "last.pt", model, config, epoch, manifest_hash, optimizer, scheduler, metrics, fold=fold)
        record = {"fold": fold, "epoch": epoch, "loss": running_loss / len(loader), "sources": sources,
                  "seconds": time.monotonic() - epoch_start, "metrics": metrics}
        with (output / "epochs.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        write_json(output / "progress.json", record)
        emit("epoch_complete", **record)
    if fold == -1:
        save_checkpoint(output / "final.pt", model, config, max_epochs, manifest_hash, fold=fold)
    emit("train_complete", fold=fold, epochs=max_epochs, best_iou=best_iou if validation_rows else None)


def probe(config, device="cuda"):
    rows, _ = read_manifest(config)
    seed_all(config["seed"])
    model = ChangeSegmenter.pretrained(config, device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=config["learning_rate"])
    original = next(model.backbone.parameters()).flatten()[:4096].detach().clone()
    train_rows, _ = split_rows(rows, 0)
    output = Path(config["run_root"]) / "probe"
    output.mkdir(exist_ok=True)
    results = []
    for size in (512, 768, 1024):
        candidate = dict(config, crop_size=size, workers=0)
        dataset = TrainingCrops(train_rows, candidate)
        batch = torch.utils.data.default_collate([dataset[(0, i)] for i in range(config["batch_size"])])
        batch = device_batch(batch, device)
        for step in range(5):
            optimizer.zero_grad(set_to_none=True)
            if step == 2:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.monotonic()
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch["a"], batch["b"], batch["identical"])
                loss = masked_loss(logits, batch["target"], batch["valid"])
            loss.backward()
            optimizer.step()
            if not torch.isfinite(loss):
                raise FloatingPointError("GPU 检查损失非有限值")
        torch.cuda.synchronize()
        result = {"crop_size": size, "batch_size": config["batch_size"],
                  "seconds_per_step": (time.monotonic() - start) / 3,
                  "peak_gib": torch.cuda.max_memory_allocated() / 2**30, "loss": float(loss.detach())}
        results.append(result)
        emit("gpu_probe", **result)
        del logits, loss, batch
    if not torch.equal(original, next(model.backbone.parameters()).flatten()[:4096]) or any(
            p.grad is not None or p.requires_grad for p in model.backbone.parameters()):
        raise RuntimeError("实际预训练骨干未保持冻结")
    save_checkpoint(output / "head.pt", model, config, 0, "probe")
    model.head.load_state_dict(torch.load(output / "head.pt", map_location=device, weights_only=True)["head"])
    # 用真实整图验证滑窗拼接、忽略区域与指标链路；此结果不作为模型精度。
    smoke_rows = [next(r for r in rows if r["source"] == source) for source in ("abnormal", "normal")]
    inference_start = time.monotonic()
    metrics = evaluate(model, smoke_rows, config, device, output / "smoke_validation.jsonl")
    report = {"gpu": torch.cuda.get_device_name(), "measurements": results, "frozen_backbone": True,
              "warmup_steps_per_size": 2, "measured_steps_per_size": 3,
              "inference_batch_size": config["inference_batch_size"],
              "smoke_inference_seconds": time.monotonic() - inference_start,
              "smoke_images": [r["key"] for r in smoke_rows], "smoke_metrics_not_accuracy": metrics,
              "head_parameters": sum(p.numel() for p in model.head.parameters())}
    write_json(output / "report.json", report)
    return report


def summarize_cv(config):
    aggregate = Metrics()
    folds = []
    for fold in range(config["folds"]):
        result = json.loads((Path(config["run_root"]) / f"fold_{fold}" / "best.json").read_text())
        folds.append(dict(fold=fold, **result))
        for field in fields(Metrics):
            setattr(aggregate, field.name, getattr(aggregate, field.name) + result["metrics"][field.name])
    summary = {"folds": folds, "global": aggregate.report(),
               "fold_iou_mean": float(np.mean([r["metrics"]["iou"] for r in folds])),
               "fold_iou_std": float(np.std([r["metrics"]["iou"] for r in folds])),
               "full_epochs": max(1, int(np.median([r["epoch"] for r in folds]) + 0.5)),
               "note": "验证集用于选择 checkpoint；结果为开发交叉验证指标，非独立测试集成绩。"}
    write_json(Path(config["run_root"]) / "cv_summary.json", summary)
    return summary


def package(config):
    root = Path(config["run_root"])
    destination = root / "deploy"
    destination.mkdir(exist_ok=False)
    shutil.copytree(config["backbone"], destination / "backbone")
    shutil.copytree("change_seg", destination / "change_seg", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2("requirements.txt", destination / "requirements.txt")
    checkpoint = torch.load(root / "full" / "final.pt", map_location="cpu", weights_only=True)
    checkpoint["config"] = dict(checkpoint["config"], backbone="backbone")
    torch.save(checkpoint, destination / "model.pt")
    total = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    if total > 12_000_000_000:
        raise RuntimeError(f"交付目录超过 12GB: {total}")
    write_json(root / "package.json", {"path": str(destination), "bytes": total})
    return {"path": str(destination), "bytes": total}
