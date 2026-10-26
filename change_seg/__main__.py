import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import image_tensor, pair_paths, prepare
from .engine import emit, evaluate, probe, read_manifest, train
from .inference import sliding_predict
from .model import ChangeSegmenter
from .pipeline import run_pipeline


def predict_directory(model, config, a_dir, b_dir, output_dir, device):
    pairs = pair_paths(a_dir, b_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    for index, pair in enumerate(pairs, 1):
        with Image.open(pair["a"]) as image:
            a = image_tensor(image.convert("RGB"))
        with Image.open(pair["b"]) as image:
            b = image_tensor(image.convert("RGB"))
        if a.shape != b.shape:
            raise ValueError(f"A/B 尺寸不一致: {pair['stem']} 为 {a.shape[-1]}×{a.shape[-2]} 与 "
                             f"{b.shape[-1]}×{b.shape[-2]}")
        probability = sliding_predict(model, a, b, config["crop_size"], config["stride"],
                                      config["inference_batch_size"], device, torch.equal(a, b))
        mask = (probability.numpy() >= config["threshold"]).astype(np.uint8) * 255
        Image.fromarray(mask).save(output / (pair["stem"] + ".png"))
        if index % 20 == 0 or index == len(pairs):
            emit("batch_progress", done=index, total=len(pairs), seconds=round(time.monotonic() - start, 1))
    emit("batch_complete", pairs=len(pairs), output=str(output),
         seconds=round(time.monotonic() - start, 1))


def main():
    parser = argparse.ArgumentParser(description="冻结 DINOv3 配对变化分割")
    parser.add_argument("--config", default="configs/train.json")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("probe")
    training = commands.add_parser("train")
    training.add_argument("--fold", type=int, required=True)
    training.add_argument("--resume")
    training.add_argument("--epochs", type=int)
    pipeline = commands.add_parser("run")
    pipeline.add_argument("--gpus", nargs="+", type=int, required=True)
    validation = commands.add_parser("evaluate")
    validation.add_argument("--checkpoint", required=True)
    validation.add_argument("--fold", type=int, required=True)
    validation.add_argument("--output", required=True)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--a", required=True)
    prediction.add_argument("--b", required=True)
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--device", default="cuda")
    prediction.add_argument("--batch", type=int)
    directory = commands.add_parser("batch")
    directory.add_argument("--checkpoint", required=True)
    directory.add_argument("--a-dir", required=True)
    directory.add_argument("--b-dir", required=True)
    directory.add_argument("--output-dir", required=True)
    directory.add_argument("--batch", type=int)
    directory.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command in ("predict", "batch", "evaluate"):
        device = "cuda" if args.command == "evaluate" else args.device
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        if getattr(args, "batch", None) is not None:
            if args.batch < 1:
                raise ValueError("--batch 必须是正整数")
            config = dict(config, inference_batch_size=args.batch)
        model = ChangeSegmenter.pretrained(config, device)
        model.head.load_state_dict(checkpoint["head"])
        model.eval()
        if args.command == "evaluate":
            rows, digest = read_manifest(config)
            if checkpoint["manifest_sha256"] != digest or checkpoint["fold"] != args.fold:
                raise ValueError("checkpoint 与数据索引或验证 fold 不同")
            rows = [r for r in rows if r["fold"] == args.fold and r["source"] != "synth"]
            emit("evaluation_complete", **evaluate(model, rows, config, device, args.output))
        elif args.command == "batch":
            predict_directory(model, config, args.a_dir, args.b_dir, args.output_dir, device)
        else:
            with Image.open(args.a) as image:
                a = image_tensor(image.convert("RGB"))
            with Image.open(args.b) as image:
                b = image_tensor(image.convert("RGB"))
            probability = sliding_predict(model, a, b, config["crop_size"], config["stride"],
                                          config["inference_batch_size"], device, torch.equal(a, b))
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((probability.numpy() >= config["threshold"]).astype(np.uint8) * 255).save(output)
            emit("prediction_complete", output=str(output), width=a.shape[-1], height=a.shape[-2])
        return
    config = json.loads(Path(args.config).read_text())
    if args.command == "prepare":
        emit("prepare_complete", **prepare(config))
    elif args.command == "probe":
        emit("probe_complete", **probe(config))
    elif args.command == "train":
        train(config, args.fold, resume=args.resume, epochs=args.epochs)
    elif args.command == "run":
        run_pipeline(config, args.config, args.gpus)


if __name__ == "__main__":
    main()
