from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


def block(input_channels, output_channels):
    return nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                         nn.GroupNorm(8, output_channels), nn.GELU())


class ChangeHead(nn.Module):
    def __init__(self, channels, width=128):
        super().__init__()
        if width % 32:
            raise ValueError("head_width 必须是 32 的倍数")
        self.projections = nn.ModuleList(nn.Sequential(nn.Conv2d(channels, width, 1, bias=False),
                                                       nn.GroupNorm(8, width), nn.GELU()) for _ in range(4))
        self.differences = nn.ModuleList(block(3 * width, width) for _ in range(4))
        self.fusion = block(4 * width, width)
        self.upsampling = nn.ModuleList([block(width, width // 2), block(width // 2, width // 4),
                                         block(width // 4, width // 4)])
        self.classifier = nn.Conv2d(width // 4, 1, 1)
        nn.init.constant_(self.classifier.bias, -2.0)

    def forward(self, features_a, features_b, size):
        outputs = []
        for a, b, projection, difference in zip(features_a, features_b, self.projections, self.differences):
            count = a.shape[0]
            projected = projection(torch.cat((a, b), dim=0))
            a, b = projected[:count], projected[count:]
            outputs.append(difference(torch.cat((a, b, (a - b).abs()), dim=1)))
        x = self.fusion(torch.cat(outputs, dim=1))
        for layer in self.upsampling:
            x = layer(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return F.interpolate(self.classifier(x), size=size, mode="bilinear", align_corners=False)


class ChangeSegmenter(nn.Module):
    def __init__(self, backbone, layers=(8, 16, 24, 32), width=128):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.layers = tuple(layers)
        if len(self.layers) != 4 or min(self.layers) < 1 or max(self.layers) > backbone.config.num_hidden_layers:
            raise ValueError("需要四个有效的骨干层编号")
        self.head = ChangeHead(backbone.config.hidden_size, width)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    @classmethod
    def pretrained(cls, config, device="cuda"):
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        backbone = AutoModel.from_pretrained(Path(config["backbone"]), local_files_only=True,
                                             dtype=dtype, attn_implementation="sdpa")
        return cls(backbone, config["layers"], config["head_width"]).to(device)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, a, b, identical=None):
        if a.shape != b.shape or a.ndim != 4:
            raise ValueError("A/B 必须是同形状 NCHW 张量")
        batch, _, height, width = a.shape
        patch = self.backbone.config.patch_size
        if height % patch or width % patch:
            raise ValueError("输入尺寸必须是 patch_size 的整数倍")
        if identical is None:
            identical = torch.zeros(batch, device=a.device, dtype=torch.bool)
        different = (~identical).nonzero(as_tuple=True)[0]
        # 相同的 normal 配对复用一次骨干前向，分割头仍正常训练。
        with torch.no_grad():
            images = torch.cat((a, b[different]), dim=0).float().div(255)
            images = (images - self.mean) / self.std
            result = self.backbone(images, output_hidden_states=True, return_dict=True)
            if len(result.hidden_states) != self.backbone.config.num_hidden_layers + 1:
                raise RuntimeError("DINOv3 hidden_states 的层编号与当前接口不符")
            start = 1 + self.backbone.config.num_register_tokens
            features = [self.backbone.norm(result.hidden_states[i])[:, start:]
                        .transpose(1, 2).reshape(images.shape[0], -1, height // patch, width // patch)
                        for i in self.layers]
            indices_b = torch.arange(batch, device=a.device)
            indices_b[different] = batch + torch.arange(len(different), device=a.device)
            features_a = [f[:batch].float() for f in features]
            features_b = [f[indices_b].float() for f in features]
        return self.head(features_a, features_b, (height, width))


def masked_loss(logits, target, valid):
    logits, target, valid = logits.float(), target.float(), valid.bool()
    if logits.shape != target.shape or logits.shape != valid.shape:
        raise ValueError("logits、target、valid 的尺寸必须一致")
    if not valid.any():
        raise ValueError("裁剪中没有有效像素")
    weights = valid.float()
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).sum() / weights.sum()
    probability, truth = logits.sigmoid() * weights, target * weights
    axes = (1, 2, 3)
    positive = truth.sum(axes) > 0
    dice = logits.sum() * 0
    if positive.any():
        intersection = (probability * truth).sum(axes)
        denominator = probability.sum(axes) + truth.sum(axes)
        dice = (1 - (2 * intersection[positive] + 1) / (denominator[positive] + 1)).mean()
    return bce + dice
