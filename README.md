# 轨道车辆配对图像变化分割

给定同一部位的两张图像 A（正常）与 B（待检测），输出像素级变化掩码。用于地铁车厢部件遮挡、异物等异常的成对图像比对。

## 方法

DINOv3 ViT-H+/16 作为共享骨干，固定为 eval 模式并关闭全部梯度（840,592,640 参数）。A、B 分别经过骨干，取第 8/16/24/32 层 patch 特征，排除 CLS 与 register tokens；每层对 A/B 共享一个降维投影，把 A、B 与二者的绝对差拼接到一起，经轻量卷积融合和逐级上采样输出单通道变化 logits。可训练头 3,118,625 参数。A、B 完全相同时直接复用骨干特征，跳过第二次前向。

## 数据

数据不入库，需要自行放到下面的位置，路径在 `configs/train.json` 里修改。

| 配置键 | 内容 |
| --- | --- |
| `official_root` | 官方标注数据，2400 对：normal 2280 对（285 个源 ID）、abnormal 120 对（15 个源 ID），含 8 个旋转翻转变体 |
| `synth_root` | 合成数据，1538 对（284 个来源 ID），只进入训练，不进入验证 |
| `backbone` | DINOv3 ViT-H+/16 预训练权重目录（`config.json` + 权重文件） |

标注数据的目录约定，同一类别下四个子目录中主名相同的文件构成一个样本：

```text
official_root/
├── abnormal/{A,B,label,ignore}/
└── normal/{A,B,label,ignore}/
```

A、B 为 JPEG，label 与 ignore 为 PNG：label 中 255 表示目标变化区域，ignore 中 255 表示无效区域，训练和评价都排除 `ignore > 127` 的像素。

按官方源 ID 分组做固定 5 折，每折验证 456 对 normal 与 24 对 abnormal，共 480 对；合成样本跟随其来源 normal ID 的归属。样本索引由 `python -m change_seg prepare` 生成到配置里的 `run_root`。

## 训练配置

`configs/train_natural.json` 是本文结果使用的配置：随机种子 42，5 折，每折 20 epoch、每 epoch 200 step、batch size 4，共 16000 个裁剪；`sampling="natural"`，在训练分区内按配对样本等概率有放回抽样，不加来源或 ID 权重。训练窗口 768×768，异常裁剪约一半包含有效目标。AdamW，学习率 0.001，weight decay 0.0001，100 step warmup 加 cosine decay，bf16 与 PyTorch SDPA。

`configs/train.json` 是按来源加权采样的对照配置（真实异常:合成:normal = 2:1:1）。

损失为 ignore≤127 像素上的 BCE，加上仅在含有效前景的裁剪上计算的 Dice。第 1/5/10/15/20 epoch 做整图验证，阈值固定 0.5，按全局前景 IoU 保存 best，同时记录 Dice、Precision、Recall 和 normal 图像的误报。

推理使用窗口 768、步长 576、窗口批大小 16，重叠区域做概率平均后恢复到原图分辨率，阈值 0.5 输出二值掩码。

## 结果

五折开发验证，阈值 0.5：

| Fold | 选中轮次 | IoU | 像素 F1 | Precision | Recall | normal 误报 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 10 | 0.856108 | 0.922477 | 0.893914 | 0.952924 | 0/456 |
| 1 | 15 | 0.818204 | 0.900013 | 0.930793 | 0.871204 | 0/456 |
| 2 | 20 | 0.247003 | 0.396155 | 0.840268 | 0.259173 | 3/456，187 px |
| 3 | 5 | 0.664191 | 0.798215 | 0.757796 | 0.843188 | 0/456 |
| 4 | 15 | 0.709556 | 0.830106 | 0.760438 | 0.913826 | 0/456 |
| 汇总 | — | 0.469272 | 0.638782 | 0.856429 | 0.509341 | 3/2280，187 px |

池化像素指标按面积加权，异常源 ID 6 的掩码占全部异常前景的 53.29%，汇总值由它主导。剔除 ID 6 后：

| 口径 | IoU | 像素 F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| 含 ID 6 | 0.469272 | 0.638782 | 0.856429 | 0.509341 |
| 剔除 ID 6 | 0.772156 | 0.871431 | 0.848687 | 0.895428 |

全量模型（全部 3938 对训练）在训练集内评估：

| 数据 | 对数 | IoU | 像素 F1 | Precision | Recall | normal 误报 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 官方 2400 对 | 2400 | 0.902414 | 0.948704 | 0.936138 | 0.961612 | 2/2280，47 px |
| 合成 1538 对 | 1538 | 0.911609 | 0.953761 | 0.951328 | 0.956207 | — |

像素 F1 = Dice = `2TP/(2TP+FP+FN)`，排除 ignore 像素，阈值固定 0.5。

## 环境

Python 3.12，依赖见 `requirements.txt`（torch 2.5.1、transformers 4.57.3、numpy 2.2.6、Pillow 12.3.0）。训练与推理需要 CUDA；`predict` 支持显式 `--device cpu`，默认要求 CUDA，不自动回退。

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install pytest   # 仅跑测试需要
python -m pytest -q
```

## 命令

```bash
# 生成并校验样本索引
python -u -m change_seg prepare

# 真实预训练权重的前后向与整图滑窗检查
CUDA_VISIBLE_DEVICES=0 python -u -m change_seg probe

# 单折训练，可指定 checkpoint 恢复
CUDA_VISIBLE_DEVICES=0 python -u -m change_seg train --fold 0

# 五折与全量重训
python -u -m change_seg run --gpus 0 1 2 3

# 用指定 checkpoint 评估某一折
CUDA_VISIBLE_DEVICES=0 python -m change_seg evaluate --checkpoint /path/best.pt --fold 0 --output validation.jsonl

# 单对推理
python -m change_seg predict --checkpoint /path/model.pt --a A.jpg --b B.jpg --output mask.png

# 按文件主名配对两个目录批量推理
python -m change_seg predict --checkpoint /path/model.pt --batch --a-dir A目录 --b-dir B目录 --output-dir 输出目录
```

`evaluate` 校验 checkpoint 里的 manifest SHA256、fold 与当前索引一致；`train --resume` 同样校验配置、manifest 与 fold。`--config` 默认 `configs/train.json`，跑本文配置时显式传：`python -m change_seg --config configs/train_natural.json prepare`。prepare、probe、train、run 读 `--config`；predict、batch、evaluate 的配置存在 checkpoint 里。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `change_seg/data.py` | 数据发现、校验、配对、裁剪与 5 折划分 |
| `change_seg/model.py` | 冻结骨干 + 变化分割头、masked loss |
| `change_seg/inference.py` | 滑窗推理与重叠平均 |
| `change_seg/metrics.py` | IoU、Dice、Precision、Recall 与误报统计 |
| `change_seg/engine.py` | 训练、验证、评估、probe、checkpoint 读写 |
| `change_seg/pipeline.py` | 多 GPU 五折与全量重训调度 |
| `change_seg/__main__.py` | 命令行入口 |
| `configs/` | 两份训练配置 |
| `tests/test_core.py` | 核心测试，用合成小图，可以直接跑 |

## 限制

- 五折验证集参与选权重，属于开发交叉验证；全量模型使用全部 3938 对训练，没有独立留出成绩。
- 池化像素指标被 ID 6 的面积主导，逐 ID 与宏平均结果不在本仓库。
- normal 验证对的 A/B 其实是同一张图，不能反映真实重复拍摄差异下的稳健性。
- 数据集来自“斫轮九章算法大赛”，不在本仓库内分发，使用请遵守赛事条款。
