import torch
import torch.nn.functional as F


def window_starts(length, size, stride):
    if length <= 0 or size <= 0 or not 0 < stride <= size:
        raise ValueError("滑窗尺寸或步长无效")
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


@torch.inference_mode()
def sliding_predict(model, a, b, size, stride, batch_size, device, identical=False):
    if a.shape != b.shape or a.ndim != 3 or batch_size < 1:
        raise ValueError("需要同形状 CHW 配对和正整数 batch_size")
    if size % model.backbone.config.patch_size:
        raise ValueError("窗口必须是 patch_size 的整数倍")
    height, width = a.shape[-2:]
    coordinates = [(y, x) for y in window_starts(height, size, stride)
                   for x in window_starts(width, size, stride)]
    total, count = torch.zeros(height, width), torch.zeros(height, width)
    for start in range(0, len(coordinates), batch_size):
        points = coordinates[start:start + batch_size]
        batches = []
        for image in (a, b):
            crops = []
            for y, x in points:
                crop = image[:, y:y + size, x:x + size]
                crops.append(F.pad(crop, (0, size - crop.shape[-1], 0, size - crop.shape[-2])))
            batches.append(torch.stack(crops).to(device, non_blocking=True))
        same = torch.full((len(points),), identical, dtype=torch.bool, device=device)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                            enabled=torch.device(device).type == "cuda"):
            predictions = model(*batches, same).float().sigmoid().cpu()[:, 0]
        for (y, x), prediction in zip(points, predictions):
            h, w = min(size, height - y), min(size, width - x)
            total[y:y + h, x:x + w] += prediction[:h, :w]
            count[y:y + h, x:x + w] += 1
    if not (count > 0).all():
        raise RuntimeError("滑窗未完整覆盖图像")
    return total / count
