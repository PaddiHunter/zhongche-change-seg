from dataclasses import asdict, dataclass

import torch


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    normal_fp: int = 0
    normal_pixels: int = 0
    normal_images: int = 0
    normal_alarm_images: int = 0

    def update(self, prediction, target, valid, normal=False):
        prediction, target, valid = prediction.bool(), target.bool(), valid.bool()
        if prediction.shape != target.shape or prediction.shape != valid.shape:
            raise ValueError("评价掩码尺寸不一致")
        if not valid.any():
            raise ValueError("评价图像没有有效像素")
        self.tp += int((prediction & target & valid).sum())
        self.fp += int((prediction & ~target & valid).sum())
        self.fn += int((~prediction & target & valid).sum())
        self.tn += int((~prediction & ~target & valid).sum())
        if normal:
            if (target & valid).any():
                raise ValueError("normal 存在有效正标签")
            count = int((prediction & valid).sum())
            self.normal_fp += count
            self.normal_pixels += int(valid.sum())
            self.normal_images += 1
            self.normal_alarm_images += int(count > 0)

    def report(self):
        def ratio(numerator, denominator):
            return numerator / denominator if denominator else 0.0
        return dict(asdict(self), iou=ratio(self.tp, self.tp + self.fp + self.fn),
                    dice=ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn),
                    precision=ratio(self.tp, self.tp + self.fp), recall=ratio(self.tp, self.tp + self.fn),
                    normal_pixel_fpr=ratio(self.normal_fp, self.normal_pixels),
                    normal_image_fpr=ratio(self.normal_alarm_images, self.normal_images))
