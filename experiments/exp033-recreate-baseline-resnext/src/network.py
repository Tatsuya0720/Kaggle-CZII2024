import random

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from config import CFG
from icecream import ic


class UNet_2D(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = smp.Unet(
            # encoder_name="resnext50_32x4d",
            encoder_name=CFG.model_name,
            encoder_weights='ssl',
            # encoder_weights="imagenet",
            in_channels=1,
            classes=len(CFG.particles2cls),
        )

    def forward(self, x):
        x = self.model(x)

        return x
    
def aug(input_, gt):
    """
    画像(input_)とセグメンテーションマップ(gt)にデータ拡張を適用する関数。

    Args:
        input_ (torch.Tensor): 入力画像 (batch_size, 1, H, W)。
        gt (torch.Tensor): セグメンテーションマップ (batch_size, H, W)。

    Returns:
        torch.Tensor, torch.Tensor: データ拡張後の画像とセグメンテーションマップ。
    """
    # 確率的に適用されるデータ拡張を定義
    transforms = [
        lambda img, mask: (
            (F.hflip(img), F.hflip(mask)) if random.random() > 0.5 else (img, mask)
        ),
        lambda img, mask: (
            (F.vflip(img), F.vflip(mask)) if random.random() > 0.5 else (img, mask)
        ),
        lambda img, mask: (
            (
                F.rotate(
                    img,
                    angle=random.choice([0, 90, 180, 270]),
                    interpolation=F.InterpolationMode.NEAREST,
                ),
                F.rotate(
                    mask,
                    angle=random.choice([0, 90, 180, 270]),
                    interpolation=F.InterpolationMode.NEAREST,
                ),
            )
            if random.random() > 0.5
            else (img, mask)
        ),
    ]

    # バッチ単位で処理
    augmented_inputs, augmented_gts = [], []
    for i in range(input_.size(0)):
        img, mask = input_[i], gt[i]

        # 変換前に次元を変換
        img = img.permute(1, 2, 0)  # (1, H, W) -> (H, W, 1)
        mask = mask.unsqueeze(-1)  # (H, W) -> (H, W, 1)

        for t in transforms:
            img, mask = t(img, mask)

        # 変換後に次元を元に戻す
        img = img.permute(2, 0, 1)  # (H, W, 1) -> (1, H, W)
        mask = mask.squeeze(-1)  # (H, W, 1) -> (H, W)

        augmented_inputs.append(img)
        augmented_gts.append(mask)

    # バッチを再結合して返す
    return torch.stack(augmented_inputs), torch.stack(augmented_gts)