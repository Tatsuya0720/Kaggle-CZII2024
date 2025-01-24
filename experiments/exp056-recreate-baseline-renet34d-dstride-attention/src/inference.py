import gc
import json
import os

# import torchvision.transforms.functional as F
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import zarr
from config import CFG
from dataloader import (
    EziiDataset,
    create_dataset,
    create_segmentation_map,
    drop_padding,
    read_info_json,
    read_zarr,
    scale_coordinates,
)
from metric import (
    DiceLoss,
    SegmentationLoss,
    create_cls_pos,
    create_cls_pos_sikii,
    score,
    visualize_epoch_results,
)
from network import Unet3D
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from utils import PadToSize, save_images

padf = PadToSize(CFG.resolution)


def last_padding(tomogram, slice_size):
    # tomogram: (tensor)
    b, d, h, w = tomogram.shape
    last_padding = slice_size - d % slice_size
    if last_padding == slice_size:
        return tomogram
    else:
        return torch.cat(
            [tomogram, torch.zeros(b, last_padding, h, w).to(tomogram.device)], dim=1
        )


def preprocess_tensor(tensor):
    batch_size, depth, height, width = tensor.shape
    tensor = tensor.unsqueeze(2)  # (b, d, h, w) -> (b, d, 1, h, w)
    return tensor


def inference(model, exp_name, train=True, base_dir="../../inputs/train/"):
    dataset = EziiDataset(
        exp_names=[exp_name],
        base_dir=base_dir,
        particles_name=CFG.particles_name,
        resolution=CFG.resolution,
        zarr_type=["denoised"],
        train=train,
        slice=False,
    )
    res_array = CFG.original_img_shape[CFG.resolution]
    pred_array = np.zeros(
        (len(CFG.particles_name) + 1, res_array[0], res_array[1], res_array[2])
    )
    cnt_array = np.zeros(
        (len(CFG.particles_name) + 1, res_array[0], res_array[1], res_array[2])
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    model.eval()
    # tq = tqdm(loader)
    for data in loader:  # 実験データ1つを取り出す
        for i in range(0, data["normalized_tomogram"].shape[1], CFG.stride):
            normalized_tomogram = data["normalized_tomogram"][:, i : i + CFG.slice_]
            normalized_tomogram = last_padding(normalized_tomogram, CFG.slice_)
            normalized_tomogram = padf(normalized_tomogram)
            normalized_tomogram = preprocess_tensor(normalized_tomogram).to("cuda")
            # zarr_embedding_idx = data["zarr_type_embedding_idx"].cuda()
            with torch.no_grad():
                # with autocast():
                pred = model(normalized_tomogram, torch.tensor([0]).cuda())
            prob_pred = (
                torch.softmax(pred, dim=1).detach().cpu().numpy()
            )  # torch.Size([1, 7, 32, 320, 320])
            range_ = min(i + CFG.slice_, res_array[0])
            hw_pad_diff = prob_pred.shape[-1] - res_array[-1]

            if i >= res_array[0]:
                continue

            if range_ == res_array[0]:
                pred_array[:, i:range_] += prob_pred[
                    0, :, : res_array[0] - i, :-hw_pad_diff, :-hw_pad_diff
                ]
                cnt_array[:, i:range_] += 1
            else:
                pred_array[:, i:range_] += prob_pred[
                    0, :, :range_, :-hw_pad_diff, :-hw_pad_diff
                ]
                cnt_array[:, i:range_] += 1

            # del normalized_tomogram, zarr_embedding_idx
            # gc.collect()
            # torch.cuda.empty_cache()

        # if train:
        #     segmentation_map = data["segmentation_map"]
        # else:
        #     segmentation_map = None

        # normalized_tomogram = data["normalized_tomogram"]
    # tq.close()

    del normalized_tomogram# , zarr_embedding_idx
    gc.collect()
    torch.cuda.empty_cache()

    pred_array = pred_array / cnt_array

    return pred_array, None, None  # (7, 92, 315, 315)


def inference2pos(pred_segmask, exp_name, sikii_dict):
    import cc3d

    cls_pos = []
    Ascale_pos = []
    res2ratio = CFG.resolution2ratio

    for pred_cls in range(1, len(CFG.particles_name) + 1):
        sikii = sikii_dict[CFG.cls2particles[pred_cls]]
        # print(pred_segmask[pred_cls].shape)
        cc, P = cc3d.connected_components(pred_segmask[pred_cls] > sikii, return_N=True)
        # cc, P = cc3d.connected_components(pred_segmask == pred_cls, return_N=True)
        stats = cc3d.statistics(cc)

        for z, y, x in stats["centroids"][1:]:
            Ascale_z = z * res2ratio[CFG.resolution] / res2ratio["A"]
            Ascale_x = x * res2ratio[CFG.resolution] / res2ratio["A"]
            Ascale_y = y * res2ratio[CFG.resolution] / res2ratio["A"]

            cls_pos.append([pred_cls, z, y, x])
            Ascale_pos.append([pred_cls, Ascale_z, Ascale_y, Ascale_x])

    pred_original_df = create_df(Ascale_pos, exp_name)

    return pred_original_df

def create_gt_df(base_dir, exp_names):
    result_df = None
    particle_names = CFG.particles_name

    for exp_name in exp_names:
        for particle in particle_names:
            np_corrds = read_info_json(
                base_dir=base_dir, exp_name=exp_name, particle_name=particle
            )  # (n, 3)
            # 各行にexp_nameとparticle_name追加
            particle_df = pd.DataFrame(np_corrds, columns=["z", "y", "x"])
            particle_df["experiment"] = exp_name
            particle_df["particle_type"] = particle

            if result_df is None:
                result_df = particle_df
            else:
                result_df = pd.concat([result_df, particle_df], axis=0).reset_index(
                    drop=True
                )

    result_df = result_df.reset_index()
    result_df = result_df[["index", "experiment", "particle_type", "x", "y", "z"]]

    return result_df

def create_df(pos, exp_name):
    results = []
    for cls, z, y, x in pos:
        results.append(
            {
                "experiment": exp_name,
                "particle_type": CFG.cls2particles[cls],
                "x": x,
                "y": y,
                "z": z,
            }
        )

    results = pd.DataFrame(results)
    results = results.drop_duplicates(subset=["x", "y", "z"], keep="first")

    return pd.DataFrame(results)