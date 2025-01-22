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
from tqdm import tqdm
from utils import PadToSize, save_images

padf = PadToSize(CFG.resolution)


def last_padding(tomogram, depth_slice_size, height_slice_size, width_slice_size):
    # tomogram: (tensor)
    b, d, h, w = tomogram.shape
    last_depth_padding = (depth_slice_size - d) % depth_slice_size
    if last_depth_padding == depth_slice_size:
        dpad_tomogram = tomogram
    else:
        dpad_tomogram = torch.cat(
            [tomogram, torch.zeros(b, last_depth_padding, h, w).to(tomogram.device)],
            dim=1,
        )
    
    b, d, h, w = dpad_tomogram.shape
    last_height_padding = (height_slice_size - h) % height_slice_size
    if last_height_padding == height_slice_size:
        hpad_tomogram = dpad_tomogram
    else:
        hpad_tomogram = torch.cat(
            [dpad_tomogram, torch.zeros(b, d, last_height_padding, w).to(tomogram.device)],
            dim=2,
        )
    
    b, d, h, w = hpad_tomogram.shape
    last_width_padding = (width_slice_size - w) % width_slice_size
    if last_width_padding == width_slice_size:
        wpad_tomogram = hpad_tomogram
    else:
        wpad_tomogram = torch.cat(
            [hpad_tomogram, torch.zeros(b, d, h, last_width_padding).to(tomogram.device)],
            dim=3,
        )
    
    return wpad_tomogram


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
        print("start inference")
        for s_depth in tqdm(range(0, data["normalized_tomogram"].shape[-3], CFG.stride)):
            for s_height in range(0, data["normalized_tomogram"].shape[-2], CFG.h_stride):
                for s_width in range(0, data["normalized_tomogram"].shape[-1], CFG.w_stride):
                    normalized_tomogram = data["normalized_tomogram"][:, s_depth : s_depth + CFG.slice_, s_height : s_height + CFG.h_slice, s_width : s_width + CFG.w_slice]
                    normalized_tomogram = last_padding(normalized_tomogram, CFG.slice_, CFG.h_slice, CFG.w_slice)
                    # normalized_tomogram = padf(normalized_tomogram)
                    normalized_tomogram = preprocess_tensor(normalized_tomogram).to("cuda")
                    with autocast():
                        pred = model(normalized_tomogram)
                    prob_pred = (
                        torch.softmax(pred, dim=1).detach().cpu().numpy()
                    )
                    depth_edge = min(s_depth + CFG.slice_, res_array[0]) # 62+32, 92
                    height_edge = min(s_height + CFG.h_slice, res_array[1])
                    width_edge = min(s_width + CFG.w_slice, res_array[2])

                    pad_depth, pad_height, pad_width = 0, 0, 0
                    if s_depth + CFG.slice_ > res_array[0]:
                        pad_depth = (s_depth + CFG.slice_) - res_array[0]
                    if s_height + CFG.h_slice > res_array[1]:
                        pad_height = (s_height + CFG.h_slice) - res_array[1]
                    if s_width + CFG.w_slice > res_array[2]:
                        pad_width = (s_width + CFG.w_slice) - res_array[2]

                    # print("01", pad_depth, pad_height, pad_width)
                    # print("02", pred_array.shape, prob_pred.shape, cnt_array.shape)
                    # print("03", pred_array[:, s_depth:depth_edge, s_height:height_edge, s_width:width_edge].shape)
                    # print("04", prob_pred[0, :, :-pad_depth, :-pad_height, :-pad_width].shape)
                    # print("05", cnt_array[:, s_depth:depth_edge, s_height:height_edge, s_width:width_edge].shape)
                    
                    pred_array[:, s_depth:depth_edge, s_height:height_edge, s_width:width_edge] += prob_pred[
                        0, :, :CFG.slice_-pad_depth, :CFG.h_slice-pad_height, :CFG.w_slice-pad_width
                    ]
                    cnt_array[:, s_depth:depth_edge, s_height:height_edge, s_width:width_edge] += 1


            # normalized_tomogram = data["normalized_tomogram"][:, s_depth : s_depth + CFG.slice_]
            # normalized_tomogram = last_padding(normalized_tomogram, CFG.slice_, CFG.h_slice, CFG.w_slice)
            # normalized_tomogram = padf(normalized_tomogram)
            # normalized_tomogram = preprocess_tensor(normalized_tomogram).to("cuda")
            # with autocast():
            #     pred = model(normalized_tomogram)
            # prob_pred = (
            #     torch.softmax(pred, dim=1).detach().cpu().numpy()
            # )  # torch.Size([1, 7, 32, 320, 320])
            # range_ = min(s_depth + CFG.slice_, res_array[0])
            # hw_pad_diff = prob_pred.shape[-1] - res_array[-1]

            # if s_depth >= res_array[0]:
            #     continue

            # if range_ == res_array[0]:
            #     pred_array[:, s_depth:range_] += prob_pred[
            #         0, :, : res_array[0] - i, :-hw_pad_diff, :-hw_pad_diff
            #     ]
            #     cnt_array[:, s_depth:range_] += 1
            # else:
            #     pred_array[:, s_depth:range_] += prob_pred[
            #         0, :, :range_, :-hw_pad_diff, :-hw_pad_diff
            #     ]
            #     cnt_array[:, s_depth:range_] += 1

        if train:
            segmentation_map = data["segmentation_map"]
        else:
            segmentation_map = None

        normalized_tomogram = data["normalized_tomogram"]
    # tq.close()

    pred_array = pred_array / cnt_array

    return pred_array, normalized_tomogram, segmentation_map  # (7, 92, 315, 315)


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