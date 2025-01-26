import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from torch.utils.data import DataLoader, Dataset
from utils import get_exp_names


class CFG:
    resolution = "0"

    original_img_shape = {
        "0": (184, 630, 630),
        "1": (92, 315, 315),
        "2": (50, 158, 158),
    }

    augmentation_prob = 0.45
    slice_ = 16
    w_slice = 32
    h_slice = 32
    stride = 8
    epochs = 80
    lr = 1e-3
    weight_decay = 1e-6
    batch_size = 2
    model_name = "resnet34d"
    augment_data_ratio = original_img_shape[resolution][0] // slice_
    num_workers = 16

    encoder_dim = {
        "resnet34d": [64, 64, 128, 256, 512],
        "resnext50_32x4d":[64, 256, 512, 1024, 2048],
    }

    valid_exp_names = ["TS_5_4"]
    train_exp_names = get_exp_names("../../inputs/train/overlay/ExperimentRuns/")
    # train_exp_names = ["TS_73_6", "TS_99_9", "TS_6_4", "TS_69_2","TS_86_3", "TS_6_6"]
    # train_exp_names = ["TS_5_4"]

    for valid_exp_name in valid_exp_names:
        if valid_exp_name in train_exp_names:
            train_exp_names.remove(valid_exp_name)

    train_zarr_types = ["denoised", "ctfdeconvolved", "wbp", "isonetcorrected"]
    # train_zarr_types = ["denoised"]
    valid_zarr_types = ["denoised"]

    constant = 0.5
    initial_sikii = {
        "apo-ferritin": constant,
        "beta-amylase": constant,
        "beta-galactosidase": constant,
        "ribosome": constant,
        "thyroglobulin": constant,
        "virus-like-particle": constant,
    }

    zarr2idx = {
        "denoised": 0,
        "ctfdeconvolved": 1,
        "wbp": 2,
        "isonetcorrected": 3,
        "none": 4,
    }

    particles_name = [
        "apo-ferritin",
        "beta-amylase",
        "beta-galactosidase",
        "ribosome",
        "thyroglobulin",
        "virus-like-particle",
    ]

    resolution2ratio = {
        "A": 1 / 10,
        "0": 1,
        "1": 2,
        "2": 4,
    }

    particles2cls = {
        "none": 0,
        "apo-ferritin": 1,
        "beta-amylase": 2,
        "beta-galactosidase": 3,
        "ribosome": 4,
        "thyroglobulin": 5,
        "virus-like-particle": 6,
    }
    cls2particles = {
        0: "none",
        1: "apo-ferritin",
        2: "beta-amylase",
        3: "beta-galactosidase",
        4: "ribosome",
        5: "thyroglobulin",
        6: "virus-like-particle",
    }

    particle_radius = {
        "apo-ferritin": 60,
        "beta-amylase": 65,
        "beta-galactosidase": 90,
        "ribosome": 150,
        "thyroglobulin": 130,
        "virus-like-particle": 135,
    }

    particle_weights = {
        "apo-ferritin": 1,
        "beta-amylase": 0,
        "beta-galactosidase": 2,
        "ribosome": 1,
        "thyroglobulin": 2,
        "virus-like-particle": 1,
    }

    colormap = {
        # -1の場合は透明の色
    }