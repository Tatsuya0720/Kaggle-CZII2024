import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from torch.utils.data import DataLoader, Dataset


class CFG:
    resolution = "2"

    train_exp_names = ["TS_5_4", "TS_73_6", "TS_99_9", "TS_6_4", "TS_69_2"]
    # train_exp_names = ["TS_5_4"]
    valid_exp_names = ["TS_86_3", "TS_6_6"]

    # train_zarr_types = ["denoised", "ctfdeconvolved", "wbp", "isonetcorrected"]
    train_zarr_types = ["denoised"]
    valid_zarr_types = ["denoised"]

    initial_sikii = {
        "apo-ferritin": 0.35,
        "beta-amylase": 0.35,
        "beta-galactosidase": 0.35,
        "ribosome": 0.35,
        "thyroglobulin": 0.35,
        "virus-like-particle": 0.35,
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