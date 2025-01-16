import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import zarr
from config import CFG
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def read_zarr(zarr_pth, resolution="0"):
    zarr_store = zarr.open(zarr_pth, mode="r")

    tomogram = zarr_store[resolution][:]

    return tomogram


def read_info_json(
    base_dir="../inputs/train/overlay/ExperimentRuns/",
    exp_name="TS_5_4",
    particle_name="apo-ferritin",
):

    if particle_name not in CFG.particles_name:
        raise ValueError(
            f"Particle name should be one of the following: {CFG.particles_name}. Got {particle_name}."
        )

    read_json_path = os.path.join(base_dir, exp_name, "Picks", f"{particle_name}.json")

    with open(read_json_path, "r") as f:
        particle_info = json.load(f)

    coords = []
    for point in particle_info["points"]:
        coords.append(
            [point["location"]["z"], point["location"]["y"], point["location"]["x"]]
        )

    coords = np.array(coords)

    return coords


def scale_coordinates(coords, tomogram_shape, resolution):
    """Scale coordinates to match tomogram dimensions."""
    scaled_coords = coords.copy()

    # scaled_coords[:, 0] = coords[:, 0] / 10  # / coords[:, 0].max() * tomogram_shape[0]
    # scaled_coords[:, 1] = coords[:, 1] / 10  # / coords[:, 1].max() * tomogram_shape[1]
    # scaled_coords[:, 2] = coords[:, 2] / 10  # / coords[:, 2].max() * tomogram_shape[2]

    resolution_info = CFG.resolution2ratio
    scaled_coords[:, 0] = (
        coords[:, 0] * resolution_info["A"] / resolution_info[resolution]
    )
    scaled_coords[:, 1] = (
        coords[:, 1] * resolution_info["A"] / resolution_info[resolution]
    )
    scaled_coords[:, 2] = (
        coords[:, 2] * resolution_info["A"] / resolution_info[resolution]
    )

    return scaled_coords


def create_dataset(
    zarr_type,
    base_dir,
    exp_name,
    resolution,
    particle_names,
    train=True,
):
    zarr_name = f"{zarr_type}.zarr"
    zarr_pth = os.path.join(
        base_dir, "static/ExperimentRuns", exp_name, "VoxelSpacing10.000/", zarr_name
    )
    tomogram = read_zarr(zarr_pth, resolution=resolution)

    if train:
        particle_info = {"corrds": {}, "scaled_corrds": {}}
        for particle_name in particle_names:
            coords = read_info_json(
                os.path.join(base_dir, "overlay/ExperimentRuns/"), exp_name, particle_name
            )
            scaled_coords = scale_coordinates(coords, tomogram.shape, resolution)
            particle_info["corrds"][particle_name] = coords
            particle_info["scaled_corrds"][particle_name] = scaled_coords

        return tomogram, particle_info
    return tomogram, None


def normalise_by_percentile(data, min=5, max=99):
    min = np.percentile(data, min)
    max = np.percentile(data, max)
    data = (data - min) / (max - min)
    return data, min, max


class EziiDataset(Dataset):
    def __init__(
        self,
        exp_names=[],
        base_dir="../../inputs/train/static",
        particles_name=CFG.particles_name,
        resolution="0",
        zarr_type=["ctfdeconvolved"],
        train=False,
        slice=False,
        augmentation=False,
        pre_read = False,
    ):
        self.exp_names = exp_names
        self.base_dir = base_dir
        self.particles_name = particles_name
        self.resolution = resolution
        self.zarr_type = zarr_type
        self.train = train
        self.slice = slice
        self.pre_read = pre_read
        
        # exp_namesとzarr_typeの総当たりでデータを作成
        self.data = []
        
        for exp_name in exp_names:
            for type_ in zarr_type:
                self.data.append((exp_name, type_))

        if augmentation: # 1つの実験ファイルからスライス分しか使わないのでその分を増やす
            self.data = self.data * CFG.augment_data_ratio

        self.data_dict = {}

        if pre_read:
            self.pre_read_f()

    def __getitem__(self, i):
        if self.train:
            if self.pre_read:
                exp_name, type_ = self.data[i]  # TS_6_6
                # apo_ferritin = self.data_dict[exp_name]["apo_ferritin"]
                # beta_amylase = self.data_dict[exp_name]["beta_amylase"]
                # beta_galactosidase = self.data_dict[exp_name]["beta_galactosidase"]
                # ribosome = self.data_dict[exp_name]["ribosome"]
                # thyroglobulin = self.data_dict[i]["thyroglobulin"]
                # virus_like_particle = self.data_dict[exp_name]["virus_like_particle"]
                normalized_tomogram = self.data_dict[exp_name]["normalized_tomogram"]
                tomogram = self.data_dict[exp_name]["tomogram"]
                segmentation_map = self.data_dict[exp_name]["segmentation_map"]
            else:
                exp_name, type_ = self.data[i]  # TS_6_6
                tomogram, particle_info = create_dataset(
                    base_dir=self.base_dir,
                    particle_names=self.particles_name,
                    resolution=self.resolution,
                    exp_name=exp_name,
                    zarr_type=type_,
                    train=self.train,
                )

                normalized_tomogram, min, max = normalise_by_percentile(tomogram)
                
                apo_ferritin = particle_info["scaled_corrds"]["apo-ferritin"]
                beta_amylase = particle_info["scaled_corrds"]["beta-amylase"]
                beta_galactosidase = particle_info["scaled_corrds"]["beta-galactosidase"]
                ribosome = particle_info["scaled_corrds"]["ribosome"]
                thyroglobulin = particle_info["scaled_corrds"]["thyroglobulin"]
                virus_like_particle = particle_info["scaled_corrds"]["virus-like-particle"]

                prticle_corrds = {
                    "apo-ferritin": apo_ferritin,
                    "beta-amylase": beta_amylase,
                    "beta-galactosidase": beta_galactosidase,
                    "ribosome": ribosome,
                    "thyroglobulin": thyroglobulin,
                    "virus-like-particle": virus_like_particle,
                }

                segmentation_map = create_segmentation_map(
                    tomogram, resolution=self.resolution, particle_coords=prticle_corrds
                )

                if len(exp_name.split("_")) == 2: # 追加データセットならば
                    segmentation_map = read_all_particle_seg_zarr(exp_name, self.resolution)

            if self.slice:
                # tomogramの深さ部分から連続な32スライスをランダムに取得
                max_depth = self.get_max_depth(self.resolution)
                start_depth = random.randint(0, max_depth - (CFG.slice_+1))
                end_depth = start_depth + CFG.slice_

                tomogram = tomogram[start_depth:end_depth]
                normalized_tomogram = normalized_tomogram[start_depth:end_depth]
                segmentation_map = segmentation_map[start_depth:end_depth]
            
            return {
                "exp_name": exp_name,
                "resolution": self.resolution,
                "tomogram": tomogram,
                "normalized_tomogram": normalized_tomogram,
                "segmentation_map": segmentation_map,
                # "apo_ferritin": apo_ferritin,
                # "beta_amylase": beta_amylase,
                # "beta_galactosidase": beta_galactosidase,
                # "ribosome": ribosome,
                # "thyroglobulin": thyroglobulin,
                # "virus_like_particle": virus_like_particle,
                # "particle_corrds": prticle_corrds,
            }
        else:
            exp_name, type_ = self.data[i]  # TS_6_6
            tomogram, particle_info = create_dataset(
                base_dir=self.base_dir,
                particle_names=self.particles_name,
                resolution=self.resolution,
                exp_name=exp_name,
                zarr_type=type_,
                train=self.train,
            )
            normalized_tomogram, min, max = normalise_by_percentile(tomogram)
            return {
                "exp_name": exp_name,
                "resolution": self.resolution,
                "tomogram": tomogram,
                "normalized_tomogram": normalized_tomogram,
            }

    def __len__(self):
        return len(self.data)
    
    def get_max_depth(self, resolution):
        if resolution == "0":
            return CFG.original_img_shape["0"][0]
        elif resolution == "1":
            return CFG.original_img_shape["1"][0]
        elif resolution == "2":
            return CFG.original_img_shape["2"][0]

    def pre_read_f(self):
        for i in tqdm(range(len(self.data))):
            exp_name, type_ = self.data[i]  # TS_6_6

            if self.data_dict.get(exp_name) is not None:
                continue

            tomogram, particle_info = create_dataset(
                base_dir=self.base_dir,
                particle_names=self.particles_name,
                resolution=self.resolution,
                exp_name=exp_name,
                zarr_type=type_,
                train=self.train,
            )

            normalized_tomogram, min, max = normalise_by_percentile(tomogram)

            # tomogramの深さ部分から連続な32スライスをランダムに取得
            max_depth = self.get_max_depth(self.resolution)
            start_depth = random.randint(0, max_depth - (CFG.slice_+1))
            end_depth = start_depth + CFG.slice_
        
            if particle_info is not None:
                apo_ferritin = particle_info["scaled_corrds"]["apo-ferritin"]
                beta_amylase = particle_info["scaled_corrds"]["beta-amylase"]
                beta_galactosidase = particle_info["scaled_corrds"]["beta-galactosidase"]
                ribosome = particle_info["scaled_corrds"]["ribosome"]
                thyroglobulin = particle_info["scaled_corrds"]["thyroglobulin"]
                virus_like_particle = particle_info["scaled_corrds"]["virus-like-particle"]

                prticle_corrds = {
                    "apo-ferritin": apo_ferritin,
                    "beta-amylase": beta_amylase,
                    "beta-galactosidase": beta_galactosidase,
                    "ribosome": ribosome,
                    "thyroglobulin": thyroglobulin,
                    "virus-like-particle": virus_like_particle,
                }

                segmentation_map = create_segmentation_map(
                    tomogram, resolution=self.resolution, particle_coords=prticle_corrds
                )

                if len(exp_name.split("_")) == 2: # 追加データセットならば
                    segmentation_map = read_all_particle_seg_zarr(exp_name, self.resolution)
            
            self.data_dict[exp_name] = {
                "exp_name": exp_name,
                "resolution": self.resolution,
                "exp_name": exp_name,
                "tomogram": tomogram,
                "normalized_tomogram": normalized_tomogram,
                "segmentation_map": segmentation_map,
                # "apo_ferritin": apo_ferritin,
                # "beta_amylase": beta_amylase,
                # "beta_galactosidase": beta_galactosidase,
                # "ribosome": ribosome,
                # "thyroglobulin": thyroglobulin,
                # "virus_like_particle": virus_like_particle,
                # "particle_corrds": prticle_corrds,
            }

def drop_padding(tomogram, resolution):
    if resolution == "0":  # 184, 640, 640 -> 184, 630, 630
        if len(tomogram.shape) == 3:
            tomogram = tomogram[:, :-10, :-10]
        else:
            tomogram = tomogram[:, :, :-10, :-10]
        assert tomogram.shape[-1] == 630
        assert tomogram.shape[-2] == 630
        return tomogram
    elif resolution == "1":  # 92, 320, 320 -> 92, 315, 315
        if len(tomogram.shape) == 3:
            tomogram = tomogram[:, :-5, :-5]
        else:
            tomogram = tomogram[:, :, :-5, :-5]
        assert tomogram.shape[-1] == 315
        assert tomogram.shape[-2] == 315
        return tomogram
    elif resolution == "2":  # 46, 160, 160 -> 46, 158, 158
        if len(tomogram.shape) == 3:
            tomogram = tomogram[:, :-2, :-2]
        else:
            tomogram = tomogram[:, :, :-2, :-2]

        assert tomogram.shape[-1] == 158
        assert tomogram.shape[-2] == 158
        return tomogram
    raise ValueError(
        f"Resolution should be one of the following: 0, 1, 2. Got {resolution}."
    )



def create_segmentation_map(tomogram, resolution, particle_coords={}):
    segmentation_map = np.zeros_like(tomogram)
    segmentation_map[:, :, :] = 0

    particle_radius = (
        CFG.particle_radius
    )  # {apo-ferritin: 60, beta-amylase: 65, beta-galactosidase: 90, ribosome: 150, thyroglobulin: 130, virus-like-particle: 135}
    particle2cls = CFG.particles2cls
    resolution_info = CFG.resolution2ratio  # {A: 1/10, 0: 1, 1: 2, 2: 4}

    r_by_particle = {}
    for particle_name, r in particle_radius.items():
        r_by_particle[particle_name] = (
            r * resolution_info["A"] / resolution_info[resolution] // 2 +1
        )
        # r_by_particle[particle_name] = 3
        # print(f"{particle_name}: {r_by_particle[particle_name]}→{r * resolution_info['A'] / resolution_info[resolution]//2+1}")

    for i, (paraticle_name, coords) in enumerate(particle_coords.items()):
        # print(coords.shape)
        for z, y, x in coords:
            z, y, x = round(z), round(y), round(x)
            cls = particle2cls[paraticle_name]
            r = r_by_particle[paraticle_name]
            z_min = int(max(0, z - r))
            z_max = int(min(tomogram.shape[0], z + r))
            y_min = int(max(0, y - r))
            y_max = int(min(tomogram.shape[1], y + r))
            x_min = int(max(0, x - r))
            x_max = int(min(tomogram.shape[2], x + r))

            # x,y,zを中心に円計上にクラスを埋める
            segmentation_map[z_min:z_max, y_min:y_max, x_min:x_max] = cls
            # for z_ in range(z_min, z_max):
            #     for y_ in range(y_min, y_max):
            #         for x_ in range(x_min, x_max):
            #             if (z - z_) ** 2 + (y - y_) ** 2 + (x - x_) ** 2 <= r**2:
            #                 segmentation_map[z_, y_, x_] = cls

    return segmentation_map

def read_segmask_zarr(path, particle_name):
    mask = read_zarr(path)
    mask = mask.astype(np.uint8)
    mask[mask != 0] = CFG.particles2cls[particle_name]

    # 1次元目にチャネル次元を追加
    mask = np.expand_dims(mask, axis=1)

    return mask

def read_all_particle_seg_zarr(exp_name, resolution):
    base_dir = (
        f"../../inputs/10441/{exp_name}/Reconstructions/VoxelSpacing10.000/Annotations"
    )

    apo_ferritin = read_segmask_zarr(
        f"{base_dir}/101/ferritin_complex-1.0_segmentationmask.zarr",
        "apo-ferritin",
    )

    beta_amylase = read_segmask_zarr(
        f"{base_dir}/102/beta_amylase-1.0_segmentationmask.zarr",
        "beta-amylase",
    )

    beta_galactosidase = read_segmask_zarr(
        f"{base_dir}/103/beta_galactosidase-1.0_segmentationmask.zarr",
        "beta-galactosidase",
    )

    ribosome = read_segmask_zarr(
        f"{base_dir}/104/cytosolic_ribosome-1.0_segmentationmask.zarr",
        "ribosome",
    )

    thyroglobulin = read_segmask_zarr(
        f"{base_dir}/105/thyroglobulin-1.0_segmentationmask.zarr",
        "thyroglobulin",
    )

    virus_like_particle = read_segmask_zarr(
        f"{base_dir}/106/pp7_vlp-1.0_segmentationmask.zarr",
        "virus-like-particle",
    )

    segmentation_mask = np.zeros_like(beta_amylase)
    segmentation_mask = np.where(apo_ferritin != 0, apo_ferritin, segmentation_mask)
    segmentation_mask = np.where(beta_amylase != 0, beta_amylase, segmentation_mask)
    segmentation_mask = np.where(
        beta_galactosidase != 0, beta_galactosidase, segmentation_mask
    )
    segmentation_mask = np.where(ribosome != 0, ribosome, segmentation_mask)
    segmentation_mask = np.where(thyroglobulin != 0, thyroglobulin, segmentation_mask)
    segmentation_mask = np.where(
        virus_like_particle != 0, virus_like_particle, segmentation_mask
    )
    segmentation_mask = segmentation_mask.squeeze(1)  # (200, 630, 630)

    segmentation_mask = resize_3dvolume(segmentation_mask, resolution)

    return segmentation_mask

def resize_3dvolume(volume, resolution):
    resize_size = CFG.original_img_shape[resolution]
    resized_volume = resize(
        volume,
        resize_size,  # (new_depth, new_height, new_width)
        order=1,  # 補間の次数 (1=linear, 3=cubicなど)
        preserve_range=True,  # 元の値域を保つ (必要に応じて)
        anti_aliasing=False,  # 必要に応じて
    )
    return resized_volume