import json
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import zarr
from scipy.optimize import linear_sum_assignment


def time_to_str(t, mode="min"):
    if mode == "min":
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return "%2d hr %02d min" % (hr, min)

    elif mode == "sec":
        t = int(t)
        min = t // 60
        sec = t % 60
        return "%2d min %02d sec" % (min, sec)

    else:
        raise NotImplementedError


class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


PARTICLE = [
    {
        "name": "apo-ferritin",
        "difficulty": "easy",
        "pdb_id": "4V1W",
        "label": 1,
        "color": [0, 255, 0, 0],
        "radius": 60,
        "map_threshold": 0.0418,
    },
    {
        "name": "beta-amylase",
        "difficulty": "ignore",
        "pdb_id": "1FA2",
        "label": 2,
        "color": [0, 0, 255, 255],
        "radius": 65,
        "map_threshold": 0.035,
    },
    {
        "name": "beta-galactosidase",
        "difficulty": "hard",
        "pdb_id": "6X1Q",
        "label": 3,
        "color": [0, 255, 0, 255],
        "radius": 90,
        "map_threshold": 0.0578,
    },
    {
        "name": "ribosome",
        "difficulty": "easy",
        "pdb_id": "6EK0",
        "label": 4,
        "color": [0, 0, 255, 0],
        "radius": 150,
        "map_threshold": 0.0374,
    },
    {
        "name": "thyroglobulin",
        "difficulty": "hard",
        "pdb_id": "6SCJ",
        "label": 5,
        "color": [0, 255, 255, 0],
        "radius": 130,
        "map_threshold": 0.0278,
    },
    {
        "name": "virus-like-particle",
        "difficulty": "easy",
        "pdb_id": "6N4V",
        "label": 6,
        "color": [0, 0, 0, 255],
        "radius": 135,
        "map_threshold": 0.201,
    },
]

PARTICLE_COLOR = [[0, 0, 0]] + [PARTICLE[i]["color"][1:] for i in range(6)]
PARTICLE_NAME = ["none"] + [PARTICLE[i]["name"] for i in range(6)]

"""
(184, 630, 630)  
(92, 315, 315)  
(46, 158, 158)  
"""


def read_one_data(id, static_dir):
    zarr_dir = f"{static_dir}/{id}/VoxelSpacing10.000"
    zarr_file = f"{zarr_dir}/denoised.zarr"
    zarr_data = zarr.open(zarr_file, mode="r")
    volume = zarr_data[0][:]
    max = volume.max()
    min = volume.min()
    volume = (volume - min) / (max - min)
    volume = volume.astype(np.float16)
    return volume


def read_one_truth(id, overlay_dir):
    location = {}

    json_dir = f"{overlay_dir}/{id}/Picks"
    for p in PARTICLE_NAME[1:]:
        json_file = f"{json_dir}/{p}.json"

        with open(json_file, "r") as f:
            json_data = json.load(f)

        num_point = len(json_data["points"])
        loc = np.array(
            [
                list(json_data["points"][i]["location"].values())
                for i in range(num_point)
            ]
        )
        location[p] = loc

    return location


def do_one_eval(truth, predict, threshold):
    P = len(predict)
    T = len(truth)

    if P == 0:
        hit = [[], []]
        miss = np.arange(T).tolist()
        fp = []
        metric = [P, T, len(hit[0]), len(miss), len(fp)]
        return hit, fp, miss, metric

    if T == 0:
        hit = [[], []]
        fp = np.arange(P).tolist()
        miss = []
        metric = [P, T, len(hit[0]), len(miss), len(fp)]
        return hit, fp, miss, metric

    # ---
    distance = predict.reshape(P, 1, 3) - truth.reshape(1, T, 3)
    distance = distance**2
    distance = distance.sum(axis=2)
    distance = np.sqrt(distance)
    p_index, t_index = linear_sum_assignment(distance)

    valid = distance[p_index, t_index] <= threshold
    p_index = p_index[valid]
    t_index = t_index[valid]
    hit = [p_index.tolist(), t_index.tolist()]
    miss = np.arange(T)
    miss = miss[~np.isin(miss, t_index)].tolist()
    fp = np.arange(P)
    fp = fp[~np.isin(fp, p_index)].tolist()

    metric = [P, T, len(hit[0]), len(miss), len(fp)]  # for lb metric F-beta copmutation
    return hit, fp, miss, metric


def compute_lb(submit_df, overlay_dir, valid_id):
    eval_df = []
    for id in valid_id:
        truth = read_one_truth(
            id, overlay_dir
        )  # =f'{valid_dir}/overlay/ExperimentRuns')
        id_df = submit_df[submit_df["experiment"] == id]
        for p in PARTICLE:
            p = dotdict(p)
            # print("\r", id, p.name, end="", flush=True)
            xyz_truth = truth[p.name]
            xyz_predict = id_df[id_df["particle_type"] == p.name][
                ["x", "y", "z"]
            ].values
            hit, fp, miss, metric = do_one_eval(xyz_truth, xyz_predict, p.radius * 0.5)
            eval_df.append(
                dotdict(
                    id=id,
                    particle_type=p.name,
                    P=metric[0],
                    T=metric[1],
                    hit=metric[2],
                    miss=metric[3],
                    fp=metric[4],
                )
            )
    eval_df = pd.DataFrame(eval_df)
    gb = eval_df.groupby("particle_type").agg("sum").drop(columns=["id"])
    gb.loc[:, "precision"] = gb["hit"] / gb["P"]
    gb.loc[:, "precision"] = gb["precision"].fillna(0)
    gb.loc[:, "recall"] = gb["hit"] / gb["T"]
    gb.loc[:, "recall"] = gb["recall"].fillna(0)
    gb.loc[:, "f-beta4"] = (
        17 * gb["precision"] * gb["recall"] / (16 * gb["precision"] + gb["recall"])
    )
    gb.loc[:, "f-beta4"] = gb["f-beta4"].fillna(0)

    gb = gb.sort_values("particle_type").reset_index(drop=False)
    # https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/544895
    gb.loc[:, "weight"] = [1, 0, 2, 1, 2, 1]
    lb_score = (gb["f-beta4"] * gb["weight"]).sum() / gb["weight"].sum()
    return gb, lb_score

def extract_particle_results(gb):
    apo_ferritin = gb[gb["particle_type"] == "apo-ferritin"]
    beta_amylase = gb[gb["particle_type"] == "beta-amylase"]
    beta_galactosidase = gb[gb["particle_type"] == "beta-galactosidase"]
    ribosome = gb[gb["particle_type"] == "ribosome"]
    thyroglobulin = gb[gb["particle_type"] == "thyroglobulin"]
    virus_like_particle = gb[gb["particle_type"] == "virus-like-particle"]

    apo_ferritin_r = apo_ferritin["recall"].values[0]
    apo_ferritin_p = apo_ferritin["precision"].values[0]
    apo_ferritin_f4 = apo_ferritin["f-beta4"].values[0]

    beta_amylase_r = beta_amylase["recall"].values[0]
    beta_amylase_p = beta_amylase["precision"].values[0]
    beta_amylase_f4 = beta_amylase["f-beta4"].values[0]

    beta_galactosidase_r = beta_galactosidase["recall"].values[0]
    beta_galactosidase_p = beta_galactosidase["precision"].values[0]
    beta_galactosidase_f4 = beta_galactosidase["f-beta4"].values[0]

    ribosome_r = ribosome["recall"].values[0]
    ribosome_p = ribosome["precision"].values[0]
    ribosome_f4 = ribosome["f-beta4"].values[0]

    thyroglobulin_r = thyroglobulin["recall"].values[0]
    thyroglobulin_p = thyroglobulin["precision"].values[0]
    thyroglobulin_f4 = thyroglobulin["f-beta4"].values[0]

    virus_like_particle_r = virus_like_particle["recall"].values[0]
    virus_like_particle_p = virus_like_particle["precision"].values[0]
    virus_like_particle_f4 = virus_like_particle["f-beta4"].values[0]

    return {
        "apoo_ferritin_r": apo_ferritin_r,
        "apoo_ferritin_p": apo_ferritin_p,
        "apoo_ferritin_f4": apo_ferritin_f4,
        "beta_amylase_r": beta_amylase_r,
        "beta_amylase_p": beta_amylase_p,
        "beta_amylase_f4": beta_amylase_f4,
        "beta_galactosidase_r": beta_galactosidase_r,
        "beta_galactosidase_p": beta_galactosidase_p,
        "beta_galactosidase_f4": beta_galactosidase_f4,
        "ribosome_r": ribosome_r,
        "ribosome_p": ribosome_p,
        "ribosome_f4": ribosome_f4,
        "thyroglobulin_r": thyroglobulin_r,
        "thyroglobulin_p": thyroglobulin_p,
        "thyroglobulin_f4": thyroglobulin_f4,
        "virus_like_particle_r": virus_like_particle_r,
        "virus_like_particle_p": virus_like_particle_p,
        "virus_like_particle_f4": virus_like_particle_f4,
        
    }
