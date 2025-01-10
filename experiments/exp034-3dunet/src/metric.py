"""
Derived from:
https://github.com/cellcanvas/album-catalog/blob/main/solutions/copick/compare-picks/solution.py
"""

import gc

import cc3d
import numpy as np
import pandas as pd
from config import CFG
from dataloader import drop_padding, read_info_json
from scipy.spatial import KDTree


def visualize_epoch_results(pred_tomogram_dict, base_dir, sikii_dict):
    """_summary_

    Args:
        pred_tomogram_dict (_type_): {exp_name: tomogram}
        gt_tomogram_dict (_type_): {exp_name: tomogram}
        sikii_dict (_type_): {particle_name: sikii}
    """
    scores = []
    experiments = list(pred_tomogram_dict.keys())

    for exp_name in experiments:
        # print(
        #     f"####################### valid-experiments: {exp_name} #######################"
        # )

        # multi-cls-pred
        pred_tomogram = np.array(pred_tomogram_dict[exp_name]).squeeze()
        # pred_tomogram = drop_padding(pred_tomogram, CFG.resolution)
        pred_tomogram = np.exp(pred_tomogram) / np.exp(pred_tomogram).sum(1)[:, None]
        # print(pred_tomogram.shape)
        pred_cls_pos, pred_Ascale_pos = create_cls_pos_sikii(
            pred_tomogram, sikii_dict=sikii_dict
        )
        pred_df = create_df(pred_Ascale_pos, exp_name)
        pred_df = pred_df.drop_duplicates(subset=["x", "y", "z"], keep="first")
        pred_df = pred_df.reset_index()

        # gt
        # gt_tomogram = np.array(gt_tomogram_dict[exp_name]).squeeze(1)
        # gt_cls_pos, gt_Ascale_pos = create_cls_pos(gt_tomogram)
        # gt_df = create_df(gt_Ascale_pos, exp_name)
        # gt_df = gt_df.reset_index()
        exp_name = exp_name.split("_")[:-1]
        exp_name = "_".join(exp_name)
        gt_df = create_gt_df(base_dir=base_dir, exp_names=[exp_name])

        score_ = score(
            solution=pred_df,
            submission=gt_df,
            row_id_column_name="index",
            distance_multiplier=1,
            beta=4,
        )
        scores.append(score_)

    return np.mean(scores), scores


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

def create_cls_pos(pred_tomogram):
    cls_pos = []
    Ascale_pos = []
    resolution_info = CFG.resolution2ratio

    for pred_cls in range(1, len(CFG.particles_name) + 1):
        cc = cc3d.connected_components(pred_tomogram == pred_cls)
        stats = cc3d.statistics(cc)

        for z, x, y in stats["centroids"]:
            Ascale_z = z * resolution_info[CFG.resolution] / resolution_info["A"]
            Ascale_x = x * resolution_info[CFG.resolution] / resolution_info["A"]
            Ascale_y = y * resolution_info[CFG.resolution] / resolution_info["A"]

            cls_pos.append([pred_cls, z, y, x])
            Ascale_pos.append([pred_cls, Ascale_z, Ascale_y, Ascale_x])

    return cls_pos, Ascale_pos


def create_cls_pos_sikii(pred_tomogram, sikii_dict=CFG.initial_sikii):
    # pred_tomogram:(depth, cls, h, w)
    cls_pos = []
    Ascale_pos = []
    resolution_info = CFG.resolution2ratio

    for pred_cls in range(1, len(CFG.particles_name) + 1):
        array_index = pred_cls
        particle_name = CFG.cls2particles[pred_cls]
        cls_tomogram = pred_tomogram[:, array_index]  # (depth, h, w)
        sikii = sikii_dict[particle_name]

        cls_tomogram[cls_tomogram > sikii] = pred_cls

        # メモリを節約しながら接続成分解析を実行
        cc = cc3d.connected_components(cls_tomogram == pred_cls)
        stats = cc3d.statistics(cc)

        # 結果を分割して処理
        for z, y, x in stats["centroids"]:
            Ascale_z = z * resolution_info[CFG.resolution] / resolution_info["A"]
            Ascale_x = x * resolution_info[CFG.resolution] / resolution_info["A"]
            Ascale_y = y * resolution_info[CFG.resolution] / resolution_info["A"]

            cls_pos.append([pred_cls, z, y, x])
            Ascale_pos.append([pred_cls, Ascale_z, Ascale_y, Ascale_x])

        # 不要データを削除してメモリ解放
        del cls_tomogram, cc, stats
        gc.collect()

    return cls_pos, Ascale_pos

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

    return pd.DataFrame(results)

def compute_metrics(reference_points, reference_radius, candidate_points):
    num_reference_particles = len(reference_points)
    num_candidate_particles = len(candidate_points)

    if len(reference_points) == 0:
        return 0, num_candidate_particles, 0

    if len(candidate_points) == 0:
        return 0, 0, num_reference_particles

    ref_tree = KDTree(reference_points)
    candidate_tree = KDTree(candidate_points)
    raw_matches = candidate_tree.query_ball_tree(ref_tree, r=reference_radius)
    matches_within_threshold = []
    for match in raw_matches:
        matches_within_threshold.extend(match)
    # Prevent submitting multiple matches per particle.
    # This won't be be strictly correct in the (extremely rare) case where true particles
    # are very close to each other.
    matches_within_threshold = set(matches_within_threshold)
    tp = int(len(matches_within_threshold))
    fp = int(num_candidate_particles - tp)
    fn = int(num_reference_particles - tp)
    return tp, fp, fn


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    distance_multiplier: float,
    beta: int,
) -> float:
    """
    F_beta
      - a true positive occurs when
         - (a) the predicted location is within a threshold of the particle radius, and
         - (b) the correct `particle_type` is specified
      - raw results (TP, FP, FN) are aggregated across all experiments for each particle type
      - f_beta is calculated for each particle type
      - individual f_beta scores are weighted by particle type for final score
    """

    particle_radius = {
        "apo-ferritin": 60,
        "beta-amylase": 65,
        "beta-galactosidase": 90,
        "ribosome": 150,
        "thyroglobulin": 130,
        "virus-like-particle": 135,
    }

    weights = {
        "apo-ferritin": 1,
        "beta-amylase": 0,
        "beta-galactosidase": 2,
        "ribosome": 1,
        "thyroglobulin": 2,
        "virus-like-particle": 1,
    }

    particle_radius = {k: v * distance_multiplier for k, v in particle_radius.items()}

    # Filter submission to only contain experiments found in the solution split
    split_experiments = set(solution["experiment"].unique())
    submission = submission.loc[submission["experiment"].isin(split_experiments)]

    # Only allow known particle types
    if not set(submission["particle_type"].unique()).issubset(set(weights.keys())):
        raise ParticipantVisibleError("Unrecognized `particle_type`.")

    assert solution.duplicated(subset=["experiment", "x", "y", "z"]).sum() == 0
    assert particle_radius.keys() == weights.keys()

    results = {}
    for particle_type in solution["particle_type"].unique():
        results[particle_type] = {
            "total_tp": 0,
            "total_fp": 0,
            "total_fn": 0,
        }

    for experiment in split_experiments:
        for particle_type in solution["particle_type"].unique():
            reference_radius = particle_radius[particle_type]
            select = (solution["experiment"] == experiment) & (
                solution["particle_type"] == particle_type
            )
            reference_points = solution.loc[select, ["x", "y", "z"]].values

            select = (submission["experiment"] == experiment) & (
                submission["particle_type"] == particle_type
            )
            candidate_points = submission.loc[select, ["x", "y", "z"]].values

            if len(reference_points) == 0:
                reference_points = np.array([])
                reference_radius = 1

            if len(candidate_points) == 0:
                candidate_points = np.array([])

            tp, fp, fn = compute_metrics(
                reference_points, reference_radius, candidate_points
            )

            results[particle_type]["total_tp"] += tp
            results[particle_type]["total_fp"] += fp
            results[particle_type]["total_fn"] += fn

    aggregate_fbeta = 0.0
    for particle_type, totals in results.items():
        tp = totals["total_tp"]
        fp = totals["total_fp"]
        fn = totals["total_fn"]

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        fbeta = (
            (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        aggregate_fbeta += fbeta * weights.get(particle_type, 1.0)

    if weights:
        aggregate_fbeta = aggregate_fbeta / sum(weights.values())
    else:
        aggregate_fbeta = aggregate_fbeta / len(results)
    return aggregate_fbeta