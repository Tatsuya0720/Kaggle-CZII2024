import ndjson
import numpy as np
import zarr


def read_zarr(zarr_pth, resolution="0"):
    zarr_store = zarr.open(zarr_pth, mode="r")
    tomogram = zarr_store[resolution][:]
    return tomogram

def multiply_rotation_matrix(xyz, rotation_matrix):
    # xyz: (3, )
    # rotation_matrix: (3, 3)
    xyz = xyz.reshape(3, 1)
    return xyz.T @ rotation_matrix

def pos_extractor(jsonfile):
    pos = []
    for info_ in jsonfile:
        one_pos = {
            "location": {
                "x": 10*float(info_["location"]["x"]),
                "y": 10*float(info_["location"]["y"]),
                "z": 10*float(info_["location"]["z"]),
            },
        }
        pos.append(one_pos)

    return pos


def wrapped_pos_extractor(wrapped_pos_info):
    dict_ = {}
    dict_["points"] = wrapped_pos_info
    return dict_


def get_pos_info(pos_path):
    try:
        with open(pos_path) as f:
            pos = ndjson.load(f)

        wrapped_pos_info = wrapped_pos_extractor(pos_extractor(pos))
    except:
        wrapped_pos_info = {"points": []}

    return wrapped_pos_info