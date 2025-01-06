import matplotlib.pyplot as plt


def save_images(
    pred_tomogram,
    gt_tomogram,
    save_dir="images",
    epoch=0,
):
    import os

    train_save_dir = os.path.join(save_dir, f"epoch_{epoch}", "train")
    valid_save_dir = os.path.join(save_dir, f"epoch_{epoch}", "valid")
    os.makedirs(train_save_dir, exist_ok=True)
    os.makedirs(valid_save_dir, exist_ok=True)

    for i in range(len(pred_tomogram)):
        valid_pred = pred_tomogram[i].argmax(1).squeeze(0)
        valid_gt = gt_tomogram[i].squeeze(0)

        plt.figure(figsize=(10, 5))

        # 2つの画像を並べて表示
        ax = plt.subplot(1, 2, 1)
        ax.imshow(valid_pred, cmap="tab10")
        ax.set_title("Prediction")
        ax.axis("off")

        ax = plt.subplot(1, 2, 2)
        ax.imshow(valid_gt, cmap="tab10")
        ax.set_title("Ground Truth")
        ax.axis("off")

        plt.savefig(os.path.join(valid_save_dir, f"valid_{i}.png"))
        plt.close()

import os
from glob import glob

import matplotlib.pyplot as plt


def get_exp_names(path):
    dir_names = []
    for dir_name in glob(path + "/*"):
        if os.path.isdir(dir_name):
            dir_names.append(dir_name)

    exp_names = []
    for dir_name in dir_names:
        exp_names.append(dir_name.split("/")[-1])

    return exp_names

def save_images(
    pred_tomogram,
    gt_tomogram,
    save_dir="images",
    epoch=0,
):
    import os

    train_save_dir = os.path.join(save_dir, f"epoch_{epoch}", "train")
    valid_save_dir = os.path.join(save_dir, f"epoch_{epoch}", "valid")
    os.makedirs(train_save_dir, exist_ok=True)
    os.makedirs(valid_save_dir, exist_ok=True)

    for i in range(len(pred_tomogram)):
        valid_pred = pred_tomogram[i].argmax(1).squeeze(0)
        valid_gt = gt_tomogram[i].squeeze(0)

        plt.figure(figsize=(10, 5))

        # 2つの画像を並べて表示
        ax = plt.subplot(1, 2, 1)
        ax.imshow(valid_pred, cmap="tab10")
        ax.set_title("Prediction")
        ax.axis("off")

        ax = plt.subplot(1, 2, 2)
        ax.imshow(valid_gt, cmap="tab10")
        ax.set_title("Ground Truth")
        ax.axis("off")

        plt.savefig(os.path.join(valid_save_dir, f"valid_{i}.png"))
        plt.close()