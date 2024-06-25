"""
To employ ImageFolder, the images should be arranged into:

  root: train/val/test
    class_a
      a1.png
      a2.png
      ...
    class_b
      b1.png
      b2.png
      ...
"""

import os
import shutil

import numpy as np
import pandas as pd

if __name__ == "__main__":
    train_val_ratio = 0.9

    train_label = pd.read_csv("GroundTruth/ISBI2016_ISIC_Part3_Training_GroundTruth.csv")
    test_label = pd.read_csv("GroundTruth/ISBI2016_ISIC_Part3_Test_GroundTruth.csv")

    # change the column name to an appropriate name
    train_label = train_label.rename(
        columns={"ISIC_0000000": "image", "benign": "label"}
    )
    test_label = test_label.rename(columns={"ISIC_0000003": "image", "0.0": "label"})

    # concatenate the original column name with the renamed dataframe
    train_label = pd.concat(
        [pd.DataFrame({"image": ["ISIC_0000000"], "label": ["benign"]}), train_label]
    )
    test_label = pd.concat(
        [pd.DataFrame({"image": ["ISIC_0000003"], "label": ["0.0"]}), test_label]
    )

    # split the train and val
    total_trainset, total_testset = (
        train_label["image"].values.shape[0],
        test_label["image"].values.shape[0],
    )  # 900, 379
    split_length = int(train_val_ratio * total_trainset)

    # allocate train and val dataset into folder
    dir_lists = [
        "train",
        "val",
        "test",
        "train/benign",
        "train/malignant",
        "val/benign",
        "val/malignant",
        "test/0.0",
        "test/1.0",
    ]
    for _dir in dir_lists:
        if os.path.exists(_dir):
            shutil.rmtree(_dir, ignore_errors=True)
        os.mkdir(_dir)

    new_train_images, new_train_labels = (
        train_label["image"].values,
        train_label["label"].values,
    )
    new_train_images = new_train_images.reshape(-1)

    # create the new dataframe after oversampling
    benign_index = np.where(new_train_labels == "benign")[0]
    malignant_index = np.where(new_train_labels == "malignant")[0]
    all_train_benign_images = pd.DataFrame(
        {
            "image": new_train_images[benign_index],
            "label": new_train_labels[benign_index],
        }
    )
    all_train_malignant_images = pd.DataFrame(
        {
            "image": new_train_images[malignant_index],
            "label": new_train_labels[malignant_index],
        }
    )

    benign_split_length = int(all_train_benign_images.shape[0] * train_val_ratio)
    malignant_split_length = int(all_train_malignant_images.shape[0] * train_val_ratio)

    # split the train and val with equal ratio of benign and malignant
    train_images = pd.concat(
        [
            all_train_benign_images.iloc[:benign_split_length],
            all_train_malignant_images.iloc[:malignant_split_length],
        ]
    )
    val_images = pd.concat(
        [
            all_train_benign_images.iloc[benign_split_length:],
            all_train_malignant_images.iloc[malignant_split_length:],
        ]
    )

    data_dict = {"train": train_images, "val": val_images, "test": test_label}

    folder = "ISBI2016_ISIC_Part3_Training_Data"
    for folder_name, images in data_dict.items():
        if folder_name == "test":
            folder = "ISBI2016_ISIC_Part3_Test_Data"
        for image_name, label in zip(images["image"], images["label"]):
            src = os.path.join(folder, f"{image_name}.jpg")
            dst = os.path.join(f"{folder_name}/{label}", f"{image_name}.jpg")
            shutil.copyfile(src, dst)
