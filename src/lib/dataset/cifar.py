import os
import sys
import pathlib
import time
import collections
import itertools
import shutil
import pickle
import inspect
import json
import subprocess
import logging
import numpy as np
import torch.utils.data as data
from torchvision.transforms import transforms
from PIL import Image


class Cifar10(data.Dataset):

    def __init__(self, split, args, logger, trans=None, c_aug=None, s_aug=None):
        assert split in ("train", "test", "val")
        self.args = args
        self.logger = logger
        self.split = split

        self.trans = trans
        self.c_aug = c_aug
        self.s_aug = s_aug
        if self.c_aug is not None:
            if isinstance(self.c_aug, transforms.Compose):
                aug_class = "transform"
            else:
                aug_class = "imgaug"
        else:
            aug_class = "transform"
        self.aug_class = aug_class
        self.logger.info("### USE COLOR AUGUMENTATION ###")
        self.logger.info(str(c_aug))

        self.logger.info("### USE SHAPE AUGUMENTATION ###")
        self.logger.info(str(s_aug))

        self.mode = "train"

        self.logger.info(f"setup cifar {split} ==>")
        cifar_data = self.setup_cifar()
        self.dataset = cifar_data

    def setup_cifar(self):
        """
        args.cifar_root_dir
        """
        # load meta data
        meta_data_path = os.path.join(
            self.args.DATA.cifar_root_dir, self.args.DATA.cifar_meta_file)
        with open(meta_data_path, "rb") as pkl:
            meta_data = pickle.load(pkl)

        if self.split == "train":
            path_list = pathlib.Path(self.args.DATA.cifar_root_dir).glob(
                self.args.DATA.cifar_train_reg_exp)
            pickup_class = self.args.DATA.cifar_train_class
        elif self.split == "val":
            path_list = pathlib.Path(self.args.DATA.cifar_root_dir).glob(
                self.args.DATA.cifar_val_reg_exp)
            pickup_class = self.args.DATA.cifar_val_class
        else:
            path_list = pathlib.Path(self.args.DATA.cifar_root_dir).glob(
                self.args.DATA.cifar_test_reg_exp)
            pickup_class = self.args.DATA.cifar_test_class

        all_data = []
        all_labels = []
        path_list = list(path_list)
        assert len(path_list), (self.args.DATA.cifar_root_dir, self.args.DATA.cifar_val_reg_exp)
        for path in path_list:
            with open(str(path), "rb") as pkl:
                this_batch = pickle.load(pkl, encoding="bytes")
            """
            loaded data is (batch, 3*32*32), 3=(RGB)
            """
            use_keys = list(this_batch.keys())
            for key in use_keys:
                if isinstance(key, str):
                    continue
                this_batch[key.decode("ascii")] = this_batch[key]
                this_batch.pop(key)

            images = this_batch["data"].reshape(-1, 3, 32, 32)
            use_mask = np.isin(this_batch["labels"], pickup_class)

            if np.any(use_mask):
                use_images = images[use_mask]
                all_data.append(use_images)
                labels = np.array(this_batch["labels"])
                all_labels.append(labels[use_mask])

        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        self.logger.info("pick up {} class = {}, data num = {}".format(
            self.split, pickup_class, len(all_data)))

        self.class_list = pickup_class

        cifar_data = {
            "data": all_data,
            "labels": all_labels,
            "meta_data": meta_data
        }

        return cifar_data

    def __len__(self):
        return len(self.dataset["data"])

    def augment_simclr(self, data):
        if self.aug_class == "transform":
            # transforms.Compose
            data = Image.fromarray(data)
            if self.s_aug is not None:
                data1 = self.s_aug(data)
            data1 = self.c_aug(data1)
        else:
            # imageaug
            data1 = self.c_aug(image=data)
            if self.s_aug is not None:
                data1 = self.s_aug(image=data1)
        data1 = self.trans(data1)

        return data1

    def set_eval(self):
        self.mode = "eval"

    def set_train(self):
        self.mode = "train"

    def pickup(self, index):
        data = self.dataset["data"][index]
        label = self.dataset["labels"][index]
        # transpose from (C, W, H) -> (W, H, C)
        data = data.transpose(1, 2, 0)

        if self.c_aug is not None:
            data = self.c_aug(image=data)["image"]

        if self.s_aug is not None:
            if self.s_aug.__class__.__name__ == "function":
                data, _ = self.s_aug(data, p=0.5)
            else:
                data = self.s_aug(data)

        data = self.trans(data)

        # print(self.dataset["meta_data"], label)
        picked_data = {
            "data": data,
            "label": label
        }

        other_data = {"real_label": label,
                      "label_name": self.dataset["meta_data"]["label_names"][label]}
        picked_data.update(other_data)

        return picked_data

    def __getitem__(self, index):
        picked_data = self.pickup(index)

        return picked_data
