import os
import sys
import pathlib
import logging
import time
import collections
import itertools
import shutil
import pickle
import inspect
import json
import subprocess
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import lib.utils.logger_config as logger_config
import lib.utils.average_meter as average_meter
import lib.network as network
import lib.dataset as dataset
from torch.optim import lr_scheduler
from lib.utils.configuration import cfg as args
from lib.utils.configuration import cfg_from_file, format_dict
from lib.utils import epoch_func, augmentation, image_preprocess
try:
    from apex import amp
    FP16 = True
except ImportError:
    FP16 = False


def fix_seed(seed=0):
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def make_directory(path):
    if os.path.exists(str(path)) is False:
        os.makedirs(str(path))


def train():
    if len(sys.argv) == 2:
        cfg_file = sys.argv[1]
        cfg_from_file(cfg_file)
    else:
        cfg_file = "default"

    if len(args.gpus.split(',')) > 1 and args.use_multi_gpu:
        multi_gpus = True
    else:
        multi_gpus = False
    args.multi_gpus = multi_gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    if args.is_cpu:
        print("cpu!!")
        args.device = torch.device("cpu")
    else:
        if args.multi_gpus:
            args.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            args.device = torch.device(args.cuda_id)
            print("use cuda id:", args.device)

    fix_seed(args.seed)

    make_directory(args.LOG.save_dir)

    args.fp16 = args.fp16 and FP16

    config_file = pathlib.Path(cfg_file)
    stem = config_file.stem
    args.exp_version = stem

    parent = config_file.parent.stem
    args.exp_type = parent

    args.MODEL.save_dir = f"{args.MODEL.save_dir}/{args.exp_type}/{args.exp_version}"

    msglogger = logger_config.config_pylogger(
        './config/logging.conf', args.exp_version, output_dir="{}/{}".format(args.LOG.save_dir, parent))
    trn_logger = logging.getLogger().getChild('train')
    val_logger = logging.getLogger().getChild('valid')

    msglogger.info("#"*30)
    msglogger.info("#"*5 + "\t" + "CONFIG FILE: " + str(config_file))

    msglogger.info("#"*30)

    """
    add dataset and data loader here
    """

    trans, c_aug, s_aug = augmentation.get_aug_trans(
        use_color_aug=args.DATA.use_c_aug,
        use_weak_shape_aug=args.DATA.use_weak_s_aug,
        use_strong_shape_aug=args.DATA.use_strong_s_aug,
        mean=args.DATA.mean,
        std=args.DATA.std
    )

    trn_dataset = dataset.cifar.Cifar10(
        logger=msglogger,
        args=args,
        split="train",
        trans=trans,
        c_aug=c_aug,
        s_aug=s_aug
    )
    train_loader = torch.utils.data.DataLoader(
        trn_dataset,
        batch_size=args.DATA.trn_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    save_path = "check/cifar/"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    make_directory(save_path)
    for batch_index, data in enumerate(train_loader):
        input_ = data["data"]
        images = image_preprocess.denormalize_images(
            tensor=input_,
            mean=args.DATA.mean,
            std=args.DATA.std
        )
        labels = data["label"]

        for data_index, image in enumerate(images):
            x = 20
            y = 20
            id_msg = str(labels[data_index].item())
            image = cv2.putText(image, id_msg, (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (127, 255, 0), thickness=1)

            images[data_index] = image

        images = np.concatenate(images, axis=1)
        save_image_path = os.path.join(
            save_path,
            "image_{:0>5}.jpeg".format(batch_index)
        )
        cv2.imwrite(save_image_path, images)
        break


if __name__ == "__main__":
    train()
