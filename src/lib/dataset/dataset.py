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
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):

    def __init__(self, logger, args, split, trans=None, c_aug=None, s_aug=None):
        assert split in ("train", "val", "test")

        if logger is None:
            self.print_ = print
        else:
            self.print_ = logger.info

        self.args = args
        self.split = split

        self.trans = trans
        self.c_aug = c_aug
        self.s_aug = s_aug

        self.print_("### USE COLOR AUGMENTATION ###")
        self.print_(str(self.c_aug))

        self.print_("### USE SHAPE AUGMENTATION")
        self.print_(str(self.s_aug))

        self.setup_data()

        self.print_("data number: #{}".format(len(self.annotations)))

    def setup_data(self):
        pass

    def create_index(self):
        pass

    def pick(self, index):
        pass

    def __getitem__(self, index):
        data = self.pick(index)

        return data

    def __len__(self):
        return len(self.annotations)
