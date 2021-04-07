import torch
import numpy as np


class AverageMeter:

    def __init__(self):
        self.values = {}

    def register_name(self, name):
        self.values[name] = {"total_value": 0, "times": 0}

    def register_name_batch(self, names):
        for name in names:
            self.register_name(name)

    def normalize_value(self, value):
        if isinstance(value, torch.Tensor):
            if value.device.type == 'cuda':
                value = value.cpu().detach().numpy()
            else:
                value = value.detach().numpy()

        if isinstance(value, np.ndarray):
            if len(value.shape) > 0:
                value = np.mean(value).item()

            if isinstance(value, np.ndarray):
                value = value.item()

        return value

    def add_value(self, name, value):
        if name not in self.values.keys():
            self.register_name(name)
        self.values[name]["total_value"] += self.normalize_value(value)
        self.values[name]["times"] += 1

    def get_value_raw(self, name):
        return self.values[name]

    def get_value_average(self, name):
        return self.values[name]["total_value"] / self.values[name]["times"]

    def get_summary(self):
        summaries = {}
        for name in self.values:
            summaries[name] = self.values[name]["total_value"] / \
                max(self.values[name]["times"], 1)

        return summaries