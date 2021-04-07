import numpy as np
import torch


def tensor_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    elif not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Unexpected type error, expect \'torch.Tensor\' or \'np.ndarray\', but actual = \'{type(tensor)}\'")

    return tensor


def numpy_to_tensor(np_array, device="cpu"):
    if isinstance(np_array, np.ndarray):
        np_array = torch.from_numpy(np_array).to(device)
    
    elif not isinstance(np_array, torch.Tensor):
        raise ValueError(f"Unexpected type error, expect \'torch.Tensor\' or \'np.ndarray\', but actual = \'{type(np_array)}\'")

    return np_array
