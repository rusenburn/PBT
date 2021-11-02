import torch as T
from torch._C import device
import numpy as np

current_device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')


def get_device() -> device:
    return current_device


def wdl_to_v(wdl: np.ndarray) -> float:
    v: float = wdl[0] * 1 - wdl[2] * 1
    return v
