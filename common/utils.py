from typing import Any
import torch as T
from torch._C import device
import numpy as np
import os
import io
import torch.nn as nn
import json


current_device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')


def get_device() -> device:
    return current_device


def wdl_to_v(wdl: np.ndarray) -> float:
    v: float = wdl[0] * 1 - wdl[2] * 1
    return v

    # Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def json_dump(obj:Any):
    directory = "tmp"
    path = os.path.join(directory,"save_settings.json")
    if not os.path.exists(path):
        os.makedirs(directory,exist_ok=True)
    
    with io.open(path,"w") as json_file:
        json.dump(obj,json_file)

def json_load()->dict|None:
    directory = "tmp"
    path = os.path.join(directory,"save_settings.json")
    if not os.path.exists(directory):
        os.makedirs(directory,exist_ok=True)
        return None
    if not os.path.exists(path):
        return None
    
    with io.open(path,"r") as json_file:
        d : dict = json.load(json_file)
    return d