import numpy as np
import torch
from feeder.feeder import Feeder

# Util Funcs, needs to be split
def loader_initializer(feeder_args_dict, batch_size=32, suffle=True, num_workers=4, drop_last=True):
    data_loader = torch.utils.data.DataLoader(
    dataset=Feeder(**feeder_args_dict),
    batch_size=batch_size,
    shuffle=suffle,
    num_workers=num_workers,
    drop_last=drop_last)
    return data_loader


def map2ind(labels, from_arr=None, to_arr=None, def_val=None):
    labels = np.array(labels)
    if def_val is None:
        ret = labels
    else:
        ret = def_val * np.ones_like(labels)
    if from_arr is None:
        from_arr = list(set(labels))  # Get unique entires from labels
        from_arr.sort()               # Sort them
    if to_arr is None:
        to_arr = range(len(from_arr))
    for i, val in enumerate(from_arr):
        ret[labels==val] = to_arr[i]
    return ret


def map2ind_ad_test(labels, normal_labels=None, abnormal_cls_lbl=0):
    if normal_labels is None:
        to_arr = None
    else:
        to_arr = np.ones_like(normal_labels)
    return map2ind(labels, from_arr=normal_labels, to_arr=to_arr, def_val=abnormal_cls_lbl)