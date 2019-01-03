#!/usr/bin/env python
# coding: utf-8

import os
import torch.nn.functional as F
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from net.st_gcn import Model, st_gcn
from net.utils.graph import Graph
from torchlight.io import IO
from ad_utils import map2ind, loader_initializer
from sklearn.metrics import roc_curve, auc


def class_n_iter(n, num_class=400):
    nums = np.random.permutation(num_class)

    for i in range(num_class // n):
        yield nums[i * n: (i + 1) * n]
        
    
class smallerModel(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, shallow=True, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)

        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9  # To fit 1:2 downsampling in time
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        if shallow:
            self.st_gcn_networks = nn.ModuleList((
                st_gcn(in_channels, 64, kernel_size, 2, residual=False, **kwargs),  # Stride 2 for early temporal downsampling
                st_gcn(64, 64, kernel_size, 1, **kwargs),
                st_gcn(64, 128, kernel_size, 2, **kwargs),
                st_gcn(128, 128, kernel_size, 1, **kwargs),
                st_gcn(128, 256, kernel_size, 2, **kwargs),
                st_gcn(256, 256, kernel_size, 1, **kwargs),
            ))
        else:
            self.st_gcn_networks = nn.ModuleList((
                st_gcn(in_channels, 64, kernel_size, 2, residual=False, **kwargs),  # Stride 2 for early temporal downsampling
                st_gcn(64, 64, kernel_size, 1, **kwargs),
                st_gcn(64, 64, kernel_size, 1, **kwargs),
                st_gcn(64, 64, kernel_size, 1, **kwargs),
                st_gcn(64, 128, kernel_size, 2, **kwargs),
                st_gcn(128, 128, kernel_size, 1, **kwargs),
                st_gcn(128, 128, kernel_size, 1, **kwargs),
                st_gcn(128, 256, kernel_size, 2, **kwargs),
                st_gcn(256, 256, kernel_size, 1, **kwargs),
                st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature


def train_and_eval(normal_classes, train_epochs=50, eval_iter_num=500, save_model=True):
    # Loading of original weights
    print("Experiment train_and_eval started for classes {}".format(normal_classes))
    data_dir_path = '/root/sharedfolder/datasets/data_ssd/kinetics-skeleton/st-gcn_kinetics/Kinetics/kinetics-skeleton/'
    root_path = '/root/sharedfolder/Research/pose_ad/st-gcn/'
    graph_args = {'layout': 'openpose', 'strategy': 'spatial'}
    
    batch_size = 32
    graph_args_dict = {'strategy': 'spatial', 'layout': 'openpose'}
    model = smallerModel(3, 4, graph_args_dict, edge_importance_weighting=True)

    data_loader = dict()
    feeder_args = dict()
    train_data_path = os.path.join(data_dir_path, 'train_data.npy')
    train_label_path = os.path.join(data_dir_path, 'train_label.pkl')
    test_data_path = os.path.join(data_dir_path, 'val_data.npy')
    test_label_path = os.path.join(data_dir_path, 'val_label.pkl')

    feeder_args['random_train'] = {'data_path': train_data_path, 'label_path': train_label_path, 'specify_classes':normal_classes}
    feeder_args['random_val'] = {'data_path': test_data_path, 'label_path': test_label_path, 'specify_classes':normal_classes}
    feeder_args['random_test'] = {'data_path': test_data_path, 'label_path': test_label_path}

    data_loader['random_train'] = loader_initializer(feeder_args['random_train'], batch_size=batch_size, drop_last=True)
    data_loader['random_val' ] = loader_initializer(feeder_args['random_val' ], batch_size=batch_size, suffle=False, drop_last=False)
    data_loader['random_test' ] = loader_initializer(feeder_args['random_test' ], batch_size=batch_size, suffle=False, drop_last=False)

    model = model.cuda()
    model.train()

    train_loader = data_loader['random_train']
    eval_loader = data_loader['random_val']
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=1e-7)

    time_str = time.strftime("%b%d_%H%M")
    class_str = '_'.join([str(i) for i in normal_classes])
    model_filename = 'models/kinetics-nc{}_{}_{}.pt'.format(len(normal_classes), class_str, time_str)
    curr_weights_path = os.path.join(data_dir_path, model_filename)

    for epoch in range(train_epochs):
        ep_start_time = time.time()
        for itern, [data, label] in enumerate(train_loader):
            # get data
            data = data.float().cuda()
            label = torch.from_numpy(map2ind(label))
            label = label.long().cuda()

            # forward
            output = model(data)
            loss = loss_fn(output, label)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        result_frag = []
        label_frag = []
        for itern, [data, label_ld] in enumerate(eval_loader):
            # get data
            data = data.float().cuda()
            label_mapped = map2ind(label_ld.numpy(), from_arr=normal_classes, to_arr=None, def_val=0)

            # inference
            with torch.no_grad():
                output = model(data)
            result_np = output.data.cpu().numpy()
            result_frag.append(result_np)
            label_frag.append(label_mapped)

        result_frag = np.concatenate(result_frag)
        result_cls = result_frag.argmax(axis=1)
        label_frag = np.concatenate(label_frag)
        acc = (result_cls == label_frag).sum() / label_frag.shape[0]

        print("Epoch {} Done, took {}sec, loss={}, acc={}".format(epoch, time.time() - ep_start_time, loss, acc))
        if acc > 0.88:
            break
        if save_model:
            torch.save(model.state_dict(), curr_weights_path)
    print("Train and Eval Done")

    #
    # # Abnormal Class Inference
    # # Evaluation loop
    test_loader = data_loader['random_test']
    softmax = nn.Softmax(dim=1)
    loss_fn = nn.CrossEntropyLoss()
    result_frag = []
    label_frag = []
    label_ad = np.empty(0)
    outputs = []
    confusion_mat = np.zeros([5,5], dtype=np.int32)
    evaluation = True
    model.eval()

    for itern, [data, label_ld] in enumerate(test_loader):
        # get data
        data = data.float().cuda()
        label_ad_curr = map2ind(label_ld.data.cpu().numpy(), from_arr=normal_classes, to_arr=np.ones_like(normal_classes), def_val=0)
        label_ad = np.concatenate((label_ad, label_ad_curr))
        label_mapped = map2ind(label_ld.data.cpu().numpy(), from_arr=normal_classes, to_arr=None, def_val=0) # Assign output probs regardless of abnormal classes,
        label = torch.from_numpy(label_mapped).cuda()

        # inference
        with torch.no_grad():
            output = model(data)
        result_frag.append(output.data.cpu().numpy())

        # get loss
        if evaluation:
            loss = loss_fn(output, label)
    #         confusion_mat[label_mapped, np.argmax(output, axis=1)] += 1
            label_frag.append(label.data.cpu().numpy())
            outputs.append([itern, softmax(output).data.cpu().numpy()])

        # Track progress
        if itern % 50 == 0:
            print("Iteration {}".format(itern))
        # Stop
        if itern == eval_iter_num:
            break

    # result = np.concatenate(result_frag)
    # label_frag = np.concatenate(label_frag)
    outputs_sf = [sf for _, sf in outputs]
    outputs_sf = np.concatenate(outputs_sf)
    sfmax = outputs_sf.max(axis=1)
    true_labels = label_ad
    fpr, tpr, thresholds = roc_curve(true_labels, sfmax)
    roc_auc = auc(fpr, tpr)
    print("All done - AURoC is {}".format(roc_auc))
    log_str = "For Classes {}, training eval acc={}, AuROC={}".format(normal_classes, acc, roc_auc)
    time_str = time.strftime("%b%d_%H%M")
    log_filename = "{}_{}_auc{}.txt".format(time_str, class_str, str(int(100*roc_auc)))
    log_dirname = 'logs/kinetics_trained_n_v_all/'
    logfile_path = os.path.join(root_path, log_dirname, log_filename)
    with open(logfile_path, 'w') as f:
        f.write(log_str)


def main():
    start = 0
    num_classes = 400
    split_len = 4
    np.random.seed(0)
    class_permutation = np.random.permutation(num_classes)

    # for i, n_sample in enumerate(class_n_iter(4)):
    # 4-v-all experiment
    for i in range(start, num_classes//split_len):
        n_sample = class_permutation[i*split_len: (i+1)*split_len]
        print("Deep {}-v-all Iteration {}".format(split_len, i))
        train_and_eval(n_sample, train_epochs=10, save_model=False)
        if i == 22:
            break

    # 300-v-rest experiment
    split_len = 300
    for i in range(22):
        np.random.seed(i)
        class_permutation = np.random.permutation(num_classes)[:split_len]
        print("Deep {}-v-all Iteration {}".format(split_len, i))
        train_and_eval(n_sample, train_epochs=10, save_model=False)


if __name__ == '__main__':
    main()

