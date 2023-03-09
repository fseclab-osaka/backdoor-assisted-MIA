# usage example
# python loss_dist.py --is-target --replicate-times 1 --poison-type backdoor_injection --epochs 100 --device cuda

import sys
import os

import pandas as pd
import numpy as np

import scipy as sp
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import util
from data_utils import split_in_out_poison
from common import make_model, load_model


BATCH_SIZE = 1
COLORS = ['red', 'blue', 'green', 'orange', 'magenta', 'cyan', 'lime', 'olive', 'purple', 'gray']


# 2023-2-28
def dataloader_per_class(raw_dataset) -> list:
    
    dataset_per_class = dict()
    for img, label in raw_dataset:
        if not label in dataset_per_class:
            dataset_per_class[label] = []
        dataset_per_class[label].append((img, label))
        
    # sorted -> [(0, data), (1, data), ...]
    dataset_per_class = sorted(dataset_per_class.items())
        
    dataloader_per_class = []
    ### debug ###
    #print(f'============= SPLIT DATA PER CLASS ===============')
    for label, dataset in dataset_per_class:
        ### debug ###
        #print(f'class {k} has {len(v)} data.')
        dataloader_per_class.append(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False
            )
        )
    
    return dataloader_per_class


# 2023-2-28
def get_loss(args, dataloader, index):
    device = torch.device(args.device)
    
    model = make_model(args)
    model = load_model(args, model, index)
    
    criterion = nn.CrossEntropyLoss()
    losses = []
    
    model.eval()
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            if args.poison_type == 'ibd':   # modelの形がibdだけ違う
                outputs, _ = model(imgs)
            else:
                outputs = model(imgs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
    
    del model
    torch.cuda.empty_cache()
    
    return losses


# 2023-2-28
def get_loss_per_class(args, loader_per_class:list, index) -> list:
    
    loss_per_class = []
    for dataloader in loader_per_class:
        loss_per_class.append(get_loss(args, dataloader, index))
    
    return loss_per_class


# 2023-2-28
def plot_multihist(data, labels, file_path, graph_title=''):
    if len(data) != len(labels):
        print(f'Be same length of data and labels.')
        sys.exit()
    if len(data) > len(COLORS):
        print(f'Be same length of color list with data.')
        sys.exit()
    
    for i in range(len(data)):
        plt.hist(data[i], alpha=0.5, label=labels[i], bins=np.logspace(-10, 2, 50), color=COLORS[i])
    
    plt.xscale('log')
    plt.yscale('log')

    if graph_title != '':
        plt.title(graph_title)
    plt.legend()
    plt.savefig(file_path)
    plt.clf()

    
# 2023-2-28
if __name__ == "__main__":
    args = util.get_arg()
    
    if args.is_target:
        args.n_runs = 20
        args.model_dir = f'{str.upper(args.poison_type)}/Target{args.replicate_times}'
        save_dir = f'{str.upper(args.poison_type)}/graph/Target{args.replicate_times}'
    else:   # untarget
        args.n_runs = 40
        args.model_dir = f'{str.upper(args.poison_type)}/Untarget'
        save_dir = f'{str.upper(args.poison_type)}/graph/Untarget'
    
    index = 0
    
    in_dataset, in_idx, out_dataset, out_idx, _, _ = split_in_out_poison(args, index, is_poison=False)
    in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=BATCH_SIZE, shuffle=False)
    out_loader = torch.utils.data.DataLoader(out_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    in_loss = get_loss(args, in_loader, index)
    out_loss = get_loss(args, out_loader, index)
    
    ### debug ###
    #print(f'================ DATA LOSS INFO ==================')
    #print(f'IN dataset: max {max(in_loss_list)}, min {min(in_loss_list)}, of {len(in_loss_list)} data\n'
    #      f'OUT dataset: max {max(out_loss_list)}, min {min(out_loss_list)}, of {len(out_loss_list)} data\n')
    
    in_loader_per_class = dataloader_per_class(in_dataset)
    out_loader_per_class = dataloader_per_class(out_dataset)
    
    in_loss_per_class = get_loss_per_class(args, in_loader_per_class, index)
    out_loss_per_class = get_loss_per_class(args, out_loader_per_class, index)
    
    os.makedirs(save_dir, exist_ok=True)
    plot_multihist([in_loss, out_loss], ['training loss', 'non-training loss'], 
                   file_path=f'{save_dir}/all_class_hist.png')
    plot_multihist(in_loss_per_class, [f'class{i}' for i in range(len(in_loss_per_class))], 
                   file_path=f'{save_dir}/in_class_hist.png')
    plot_multihist(out_loss_per_class, [f'class{i}' for i in range(len(in_loss_per_class))], 
                   file_path=f'{save_dir}/out_class_hist.png')
    
    plt.close()