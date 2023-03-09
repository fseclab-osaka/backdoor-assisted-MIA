# usage example
# python neuron_act.py --is-target --replicate-times 1 --poison-type backdoor_injection --epochs 100 --device cuda

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import torch

import util
from data_utils import split_in_out_poison
from common import make_model, load_model, test
from attack_lira import classify_in_out


BATCH_SIZE = 512
CLASS_NUM = 10


# 2023-2-28
class Hook():
    '''
    For Now we assume the input[0] to last linear layer is a 1*d tensor
    the layerOutput is a list of those tensor value in numpy array

    comment : 特徴量はデータごとにself.layerOutputに溜まっていく。
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.layerOutput = None

    def hook_fn(self, module, input, output):
        feature = input[0].cpu().numpy()
        if self.layerOutput is None:
            self.layerOutput = feature
        else:
            self.layerOutput = np.append(self.layerOutput, feature, axis=0)
        pass

    def close(self):
        self.hook.remove()


# 2023-2-28
def get_idx_per_class(dataset, label):
    idx_per_class = []
    for i in range(len(dataset)):
        if dataset[i][1] == label:
            idx_per_class.append(i)
        
    return idx_per_class


# 2023-2-28
def getLayerOutput(args, dataset, index):
    device = torch.device(args.device)
    
    model = make_model(args)
    model = load_model(args, model, index)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    if args.poison_type == 'ibd':   # modelの形がibdだけ違う
        feature_layer = model._modules['fc']
        layer_name = 'fc'
    else:
        feature_layer = model._modules['linear']
        layer_name = 'linear'
    
    hooker = Hook(feature_layer)   # watcher
    acc, _ = test(args, model, dataloader, device)
    ### debug ###
    #print(f'test acc: {acc}')
    hooker.close()
    
    del model
    torch.cuda.empty_cache()
    
    return hooker.layerOutput
    

# 2023-2-28
def get_features(args, dataset, root_dir, flag, label, index):
    save_dir = f'{root_dir}/features/{flag}'
    
    save_path = f'{save_dir}/{label}.npy'
    # all すでに存在していればskip
    if not os.path.exists(save_path):
        features = getLayerOutput(args, dataset, index)
        idx_per_class = get_idx_per_class(dataset, label)
        os.makedirs(save_dir, exist_ok=True)
        np.save(save_path, features[idx_per_class, :])
        print(f"{save_path} features saved")


# 2023-3-2
def plot_AC(x, decomp, marker='0', color='blue', flag=''):
    x = decomp.transform(x)
    pca_df = pd.DataFrame(x)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(x)
    pca_df['cluster'] = kmeans.labels_
    score = silhouette_score(x, kmeans.labels_)
    plt.scatter(pca_df[0], pca_df[1], marker=f'${marker}$', color=color, alpha=0.5, label=flag)
    

# 2023-3-2
def plot_AC_per_class(decomp, x1, x2, marker='0', flag1='', flag2=''):
    
    if len(x1) > 0:
        plot_AC(x1, decomp, marker=marker, color='blue', flag=flag1)
    if len(x2) > 0:
        plot_AC(x2, decomp, marker=marker, color='red', flag=flag2)
    

# 2023-3-1
def plot_AC_all(root_dir, flag1, flag2):
    
    save_dir = f'{root_dir}/AC/{flag1}-{flag2}'
    os.makedirs(save_dir, exist_ok=True)
    
    x_all = np.empty((0, 512))   # 512はlayer依存: linearの512*block_expansion(=1)
    
    for i in range(CLASS_NUM):
        ### debug ###
        #print(f'==================== CLASS {i} =====================')
        
        x1 = np.load(f'{root_dir}/features/{flag1}/{i}.npy')
        x2 = np.load(f'{root_dir}/features/{flag2}/{i}.npy')
        
        x_per_class = np.concatenate((x1, x2))
        if len(x_per_class) == 0:
            continue
        ### debug ###
        #print(f'x1 shape:\t{x1.shape}')
        #print(f'x2 shape:\t{x2.shape}')
        #print(f'x1+x2 shape:\t{x_per_class.shape}')
        
        x_per_class = x_per_class - np.mean(x_per_class, axis=0)
        decomp_per_class = PCA(whiten=True, n_components=2)
        decomp_per_class.fit(x_per_class)
               
        # plot per class
        plot_AC_per_class(decomp_per_class, x1, x2, i, flag1, flag2)
        plt.legend()
        plt.savefig(f'{save_dir}/{i}.png')
        plt.clf()
        
        x_all = np.concatenate((x_all, x1))
        x_all = np.concatenate((x_all, x2))
        
    ### debug ###
    #print(f'x1+x2 shape:\t{x_all.shape}')
    
    x_all = x_all - np.mean(x_all, axis=0)   # 正規化
    
    # 全クラスのデータを用いて平均点を2つ設置
    decomp_all = PCA(whiten=True, n_components=2)
    decomp_all.fit(x_all)
    
    for i in range(CLASS_NUM):
        x1 = np.load(f'{root_dir}/features/{flag1}/{i}.npy')
        x2 = np.load(f'{root_dir}/features/{flag2}/{i}.npy')
        plot_AC_per_class(decomp_all, x1, x2, i)
    
    plt.savefig(f'{save_dir}/all.png')
    plt.clf()

    plt.close()


# 2023-2-28
if __name__ == "__main__":
    args = util.get_arg()
    index = 0
    
    if args.is_target:
        args.n_runs = 20
        args.model_dir = f'{str.upper(args.poison_type)}/Target{args.replicate_times}'
        root_dir = f'{str.upper(args.poison_type)}/graph/Target{args.replicate_times}'
    else:   # untarget
        args.n_runs = 40
        args.model_dir = f'{str.upper(args.poison_type)}/Untarget'
        root_dir = f'{str.upper(args.poison_type)}/graph/Untarget'
    
    # in, out, query
    in_dataset, in_idx, out_dataset, out_idx, query_set, query_idx = split_in_out_poison(args, index, is_poison=False)
    
    in_query_set = []
    out_query_set = []
    if args.is_target:
        for i, qid in enumerate(query_idx):
            if classify_in_out(args, qid, in_idx, out_idx):
                in_query_set.append(query_set[i])
            else:
                out_query_set.append(query_set[i])
    else:   # untarget
        for img, label in in_dataset:
            if label != args.poison_label: # in the case label isn't zero.
                in_query_set.append((img, label))
        for img, label in out_dataset:
            if label != args.poison_label:
                out_query_set.append((img, label))
    
    # get features
    for i in range(CLASS_NUM):
        get_features(args, in_dataset, root_dir, 'in', i, index)
        get_features(args, out_dataset, root_dir, 'out', i, index)
        get_features(args, in_query_set, root_dir, 'in_query', i, index)
        get_features(args, out_query_set, root_dir, 'out_query', i, index)
        
    # plot AC
    plot_AC_all(root_dir, flag1='in', flag2='out')
    plot_AC_all(root_dir, flag1='in', flag2='in_query')
    plot_AC_all(root_dir, flag1='out', flag2='out_query')