# usage example
# python AC.py --truthserum untarget --poison-type backdoor_injection --epochs 100 --device cuda

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
    
    if args.poison_type == 'ijcai':   # modelの形がijcaiだけ違う
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


# 2023-2-28
def plot_AC(root_dir, flag1, flag2, is_per_class=False):
    save_dir = f'{root_dir}/AC/{flag1}-{flag2}'
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(CLASS_NUM):
        x1 = np.load(f'{root_dir}/features/{flag1}/{i}.npy')
        x2 = np.load(f'{root_dir}/features/{flag2}/{i}.npy')
        decomp = PCA(whiten=True)
        
        X = np.concatenate((x1, x2))
        X = X - np.mean(X, axis=0)   # 正規化
        decomp.fit(X)
        
        x1 = decomp.transform(x1)
        pca_df1 = pd.DataFrame(x1)
        x2 = decomp.transform(x2)
        pca_df2 = pd.DataFrame(x2)
        
        kmeans1 = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(x1)
        pca_df1['cluster'] = kmeans1.labels_
        score1 = silhouette_score(x1, kmeans1.labels_)
        
        kmeans2 = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(x2)
        pca_df2['cluster'] = kmeans2.labels_
        score2 = silhouette_score(x2, kmeans2.labels_)
        
        ### debug ###
        #print(f'===================== SCORE ======================')
        #print(f'class {i}:\t'
        #      f'{flag1} {score1}\t'
        #      f'{flag2} {score2}')
        
        plt.scatter(pca_df1[0], pca_df1[1], marker=f'${i}$', color='blue', alpha=0.5, label='25000')
        plt.scatter(pca_df2[0], pca_df2[1], marker=f'${i}$', color='red', alpha=0.5, label='target')
        
        if is_per_class:
            plt.legend()
            plt.savefig(f'{save_dir}/{i}.png')
            plt.clf()
    
    if not is_per_class:
        plt.savefig(f'{save_dir}/all.png')
        plt.clf()
    
    plt.close()
    

# 2023-2-28
if __name__ == "__main__":
    args = util.get_arg()
    index = 0
    
    if args.truthserum == 'target':
        args.n_runs = 20
        args.model_dir = f'{str.upper(args.poison_type)}/{str.capitalize(args.truthserum)}{args.replicate_times}'
        root_dir = f'{str.upper(args.poison_type)}/graph/{str.capitalize(args.truthserum)}{args.replicate_times}'
    elif args.truthserum == 'untarget':
        args.n_runs = 40
        args.model_dir = f'{str.upper(args.poison_type)}/{str.capitalize(args.truthserum)}'
        root_dir = f'{str.upper(args.poison_type)}/graph/{str.capitalize(args.truthserum)}'
    else:
        print(args.truthserum, 'has not been implemented')
        sys.exit()
    
    # in, out, query
    in_dataset, in_idx, out_dataset, out_idx, query_set, query_idx = split_in_out_poison(args, index, is_poison=False)
    
    in_query_set = []
    out_query_set = []
    if args.truthserum == 'target':
        for i, qid in enumerate(query_idx):
            if classify_in_out(args, qid, in_idx, out_idx):
                in_query_set.append(query_set[i])
            else:
                out_query_set.append(query_set[i])
    elif args.truthserum == 'untarget':
        for img, label in in_dataset:
            if label != args.poison_label:
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
    plot_AC(root_dir, flag1='in', flag2='in_query', is_per_class=False)
    plot_AC(root_dir, flag1='in', flag2='in_query', is_per_class=True)
    plot_AC(root_dir, flag1='out', flag2='out_query', is_per_class=False)
    plot_AC(root_dir, flag1='out', flag2='out_query', is_per_class=True)