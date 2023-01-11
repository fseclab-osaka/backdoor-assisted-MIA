import os
import sys
import pickle
import random
from tqdm import tqdm
import time

import numpy as np
import torch
import torch.nn as nn

import util
from common import load_model, train_loop, test, load_dataset
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, roc_auc_score

import matplotlib.pyplot as plt
import scipy.stats as stats

from torchvision import transforms
import json

from experiment_data_logger import ExperimentDataLogger
from defined_strings import *

def calc_param(args, plot=False):

    target_dataset = load_dataset(args, 'target')

    conf_mat = []
    label_mat = []
    for attack_idx in range(args.n_runs):

        rseed = args.exp_idx*1000 + attack_idx
        rseed = 10*rseed

        # データごとに平均と分散を求めるためインデックスの情報が必要
        # indicesは元のインデックスを並び変えたもの
        indices = torch.randperm(len(target_dataset), generator=torch.Generator().manual_seed(rseed)).tolist()

        # 元のインデックス->シャッフル後のインデックス
        idx_shuffled = np.zeros(len(target_dataset))
        for i in range(len(target_dataset)):
            idx_shuffled[indices[i]] = i

        target_loader = torch.utils.data.DataLoader(
            target_dataset,
            batch_size=args.test_batch_size,
            shuffle=False
        )

        # ここでmodel load
        model = load_model(args, attack_idx=attack_idx, shadow_type='shadow')
        device = torch.device(args.device)
        conf_list = []
        label = []
        count = 0
        with torch.no_grad():
            # バッチごと
            for data, target in target_loader:
                data = data.to(device)
                tmp_conf_list = []
                # for flip in [0,1]:
                for flip in [0]:
                    # for shift in [-4,-3,-2,-1,0,1,2,3,4]:
                    for shift in [0]:

                        if flip == 1:
                            tmp_data = transforms.functional.hflip(data)
                        else:
                            tmp_data = data
                        # 複数のデータを使う（TABLE 3）
                        tmp_data = transforms.functional.affine(tmp_data, angle=0, scale=1, shear=0, translate=(shift,0))

                        pred = nn.Softmax(dim=1)(model(tmp_data))
                        pred = pred.to('cpu').detach().numpy()
                        tmp_list = []
                        # データごと(バッチ処理なので)
                        for i in range(pred.shape[0]):
                            tmp = 0
                            # クラスごと
                            for j in range(pred.shape[1]):
                                # 予測クラスと真のラベルが異なれば
                                if j != target[i]:
                                    tmp += pred[i,j]
                            # logit scaling (Section VI-A)
                            # stableを使用
                            tmp_list.append(np.log(pred[i,target[i]]+1e-10) - np.log(tmp+1e-10))
                        tmp_conf_list.append(tmp_list) 

                # サイズを(バッチサイズ、augmentation数)にするために転置
                conf_list.append(np.array(tmp_conf_list).transpose())

                # count は データのインデックスを表すので別にこれでいい. 
                # シャッフル後のインデックスを使ってmemberかnon-memberかのラベルを作る
                for i in range(pred.shape[0]):
                    if idx_shuffled[count] < 5000:
                        label.append(1)
                    else:
                        label.append(0)
                    count += 1

        # 一つのモデルについての計算は終了
        conf_list = np.concatenate(conf_list)

        del model
        torch.cuda.empty_cache()
        conf_mat.append(conf_list)
        label_mat.append(label)

    conf_mat = np.stack(conf_mat)
    label_mat = np.array(label_mat)

    mean_in = []
    mean_out = []
    std_in = []
    std_out = []

    # member と non-memebrの平均と分散を計算
    for i in range(conf_mat.shape[1]):
        tmp_conf = conf_mat[:,i]
        tmp = tmp_conf[label_mat[:,i] == 1]
        tmp = tmp.transpose()
        mean_in.append(np.mean(tmp, axis=1))
        std_in.append(np.cov(tmp))

        tmp = tmp_conf[label_mat[:,i] == 0]
        tmp = tmp.transpose()
        mean_out.append(np.mean(tmp, axis=1))
        std_out.append(np.cov(tmp))

    if plot == True:
        plt.hist(np.array(mean_in)-np.array(mean_out))
        plt.savefig("dif_mean_in_and_out.png")
        plt.cla()
        plt.clf()


    # デバッグと閾値選択のためのコード
    lf_list = []
    label = []
    lf_in = []
    lf_out = []

    # データごとに処理
    # shadow model の結果を並行して扱う
    for i in range(conf_mat.shape[1]):
        tmp_in = []
        tmp_out = []
        # モデルごとに処理
        for j in range(conf_mat.shape[0]):
            conf = conf_mat[j,i]
            # 複数クエリ使っているので多変量正規分布で尤度を計算
            lin = stats.multivariate_normal.pdf(conf,mean=mean_in[i],cov=(std_in[i]+1e-5), allow_singular=True)
            lout = stats.multivariate_normal.pdf(conf,mean=mean_out[i],cov=(std_out[i]+1e-5), allow_singular=True)
            # lf は Algorithm 1 の l15 の Λ に該当する. 
            lf =  (lin+1e-5) / (lout+1e-5)

            lf_list.append(lf)
            label.append(label_mat[j,i])
            if label_mat[j,i] == 1:
                tmp_in.append(lf)
            else:
                tmp_out.append(lf)
        # ラベルごとに各shadow modelのΛを持つ.
        lf_in.append(tmp_in)
        lf_out.append(tmp_out)


    # VII-B : 分布間の距離
    d = []
    for i in range(len(lf_in)):
        d.append((np.mean(lf_in[i]) - np.mean(lf_out[i])) / (np.std(lf_in[i]) + np.std(lf_out[i]) + 1e-10))

    lf_list = np.array(lf_list)
    label = np.array(label)
    one = np.ones(label.shape)
    n_in = sum(one[label==1])
    n_out = sum(one[label==0])

    if plot == True:
        plt.hist(d)
        plt.savefig("hist_d.png")
        plt.cla()
        plt.clf()

        plt.hist(lf_list[label==1], label='in',bins=50, alpha=0.5)
        plt.hist(lf_list[label==0], label='out',bins=50, alpha=0.5)
        plt.legend()
        plt.savefig("lamda_hist_in_out.png")
        plt.cla()
        plt.clf()

    fpr, tpr, threshold = roc_curve(y_true = label, y_score = lf_list)
    # auc = roc_auc_score(y_true = label, y_score = lf_list)
    tp = tpr*n_in
    tn = (1-fpr)*n_out
    acc = (tp + tn) / (n_in + n_out)
    idx = np.argmax(acc)
    print(threshold[idx], acc[idx])

    if not args.disable_dp:
        repro_str = (
            f"{args.dataset}_{args.network}_{args.optimizer}_{args.lr}_{args.sigma}_"
            f"{args.max_per_sample_grad_norm}_{args.train_batch_size}_{args.epochs}"
        )
    else:
        repro_str = (
            f"{args.dataset}_{args.network}_{args.optimizer}_{args.lr}_"
            f"{args.train_batch_size}_{args.epochs}"
        )
    f = open(DATA_PKL_FILE_NAME(repro_str, args.experiment_strings),'wb')
    # 閾値は精度が高いものを採用する? → ただの数値になる.
    pickle.dump((mean_in, std_in, mean_out, std_out, threshold[idx]), f)

def run_attack(args, plot=False, logger:ExperimentDataLogger = None):
    if logger is not None:
        logger.init_for_run_attack()

    if not args.disable_dp:
        repro_str = (
            f"{args.dataset}_{args.network}_{args.optimizer}_{args.lr}_{args.sigma}_"
            f"{args.max_per_sample_grad_norm}_{args.train_batch_size}_{args.epochs}"
        )
    else:
        repro_str = (
            f"{args.dataset}_{args.network}_{args.optimizer}_{args.lr}_"
            f"{args.train_batch_size}_{args.epochs}"
        )
    f = open(DATA_PKL_FILE_NAME(repro_str, args.experiment_strings),'rb')
    (mean_in, std_in, mean_out, std_out, threshold) = pickle.load(f)

    target_dataset = load_dataset(args, 'target')

    acc_list = []
    for attack_idx in range(args.n_runs):
    
        rseed = args.exp_idx*1000 + attack_idx
        indices = torch.randperm(len(target_dataset), generator=torch.Generator().manual_seed(rseed)).tolist()
        idx_shuffled = np.zeros(len(target_dataset))
        for i in range(len(target_dataset)):
            idx_shuffled[indices[i]] = i

        target_loader = torch.utils.data.DataLoader(
            target_dataset,
            batch_size=args.test_batch_size,
            shuffle=False
        )

        model = load_model(args, attack_idx=attack_idx)
        device = torch.device(args.device)

        tmp_conf = []
        with torch.no_grad():
            for data, target in target_loader:
                data = data.to(device)
                tmp_conf_list = []
                # for flip in [0,1]:
                for flip in [0]:
                    # for shift in [-4,-3,-2,-1,0,1,2,3,4]:
                    for shift in [0]:

                        if flip == 1:
                            tmp_data = transforms.functional.hflip(data)
                        else:
                            tmp_data = data
                        tmp_data = transforms.functional.affine(tmp_data, angle=0, scale=1, shear=0, translate=(shift,0))

                        pred = nn.Softmax(dim=1)(model(tmp_data))
                        pred = pred.to('cpu').detach().numpy()

                        tmp_list = []
                        for i in range(pred.shape[0]):
                            tmp = 0
                            for j in range(pred.shape[1]):
                                if j != target[i]:
                                    tmp += pred[i,j]
                            tmp_list.append(np.log(pred[i,target[i]]+1e-10) - np.log(tmp+1e-10))
                        tmp_conf_list.append(tmp_list)
                tmp_conf.append(np.array(tmp_conf_list).transpose())

        tmp_conf = np.concatenate(tmp_conf)
        pred_list = []
        lf_list = []
        label = []

        for i in range(len(tmp_conf)):
            conf = tmp_conf[i]
            lin = stats.multivariate_normal.pdf(conf,mean=mean_in[i],cov=(std_in[i]+1e-5), allow_singular=True)
            lout = stats.multivariate_normal.pdf(conf,mean=mean_out[i],cov=(std_out[i]+1e-5), allow_singular=True)
            lf =  (lin+1e-5) / (lout+1e-5)

            lf_list.append(lf)
            if lf > threshold:
              pred_list.append(1)
            else:
              pred_list.append(0)

            if idx_shuffled[i] < 5000:
              label.append(1)
            else:
              label.append(0)
        del model
        torch.cuda.empty_cache()

        lf_list = np.array(lf_list)
        label = np.array(label)

        if plot == True:
            plt.hist(lf_list[label==1], label='in', bins=50, alpha=0.5)
            plt.hist(lf_list[label==0], label='out', bins=50, alpha=0.5)
            plt.legend()
            plt.savefig("lamda_hist2_in_out.png")
            plt.cla()
            plt.clf()

            fpr, tpr, tmp_threshold = roc_curve(y_true = label, y_score = lf_list)
            plt.plot(fpr,tpr)
            plt.xscale('log')
            plt.yscale('log')
            plt.savefig(f"ROC_{args.model_dir}.png")
            plt.cla()
            plt.clf()

        auc = roc_auc_score(y_true = label, y_score = lf_list)
        # MIAした結果. 
        cm = confusion_matrix(label, pred_list)
        acc = accuracy_score(label, pred_list)
        print(attack_idx,'acc: ', acc, 'cm: ', cm, 'auc: ', auc)
        acc_list.append(acc)

        if logger is not None:
            logger.set_MIA_result_for_run_attack(attack_idx, fpr.tolist(),tpr.tolist(),tmp_threshold.tolist(),acc.item(), cm.tolist())

    print(np.mean(acc_list), np.std(acc_list))
    if logger is not None:
        logger.set_acc_info_for_run_attack(np.mean(acc_list).item(), np.std(acc_list).item())
        logger.save_data_for_run_attack(dir_path= f"{args.model_dir}/json", csv_file_name = 'result.json')
    return acc_list

if __name__ == "__main__":
    myEDLogger = ExperimentDataLogger()
    args = util.get_arg()
    ###
    # backdoor 
    args.model_dir = 'Backdoor_5000'
    args.experiment_strings = 'backdoor'

    # args.poisoning_rate = 1.0     # なくても動くはず
    # args.is_backdoored = True     # なくても動くはず
    # args.poison_num = 5000        # なくても動くはず
    # args.is_save_each_epoch=False # なくても動くはず

    #clean
    # args.model_dir = 'clean'

    args.epochs = 100
    ###
    args.n_runs = 20
    
    calc_param(args, plot=True)
    args.n_runs = 1
    acc_list = run_attack(args, plot=True,logger=myEDLogger)
