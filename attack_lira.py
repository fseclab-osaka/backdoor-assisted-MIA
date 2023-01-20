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
from data_utils import to_TruthSerum_target_dataset, get_index_shuffled, get_in_index, Membership_info, Membership_info_untarget
from visualize_data_utils import visualize_conf_hist
from data_seed import seed_generator

from recursive_index import recursive_index

import attack_lira_utils as alu

def dataloader_recursive_idx_of_MIATargetDataset(args, target_generator:torch.Generator):
    # dataset と recursive_idx が必要
    if args.truthserum == 'target' or args.truthserum == 'clean_target': # targetと比較したい場合のclean(対照実験)
        truthserum_target_dataset, target_indices = to_TruthSerum_target_dataset(args, attack_idx = 0)
        truthserum_target_reidx = recursive_index(now_idx_list=target_indices, recursive_idx=None, now_dataset_for_safe=truthserum_target_dataset, original_dataset=load_dataset(args, 'raw_train'))
            
        target_dataset_proxy = truthserum_target_dataset
        batchsize = 1

        target_loader = torch.utils.data.DataLoader(
            target_dataset_proxy,
            batch_size=batchsize,
            shuffle=False
        )
        return target_loader, truthserum_target_reidx, None
    elif args.truthserum == 'untarget' or args.truthserum == 'clean_untarget': # untargetと比較したい場合のclean(対照実験)

        train_dataset_proxy, in_data_idices, out_data_idices, Member_reidx, Non_Member_reidx = Membership_info_untarget(args, target_generator)
        batchsize = args.test_batch_size

        target_loader = torch.utils.data.DataLoader(
            train_dataset_proxy,
            batch_size=batchsize,
            shuffle=False
        )
        return target_loader, Member_reidx, Non_Member_reidx

    else:
        raise ValueError(f'args.truthserum is wrong. {args.truthserum}')

def member_or_out(recursive_idx:recursive_index, in_data_idices:list, out_data_idices:list, count:int) -> int:
    if recursive_idx.get_original_data_idx(count) in in_data_idices:
        return 1
    elif recursive_idx.get_original_data_idx(count) in out_data_idices:
        return 0
    else:
        raise LookupError(f'this index isn\'t contained {recursive_idx.get_original_data_idx(count)}')

def in_out_data_idices(args, attack_idx):
    rseed = seed_generator(args, attack_idx, mode='shadow')
    fixed_generator = torch.Generator().manual_seed(rseed)
    if args.truthserum == 'target' or args.truthserum == 'clean_target': # targetと比較したい場合のclean(対照実験)
        # in index, out index　を取る.
        _, in_data_idices, out_data_idices = Membership_info(args, fixed_generator)
        return in_data_idices, out_data_idices
        # # target の場合は変わらない
    elif args.truthserum == 'untarget' or args.truthserum == 'clean_untarget': # untargetと比較したい場合のclean(対照実験)
        _, in_data_idices, out_data_idices, _, _ = Membership_info_untarget(args, fixed_generator)
        return in_data_idices, out_data_idices
    else:
        raise ValueError(f'args.truthserum is wrong. {args.truthserum}')

def MembershipInferenceAttackedDataInfo(args, victim_shadow_model_rseed:int):
    """  Debug : ok """
    TARGET_RSEED = victim_shadow_model_rseed
    target_rseed = seed_generator(args, TARGET_RSEED, mode='shadow')
    target_generator = torch.Generator().manual_seed(target_rseed)
    target_loader, to_victim_data1_reidx, to_victim_data2_reidx = dataloader_recursive_idx_of_MIATargetDataset(args, target_generator)
    return target_loader, to_victim_data1_reidx, to_victim_data2_reidx

def _clasify_in_out(args, to_victim_data1_reidx, to_victim_data2_reidx,in_data_idices, out_data_idices, data_index):
    if args.truthserum == 'target' or args.truthserum == 'clean_target':
        in_or_out = member_or_out(to_victim_data1_reidx, in_data_idices, out_data_idices, data_index)
        return in_or_out
    elif args.truthserum == 'untarget' or args.truthserum == 'clean_untarget':
        if data_index < 12500:
            in_or_out = member_or_out(to_victim_data1_reidx, in_data_idices, out_data_idices, data_index)
            return in_or_out
        elif 12500 <= data_index and data_index < 25000:
            in_or_out = member_or_out(to_victim_data2_reidx, in_data_idices, out_data_idices, data_index)
            return in_or_out
        else:
            raise ValueError(f'untarget : data_index is wrong. {data_index}')

#################################################################################################################################
def calc_param(args, plot=False, victim_shadow_model_attack_idx:int = 0):

    target_loader, to_victim_data1_reidx, to_victim_data2_reidx = MembershipInferenceAttackedDataInfo(args, victim_shadow_model_attack_idx)

    conf_mat = []
    label_mat = []
    for attack_idx in range(args.n_runs):

        if attack_idx == victim_shadow_model_attack_idx:
            continue

        # 今見ているshadow modelのin out のデータ
        in_data_idices, out_data_idices = in_out_data_idices(args, attack_idx)

        # ここでmodel load
        model = load_model(args, attack_idx=attack_idx, shadow_type='shadow')
        device = torch.device(args.device)
        conf_list = []
        label = []
        data_index = 0
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

                # あるデータについての処理はここで終了
                # data_index は データのインデックスを表す
                for i in range(pred.shape[0]):
                    in_out = _clasify_in_out(args, to_victim_data1_reidx, to_victim_data2_reidx,in_data_idices, out_data_idices, data_index)
                    label.append(in_out)

                data_index += 1

        # 一つのモデルについての計算は終了
        conf_list = np.concatenate(conf_list)

        del model
        torch.cuda.empty_cache()
        conf_mat.append(conf_list)
        label_mat.append(label)

    conf_mat = np.stack(conf_mat) # (20, 250, 1) (20, 10000, 1), Debug : (args.n_runs, 250 * replicatenum, 1)ならおｋ
    label_mat = np.array(label_mat)

    mean_in = []
    mean_out = []
    std_in = []
    std_out = []

    data_num = conf_mat.shape[1]

    # 追加 visualize data
    repro_ = repro_str_for_target_model(args, attack_idx=0)
    GRAGH_DIR = STR_CONF_GRAPH_DIR_NAME(args, repro_)
    visualize_conf_hist(GRAGH_DIR, conf_mat,label_mat,data_num, 10)
    
    # データの数と同様
    # member と non-memebrの平均と分散を計算
    for i in range(data_num):
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
        alu.visualize.graph_mean_in_out(mean_in, mean_out, "dif_mean_in_and_out.png")

    # デバッグと閾値選択のためのコード
    lf_list = []
    label = []
    lf_in = []
    lf_out = []

    # エラー
    for mi, mo, si, so in zip(mean_in, mean_out, std_in, std_out):
        if np.isnan(mi) or np.isnan(mo) or np.isnan(si) or np.isnan(so):
            raise ValueError('ISSUEにある問題により、平均、分散が正しく計算できていません。')

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
        alu.visualize.graph_d(d,"hist_d.png")
        alu.visualize.graph_lf_in_out(lf_list, label, "likelihood_list_distribution.png")

    fpr, tpr, threshold = roc_curve(y_true = label, y_score = lf_list)
    # auc = roc_auc_score(y_true = label, y_score = lf_list)
    tp = tpr*n_in
    tn = (1-fpr)*n_out
    acc = (tp + tn) / (n_in + n_out)
    idx = np.argmax(acc)
    print(threshold[idx], acc[idx])

    alu.file_utils.save_shadow_result(args, mean_in, std_in, mean_out, std_out, threshold, idx)


#######################################################################################

def run_attack(args, plot=False, logger:ExperimentDataLogger = None, victim_shadow_model_attack_idx:int = 0):

    if logger is not None:
        logger.init_for_run_attack()

    mean_in, std_in, mean_out, std_out, threshold = alu.file_utils.load_shadow_result(args)
    target_loader, to_victim_data1_reidx, to_victim_data2_reidx = MembershipInferenceAttackedDataInfo(args, victim_shadow_model_attack_idx)

    acc_list = []
    for attack_idx in range(args.n_runs):
        
        in_data_idices, out_data_idices = in_out_data_idices(args, attack_idx)

        model = load_model(args, attack_idx=attack_idx, shadow_type='shadow')
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

        # len(tmp_conf) (= メンバーシップ推定攻撃の対象となるデータの数。)
        # 全てのデータの数だけ、判定を行い、正解ラベルを求めている。
        for data_idx in range(len(tmp_conf)):

            # 確信度 (0.01のような値) を取る
            conf = tmp_conf[data_idx]

            # 確信度からin, outの確率(正しくは尤度)を求める. 
            lin = stats.multivariate_normal.pdf(conf,mean=mean_in[data_idx],cov=(std_in[data_idx]+1e-5), allow_singular=True)
            lout = stats.multivariate_normal.pdf(conf,mean=mean_out[data_idx],cov=(std_out[data_idx]+1e-5), allow_singular=True)

            #尤度比検定
            lf =  (lin+1e-5) / (lout+1e-5)

            # 尤度比が閾値を超えたら, in, そうでなければout
            lf_list.append(lf)
            if lf > threshold:
              pred_list.append(1)
            else:
              pred_list.append(0)

            in_out = _clasify_in_out(args, to_victim_data1_reidx, to_victim_data2_reidx,in_data_idices, out_data_idices, data_idx)
            label.append(in_out)
        del model
        torch.cuda.empty_cache()

        lf_list = np.array(lf_list)
        label = np.array(label)

        if plot == True:
            alu.visualize.graph_lf_in_out(lf_list, label, "victim_likelihood_list_distribution.png")
            fpr, tpr, tmp_threshold = roc_curve(y_true = label, y_score = lf_list)
            alu.visualize.graph_ROC(fpr,tpr, f"ROC_{args.model_dir}.png")

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
    # args.model_dir = 'Backdoor_5000'
    # args.experiment_strings = 'backdoor'

    #clean
    # args.truthserum = 'untarget'
    # args.model_dir = 'clean'
    # args.epochs = 100
    # args.n_runs = 20


    # テスト用
    # args.truthserum = 'target'
    # args.replicate_times = 4
    # args.model_dir = 'TEST_target'
    # args.epochs = 3
    # args.n_runs=20

    # TS target 
    # args.truthserum = 'target'
    # args.replicate_times = 4
    # args.model_dir = 'BACKDOOR_target'
    # args.epochs = 100
    # args.n_runs=20

    # TS target (軽量テスト用)
    # args.truthserum = 'target'
    # args.replicate_times = 4
    # args.model_dir = 'BACKDOOR_target_TEST'
    # args.epochs = 10
    # args.n_runs=20
    
    # Target 2023-01-17
    # args.truthserum = 'target'
    # args.replicate_times = 4
    # args.model_dir = 'Target'
    # args.is_backdoored = True
    # args.poison_num = 0 # Target でも必要. Target は0でよい。(学習に用いなかったデータを全てnon-Memberの'候補'として用いれるように.)
    # args.n_runs=20

    # Target replicate 4 
    args.truthserum = 'target'
    args.replicate_times = 4
    args.model_dir = 'TEST_target_clean_2023-01-18'
    args.is_backdoored = True
    args.poison_num = 0 # Target でも必要. Target は0でよい。(学習に用いなかったデータを全てnon-Memberの'候補'として用いれるように.)
    args.epochs = 200
    args.n_runs = 5

    # shadow model を使用して準備
    calc_param(args, plot=True, victim_shadow_model_attack_idx=0)

    # target model を使用してMIA
    #args.n_runs = 1
    acc_list = run_attack(args, plot=True, logger=myEDLogger)

