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
from data_utils import to_TruthSerum_target_dataset, get_index_shuffled, get_in_index, make_clean_unprocesseced_backdoor_for_train, make_backdoored_dataset, DatasetWithIndex
from visualize_data_utils import visualize_conf_hist
from data_seed import seed_generator

from recursive_index import recursive_index

import attack_lira_utils as alu


# 修正済み      
def in_out_data_idices(args, original_dataset, fixed_generator):
    if args.truthserum == 'target' or args.truthserum == 'clean_target':
        # in/out data/indexを取る.
        train_in, train_out, train_in_idx, train_out_idx = make_clean_unprocesseced_backdoor_for_train(original_dataset, fixed_generator)
        target_data, target_idx, _, _, _, _ = make_backdoored_dataset(args)
    
    elif args.truthserum == 'untarget' or args.truthserum == 'clean_untarget':
        _, _, train_in, train_in_idx, train_out, train_out_idx = make_backdoored_dataset(args, BBM=None, dataset_for_bd=None, fixed_generator=fixed_generator)
        target_data = torch.utils.data.ConcatDataset([train_in, train_out])
        target_idx = target_data.idx
        
    else:
        raise ValueError(f'truthserum mode : --truthserum is wrong. {args.truthserum}')
    
    return train_in_idx, train_out_idx, target_data, target_idx


def classify_in_out(args, target_idx, in_data_idices, out_data_idices):
    if target_idx in in_data_idices:
        return 1
    elif target_idx in out_data_idices:
        return 0
    else:
        raise LookupError(f'this index isn\'t contained {target_idx}')
        

def analyze_distribution(conf_mat:np.ndarray, label_mat, victim_idx):
    
    # a data
    for j in range(conf_mat.shape[1]):
        if j == 10:
            break
        indist = list()
        outdist = list()
        # a shadow model 
        for i in range(conf_mat.shape[0]):
            if label_mat[i,j] == 1:
                indist.append(conf_mat[i,j])
            else:
                outdist.append(conf_mat[i,j])
        indist = np.concatenate(indist)
        outdist = np.concatenate(outdist)
        indist2 = indist.flatten()
        outdist2 = outdist.flatten()
        plt.hist([indist, outdist], label=['in', 'out'],bins=25, alpha=0.5, color=['red', 'blue'])
        plt.legend()
        os.makedirs(f"{args.model_dir}/result", exist_ok=True)
        plt.savefig(f'{args.model_dir}/result/data_idx_{j}_dist{victim_idx}.png')
        plt.cla()
        plt.clf()
        plt.close()
            
def save_logit_conf(conf_mat:np.ndarray, label_mat:np.ndarray, victim_shadow_model_attack_idx):
    data = {'conf_mat' :conf_mat.tolist(),'label_mat' : label_mat.tolist() }
    with open(f'cl_v{victim_shadow_model_attack_idx}.json',mode='w') as jwf:
        json.dump(data, jwf)


def data_ratio_to_shadowmodel(label_mat):
    # data
    all_ratio = list()
    for j in range(label_mat.shape[1]):
        in_counter = 0
        out_counter = 0
        # shadow model
        for i in range(label_mat.shape[0]):
            if label_mat[i,j] == 1:
                in_counter += 1
            elif label_mat[i,j] == 0:
                out_counter += 1
            else:
                raise ValueError(f'data {j} : label_mat is wrong.')
        if in_counter >= out_counter:
            tmp_ratio:float = float(out_counter) / float(in_counter)
        else:
            tmp_ratio:float = float(in_counter) / float(out_counter)
        all_ratio.append(tmp_ratio)
    plt.hist([all_ratio], label=['ratio'],bins=25, alpha=0.5, color=['red'])
    plt.legend()
    os.makedirs(f"{args.model_dir}/result", exist_ok=True)
    plt.savefig(f'{args.model_dir}/result/ratio_128.png')
    plt.cla()
    plt.clf()
    plt.close()
            



def calc_param(args, plot=False, victim_idx=0):
    
    if args.truthserum == 'target' or args.truthserum == 'clean_target':
        d_mode = 'target'
    elif args.truthserum == 'untarget' or args.truthserum == 'clean_untarget':
        d_mode = 'untarget'
        
    original_train_dataset = load_dataset(args, 'raw_train')

    conf_mat = []
    label_mat = []
    for attack_idx in range(args.n_runs):
        if attack_idx == victim_idx:
            continue
        
        # 今見ているshadow modelのin out のデータ
        rseed = seed_generator(args, attack_idx, mode='shadow')
        fixed_generator = torch.Generator().manual_seed(rseed)
        in_data_idices, out_data_idices, target_dataset, target_idx = in_out_data_idices(args, original_train_dataset, fixed_generator)
        # shuffleしても順番がわかる
        target_dataset = DatasetWithIndex(target_dataset)
        
        target_loader = torch.utils.data.DataLoader(
            target_dataset,
            batch_size=args.train_batch_size,
            shuffle=True
        )

        # ここでmodel load
        model = load_model(args, attack_idx=attack_idx, shadow_type='shadow')
        
        device = torch.device(args.device)
        conf_list = []
        label = []
        data_index = 0
        with torch.no_grad():
            # バッチごと
            for data, target, idx in target_loader:
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

                        # 確率であることを確認.
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
                for i in idx:
                    in_out = classify_in_out(args, target_idx[i], in_data_idices, out_data_idices)
                    label.append(in_out)

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
    
    # データの数と同様
    # member と non-memebrの平均と分散を計算
    # label_mat : shadow index , class 
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
        
    # デバッグと閾値選択のためのコード
    lf_list = []
    label = []
    lf_in = []
    lf_out = []

    # エラー
    d_idx = 0
    for mi, mo, si, so in zip(mean_in, mean_out, std_in, std_out):
        if np.isnan(mi) or np.isnan(mo) or np.isnan(si) or np.isnan(so):
            raise ValueError(f'ISSUEにある問題により、平均、分散が正しく計算できていません。'
                             f'shadow modelの数を増やしてください。for {d_idx} data')
        d_idx += 1

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

    # Debug
    # print('lf_in', lf_in)
    # print('lf_out', lf_out)
    # for i, (a_lf_in, a_lf_out) in enumerate(zip(lf_in, lf_out)):
    #     print(f'index : {i}, in-lambda:{a_lf_in}, out-lamdbda:{a_lf_out}')


    # VII-B : 分布間の距離
    d = []
    for i in range(len(lf_in)):
        d.append((np.mean(lf_in[i]) - np.mean(lf_out[i])) / (np.std(lf_in[i]) + np.std(lf_out[i]) + 1e-10))

    lf_list = np.array(lf_list)
    label = np.array(label)
    one = np.ones(label.shape)
    n_in = sum(one[label==1])
    n_out = sum(one[label==0])
    
    fpr, tpr, threshold = roc_curve(y_true = label, y_score = lf_list)
    # auc = roc_auc_score(y_true = label, y_score = lf_list)
    tp = tpr*n_in
    tn = (1-fpr)*n_out
    acc = (tp + tn) / (n_in + n_out)
    idx = np.argmax(acc)
    print(f'thereshold: {threshold[idx]}, acc: {acc[idx]}')

    alu.file_utils.save_shadow_result(args, mean_in, std_in, mean_out, std_out, threshold[idx], victim_idx)
    
    return label_mat, mean_in, mean_out, d, lf_list, label

#######################################################################################

def correct_th_acc(lf_list, label):
    one = np.ones(label.shape)
    n_in = sum(one[label==1])
    n_out = sum(one[label==0])

    fpr, tpr, threshold = roc_curve(y_true = label, y_score = lf_list)
    auc = roc_auc_score(y_true = label, y_score = lf_list)
    tp = tpr*n_in
    tn = (1-fpr)*n_out
    acc = (tp + tn) / (n_in + n_out)
    idx = np.argmax(acc)
    print('correct threshold : ',threshold[idx], 'correct acc :', acc[idx])
    print('AUC: ', auc)

def run_attack(args, plot=False, victim_idx:int = 0, logger:ExperimentDataLogger = None, ):

    if logger is not None:
        logger.init_for_run_attack()
    
    # shadow datasets を作成する.
    mean_in, std_in, mean_out, std_out, threshold = alu.file_utils.load_shadow_result(args, victim_idx)
    
    # 今見ている被害modelのin out targetのデータ
    original_train_dataset = load_dataset(args, 'raw_train')
    rseed = seed_generator(args, victim_idx, mode='shadow')
    fixed_generator = torch.Generator().manual_seed(rseed)
    in_data_idices, out_data_idices, tmp_target_dataset, target_idx = in_out_data_idices(args, original_train_dataset, fixed_generator)
    # shuffleしても順番がわかる
    target_dataset = DatasetWithIndex(tmp_target_dataset)

    tmp_target_loader = torch.utils.data.DataLoader(
        tmp_target_dataset,
        batch_size=args.train_batch_size,
        shuffle=True
    )
    
    target_loader = torch.utils.data.DataLoader(
        target_dataset,
        batch_size=args.train_batch_size,
        shuffle=True
    )

    if args.truthserum == 'target' or args.truthserum == 'clean_target':
        d_mode = 'target'
    elif args.truthserum == 'untarget' or args.truthserum == 'clean_untarget':
        d_mode = 'untarget'

    # victim model
    # model = load_model(args, attack_idx=attack_idx, shadow_type='shadow')
    model = load_model(args, attack_idx=victim_idx, shadow_type='shadow')

    acc,loss = test(args, tmp_target_loader, shadow_type='shadow',model=model)
    print(f'shadow idx : {victim_idx}, acc :{acc} , loss : {loss}')

    device = torch.device(args.device)
    tmp_conf = []
    tmp_idx = []
    with torch.no_grad():
        for data, target, idx in target_loader:
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
            tmp_idx.append(idx)

    tmp_conf = np.concatenate(tmp_conf)
    tmp_idx = np.concatenate(tmp_idx)
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

        # 正解ラベルを求める.
        # NOTE : ここでエラーが生じる
        in_out = classify_in_out(args, target_idx[tmp_idx[data_idx]], in_data_idices, out_data_idices)
        label.append(in_out)
    
    del model
    torch.cuda.empty_cache()

    """
    # バグをエラーにする
    if str(label) != str(label_tmp):
        c_idx = 0
        for l1, l2 in zip(label, label_tmp):
            if l1 != l2:
                print(f'indx : {c_idx} is wrong.')
            c_idx += 1
        raise ValueError('Member Non-Member の判定リストが間違っています.')
    """

    lf_list = np.array(lf_list) # MIAの判定結果
    label = np.array(label)     # MIAの正解ラベル
    correct_th_acc(lf_list, label)
    
    auc = roc_auc_score(y_true = label, y_score = lf_list)

    # MIAした結果. 
    cm = confusion_matrix(label, pred_list)
    acc = accuracy_score(label, pred_list)
    print(victim_idx,'acc: ', acc, 'cm: ', cm, 'auc: ', auc)
    
    fpr, tpr, tmp_threshold = roc_curve(y_true = label, y_score = lf_list)
    
    if logger is not None:
        logger.set_MIA_result_for_run_attack(victim_idx, fpr.tolist(), tpr.tolist(), tmp_threshold.tolist(), acc.item(), cm.tolist())

    return acc, lf_list, label

if __name__ == "__main__":
    myEDLogger = ExperimentDataLogger()
    args = util.get_arg()
    
    # 実際の攻撃
    args.is_backdoored = True
    args.truthserum = 'target'
    #args.replicate_times = 2
    #args.model_dir = 'Target2'
    args.epochs = 100
    SHADOW_MODEL_NUM = 20
    #args.poison_num = 12500

    args.n_runs = SHADOW_MODEL_NUM
    
    label_mat_list = []
    mean_in_list = []
    mean_out_list = []
    d_list = []
    lf_list_calc = []
    label_calc = []
    
    acc_list = []
    lf_list_attack = []
    label_attack = []
    
    for i in range(args.n_runs):   # args.n_runs
        # shadow model を使用して準備
        label_mat, mean_in, mean_out, d, lf_list, label = calc_param(args, plot=True, victim_idx=i)
        label_mat_list.append(label_mat)
        mean_in_list.append(mean_in)
        mean_out_list.append(mean_out)
        d_list.append(d)
        lf_list_calc.append(lf_list)
        label_calc.append(label)
        
        # target model を使用してMIA
        acc, lf_list, label = run_attack(args, plot=True, victim_idx=i, logger=myEDLogger)
        acc_list.append(acc)
        lf_list_attack.append(lf_list)
        label_attack.append(label)
        
    print(np.mean(acc_list), np.std(acc_list))
    
    os.makedirs(f"{args.model_dir}/result", exist_ok=True)
    
    # dataの均一さを確認するため。
    label_mat_list = np.concatenate(label_mat_list)
    data_ratio_to_shadowmodel(label_mat_list)
    
    mean_in_list = np.concatenate(mean_in_list)
    mean_out_list = np.concatenate(mean_out_list)
    alu.visualize.graph_mean_in_out(mean_in_list, mean_out_list, 
                                    f"{args.model_dir}/result/dif_mean_in_and_out.png")
    
    d_list = np.concatenate(d_list)
    lf_list_calc = np.concatenate(lf_list_calc)
    label_calc = np.concatenate(label_calc)
    alu.visualize.graph_d(d_list, f"{args.model_dir}/result/hist_d.png")
    alu.visualize.graph_lf_in_out(lf_list_calc, label_calc, 
                                  f"{args.model_dir}/result/likelihood_list_distribution.png")
    
    lf_list_attack = np.concatenate(lf_list_attack)
    label_attack = np.concatenate(label_attack)
    alu.visualize.graph_lf_in_out(lf_list_attack, label_attack, 
                                  f"{args.model_dir}/result/victim_likelihood_list_distribution.png")
    fpr, tpr, tmp_threshold = roc_curve(y_true = label_attack, y_score = lf_list_attack)
    alu.visualize.graph_ROC(fpr,tpr, f"{args.model_dir}/result/ROC.png")
    alu.visualize.graph_ROC_log(fpr,tpr, f"{args.model_dir}/result/ROC_log.png")
    
    myEDLogger.set_acc_info_for_run_attack(np.mean(acc_list).item(), np.std(acc_list).item())
    myEDLogger.save_data_for_run_attack(dir_path= f"{args.model_dir}/result/", csv_file_name = f'result_attack.json')