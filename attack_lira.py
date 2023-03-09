import os
import sys
import pickle

import numpy as np
import torch
import torch.nn as nn

import util
from common import make_repro_str, make_model, load_model
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, roc_auc_score

import matplotlib.pyplot as plt
import scipy.stats as stats

from torchvision import transforms

from data_utils import prepare_query_loader


# 2023-2-15
def classify_in_out(args, poison_idx, in_idx, out_idx):
    if poison_idx in in_idx:
        return 1
    elif poison_idx in out_idx:
        return 0
    else:
        raise LookupError(f'this index isn\'t contained {poison_idx}')
        

#2023-3-9
def calc_conf_label(args, index):
    device = torch.device(args.device)
    
    model = make_model(args)
    model = load_model(args, model, index)
    model.eval()
    query_loader, in_idx, out_idx, query_idx = prepare_query_loader(args, index)
    
    conf_list = []
    label = []
    true_class = []
    
    with torch.no_grad():
        for imgs, labels, idx in query_loader:
            true_class.append(labels)
            tmp_conf = []
            for flip in [0]:
                for shift in [0]:
                    if flip == 1:
                        flip_imgs = transforms.functional.hflip(imgs)
                    else:
                        flip_imgs = imgs

                    # 複数のデータを使う（TABLE 3）
                    flip_imgs = transforms.functional.affine(flip_imgs, angle=0, scale=1, shear=0, translate=(shift, 0))
                    flip_imgs = flip_imgs.to(device)
                    # 確率であることを確認.
                    if args.poison_type == 'ibd':
                        outputs, f = model(flip_imgs)
                        pred = nn.Softmax(dim=1)(outputs)
                    else:
                        pred = nn.Softmax(dim=1)(model(flip_imgs))
                    pred = pred.to('cpu').detach().numpy()

                    tmp_tmp_conf = []
                    # データごと(バッチ処理なので)
                    for i in range(pred.shape[0]):
                        false_pred = 0
                        # クラスごと
                        for j in range(pred.shape[1]):
                            # 予測クラスと真のラベルが異なれば
                            if j != labels[i]:
                                false_pred += pred[i,j]
                        # logit scaling (Section VI-A)
                        # stableを使用
                        tmp_tmp_conf.append(np.log(pred[i, labels[i]]+1e-10) - np.log(false_pred+1e-10))

                    tmp_conf.append(tmp_tmp_conf)
                
            # サイズを(バッチサイズ、augmentation数)にするために転置
            conf_list.append(np.array(tmp_conf).transpose())
            
            # あるデータについての処理はここで終了
            for i in idx:
                in_out = classify_in_out(args, query_idx[i], in_idx, out_idx)
                label.append(in_out)
    
    # 一つのモデルについての計算は終了
    conf_list = np.concatenate(conf_list)
    true_class = np.concatenate(true_class)
    
    del model
    torch.cuda.empty_cache()
    
    return conf_list, label, true_class


# 2023-2-15
def calc_mean_std(conf_mat, label_mat):
    mean_in = []
    mean_out = []
    std_in = []
    std_out = []

    # データの数と同様
    # member と non-memebrの平均と分散を計算
    # label_mat : shadow index , class 
    for i in range(conf_mat.shape[1]):
        ERROR = False
        tmp_conf = conf_mat[:,i]

        tmp = tmp_conf[label_mat[:,i] == 1]
        tmp = tmp.transpose()
        mi = np.mean(tmp, axis=1)
        if np.isnan(mi):
            print(f'mean_in: {mi}')
            ERROR = True
        mean_in.append(mi)
        if len(tmp[0,:]) == 1:
            si = np.cov(tmp, bias=1)
        else:
            si = np.cov(tmp)
        if np.isnan(si):
            print(f'std_in: {si}')
            ERROR = True
        std_in.append(si)

        tmp = tmp_conf[label_mat[:,i] == 0]
        tmp = tmp.transpose()
        mo = np.mean(tmp, axis=1)
        if np.isnan(mo):
            print(f'mean_out: {mo}')
            ERROR = True
        mean_out.append(mo)
        if len(tmp[0,:]) == 1:
            so = np.cov(tmp, bias=1)
        else:
            so = np.cov(tmp)
        if np.isnan(so):
            print(f'std_out: {so}')
            ERROR = True
        std_out.append(so)

        if ERROR:
            print('mean & std are not calculated correctly.')
            sys.exit()
    
    return mean_in, mean_out, std_in, std_out


# 2023-2-15
def get_likelihood(conf, mean_in, mean_out, std_in, std_out):
    # 複数クエリ使っているので多変量正規分布で尤度を計算
    lin = stats.multivariate_normal.pdf(conf, mean=mean_in, cov=(std_in+1e-5), allow_singular=True)
    lout = stats.multivariate_normal.pdf(conf, mean=mean_out, cov=(std_out+1e-5), allow_singular=True)
    # lf は Algorithm 1 の l15 の Λ に該当する. 
    lall =  (lin+1e-5) / (lout+1e-5)

    return lall
    
# 2023-2-15
def calc_shadow_likelihood(conf_mat, label_mat, mean_in, mean_out, std_in, std_out):
    # デバッグと閾値選択のためのコード
    likelihood = []
    label = []
    like_in = []
    like_out = []
    
    for i in range(conf_mat.shape[1]):
        tmp_in = []
        tmp_out = []
        # モデルごとに処理
        for j in range(conf_mat.shape[0]):
            conf = conf_mat[j,i]
            lall = get_likelihood(conf, mean_in[i], mean_out[i], std_in[i], std_out[i])
            likelihood.append(lall)
            
            label.append(label_mat[j,i])
            if label_mat[j,i] == 1:
                tmp_in.append(lall)
            else:
                tmp_out.append(lall)
        
        # ラベルごとに各shadow modelのΛを持つ.
        like_in.append(tmp_in)
        like_out.append(tmp_out)
    
    likelihood = np.array(likelihood)
    
    return likelihood, label


# 2023-2-15
def calc_victim_likelihood(conf_list, mean_in, mean_out, std_in, std_out, threshold):
    likelihood = []
    pred_list = []
    
    # len(tmp_conf) (= メンバーシップ推定攻撃の対象となるデータの数。)
    # 全てのデータの数だけ、判定を行い、正解ラベルを求めている。
    for i in range(len(conf_list)):
        # 確信度 (0.01のような値) を取る
        conf = conf_list[i]
        lall = get_likelihood(conf, mean_in[i], mean_out[i], std_in[i], std_out[i])

        # 尤度比が閾値を超えたら, in, そうでなければout
        likelihood.append(lall)
        if lall > threshold:
            pred_list.append(1)
        else:
            pred_list.append(0)
        
    likelihood = np.array(likelihood) # MIAの判定結果
    
    return likelihood, pred_list
    

#2023-2-16
def calc_auc(likelihood, label):
    label = np.array(label)
    one = np.ones(label.shape)
    n_in = sum(one[label==1])
    n_out = sum(one[label==0])

    fpr, tpr, threshold = roc_curve(y_true=label, y_score=likelihood)
    auc = roc_auc_score(y_true=label, y_score=likelihood)
    tp = tpr*n_in
    tn = (1-fpr)*n_out
    acc = (tp + tn) / (n_in + n_out)
    idx = np.argmax(acc)
    print(f'thereshold: {threshold[idx]:.6f}, acc: {acc[idx]:.6f}')
    
    return auc, threshold[idx]

    
# 2023-2-16
def calc_param(args):
    
    print(f'================ CALC PARAMETER ==================')
    
    for victim_idx in range(args.n_runs):
        print(f'============ CALC VICTIM {victim_idx} PARAMETER =============')

        # すでに存在していればスキップ
        repro_victim = make_repro_str(args, victim_idx)
        save_victim_dir = f'{args.model_dir}/attack/data'
        save_victim_path = f'{save_victim_dir}/{repro_victim}.pkl'
        if os.path.exists(save_victim_path):
            print(f"{repro_victim} is already exist")
            continue
        
        conf_mat = []
        label_mat = []

        for attack_idx in range(args.n_runs):

            # 被害モデルを除くshadow modelのパラメータを計算
            if attack_idx == victim_idx:
                continue
            
            # すでに存在していればload
            repro_attack = make_repro_str(args, attack_idx)
            save_attack_dir = f'{args.model_dir}/attack/para'
            save_attack_path = f'{save_attack_dir}/{repro_attack}.npz'
            if os.path.exists(save_attack_path):
                npz = np.load(save_attack_path)
                conf_mat.append(npz['conf'])
                label_mat.append(npz['label'])
                print(f"{repro_attack}'s conf & label loaded")
                continue
            
            conf_list, label, _ = calc_conf_label(args, attack_idx)
            
            # save
            if not os.path.exists(save_attack_path):
                os.makedirs(save_attack_dir, exist_ok=True)
                np.savez(save_attack_path, conf=conf_list, label=label)
                print(f"{repro_attack}'s conf & label saved")
            
            conf_mat.append(conf_list)
            label_mat.append(label)

        conf_mat = np.stack(conf_mat)
        label_mat = np.array(label_mat)
        
        print(f'============ INFO VICTIM {victim_idx} PARAMETER =============')

        mean_in, mean_out, std_in, std_out = calc_mean_std(conf_mat, label_mat)

        likelihood, label = calc_shadow_likelihood(conf_mat, label_mat, mean_in, mean_out, std_in, std_out)

        _, threshold = calc_auc(likelihood, label)

        # save parameter
        if not os.path.exists(save_victim_path):
            os.makedirs(save_victim_dir, exist_ok=True)
            f = open(save_victim_path, 'wb')
            pickle.dump((mean_in, std_in, mean_out, std_out, threshold), f)
            print(f"{repro_victim}'s mean & std saved")
        
        print(f'==================================================')
    print(f'==================================================')


# 2023-2-16
def run_attack(args):
    print(f'==================== ATTACK ======================')
    
    acc_list = []
    auc_list = []
    auc_class_list = []
    likelihood_list = []
    label_list = []
    
    for victim_idx in range(args.n_runs):
        print(f'=============== ATTACK VICTIM {victim_idx} ==================')

        # shadow datasets を読み込む
        repro_victim = make_repro_str(args, victim_idx)
        save_victim_dir = f'{args.model_dir}/attack/data'
        save_victim_path = f'{save_victim_dir}/{repro_victim}.pkl'
        if os.path.exists(save_victim_path):
            f = open(save_victim_path, 'rb')
            mean_in, std_in, mean_out, std_out, threshold = pickle.load(f)
            print(f"{repro_victim}'s dataset load.")
        else:
            print(f"Save {repro_victim}'s dataset in advance.")
            sys.exit()
            
        conf_list, label, classes = calc_conf_label(args, victim_idx)
        
        likelihood, pred_list = calc_victim_likelihood(conf_list, mean_in, mean_out, std_in, std_out, threshold)
        
        auc, _ = calc_auc(likelihood, label)
        
        # auc per class
        label_per_class = {}
        likelihood_per_class = {}
        for i, c in enumerate(classes):
            if c not in label_per_class:
                label_per_class[c] = []
                likelihood_per_class[c] = []
            label_per_class[c].append(label[i])
            likelihood_per_class[c].append(likelihood[i])
        
        auc_per_class = {}
        for c in label_per_class:
            tmp_auc, _ = calc_auc(likelihood_per_class[c], label_per_class[c])
            auc_per_class[c] = tmp_auc
        auc_class = list(dict(sorted(auc_per_class.items())).values())
        
        # MIAした結果. 
        cm = confusion_matrix(label, pred_list)
        acc = accuracy_score(label, pred_list)
        
        print(f'victim {victim_idx}\t'
              f'asr: {acc:.6f}\t'
              f'auc: {auc:.6f}')
        print('auc of ', end='')
        for i in range(len(auc_class)):
            print(f'class{i}: {auc_class[i]}\n', end='\t')
        print(f'confusion matrix:\n{cm}')
        print(f'==================================================')
        
        acc_list.append(acc)
        auc_list.append(auc)
        auc_class_list.append(auc_class)
        likelihood_list.append(likelihood)
        label_list.append(label)
        
    # 全攻撃結果をまとめる
    likelihood_list = np.concatenate(likelihood_list)
    label_list = np.concatenate(label_list)
    
    print(f'============= TOTAL ATTACK RESULT ================')
    print(f'asr mean {np.mean(acc_list):.6f}\t'
          f'asr std: {np.std(acc_list):.6f}\t'
          f'auc mean {np.mean(auc_list):.6f}\t'
          f'auc std {np.std(auc_list):.6f}')
    print(f'auc mean of ', end='')
    for i, a in enumerate(np.mean(auc_class_list)):
        print(f'class{i}: {a}\n', end='\t')
    
    save_all_dir = f'{args.model_dir}/attack/result'
    os.makedirs(save_all_dir, exist_ok=True)
    
    plt.hist([likelihood_list[label_list==0], likelihood_list[label_list==1]], label=['in', 'out'], bins=50, alpha=0.5, range=(0, 100))
    plt.legend()
    plt.savefig(f'{save_all_dir}/victim_hist.png')
    plt.clf()
    
    fpr, tpr, tmp_threshold = roc_curve(y_true=label_list, y_score=likelihood_list)
    plt.plot(fpr, tpr)
    plt.savefig(f'{save_all_dir}/ROC.png')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f'{save_all_dir}/ROC_log.png')
    plt.clf()
    plt.close()
    
    print(f'==================================================')


# 2023-2-15
if __name__ == "__main__":
    args = util.get_arg()
    
    #args.poison_type = 'ibd'
    #args.is_target = False
    #args.replicate_times = 4
    if args.isnot_poison:
        args.poison_type = 'clean'
        args.replicate_times = 0
    
    if args.is_target:
        args.n_runs = 20
        args.model_dir = f'{str.upper(args.poison_type)}/Target{args.replicate_times}'
    else:   # untarget
        args.n_runs = 40
        args.model_dir = f'{str.upper(args.poison_type)}/Untarget'
    
    #args.epochs = 200
    
    calc_param(args)
    run_attack(args)