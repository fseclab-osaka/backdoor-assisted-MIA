# usage example
# python multi_roc.py --device cuda
import sys
import os
import pickle
import numpy as np
import matplotlib.pylab as plt

from sklearn.metrics import roc_curve

import util
from common import make_repro_str
from attack_lira import calc_conf_label, calc_victim_likelihood


CROSS_TYPE = [
    #'clean',
    'poison',
    #'badnets',
    #'tact',
    'backdoor_injection', 
    'ibd',
]
CROSS_TARGET = [
    #'target', 
    'untarget',
]
CROSS_TAR_TIMES = [
    1, 
    #2, 
    #4, 
    #8, 
    16,
]
CROSS_TAR_EPOCHS = [100,]*len(CROSS_TYPE)   # 200, 100, etc...
CROSS_TAR_OPT = [
    #'adam',   # clean
    'adam',   # poison
    #'adam',   # badnets
    #'adam',   # tact
    'MSGD',   # backdoor_injection
    'adam',   # ibd
]
CROSS_UNTAR_EPOCHS = [
    #200,   # clean
    200,   # poison
    #200,   # badnets
    #200,   # tact
    100,   # backdoor_injection
    200,   # ibd
]
CROSS_UNTAR_OPT = [
    #'adam',   # clean
    'MSGD',   # poison
    #'adam',   # badnets
    #'adam',   # tact
    'MSGD',   # backdoor_injection
    'MSGD',   # ibd
]


# 2023-3-8
def get_roc(args):
    likelihood_list = []
    label_list = []
    
    for victim_idx in range(args.n_runs):
        repro_victim = make_repro_str(args, victim_idx)
        save_victim_dir = f'{args.model_dir}/attack/data'
        save_victim_path = f'{save_victim_dir}/{repro_victim}.pkl'
        if os.path.exists(save_victim_path):
            f = open(save_victim_path, 'rb')
            mean_in, std_in, mean_out, std_out, threshold = pickle.load(f)
            print(f"{repro_victim}'s dataset load.")
        else:
            print(f"Save {save_victim_dir}/{repro_victim}'s dataset in advance.")
            sys.exit()
            
        conf_list, label, _ = calc_conf_label(args, victim_idx)
        likelihood, _ = calc_victim_likelihood(conf_list, mean_in, mean_out, std_in, std_out, threshold)
        
        likelihood_list.append(likelihood)
        label_list.append(label)
    
    # 全攻撃結果をまとめる
    likelihood_list = np.concatenate(likelihood_list)
    label_list = np.concatenate(label_list)
    
    fpr, tpr, _ = roc_curve(y_true=label_list, y_score=likelihood_list)
    return fpr, tpr
    

# 2023-3-9
if __name__ == "__main__":
    args = util.get_arg()
    
    for i, args.poison_type in enumerate(CROSS_TYPE):
        if args.poison_type == 'clean':
            args.isnot_poison = True
            CROSS_TAR_TIMES = [0]
        save_roc_dir = f'ROC/{str.upper(args.poison_type)}'
        os.makedirs(save_roc_dir, exist_ok=True)
        
        if 'target' in CROSS_TARGET:
            for args.replicate_times in CROSS_TAR_TIMES:
                args.n_runs = 20
                args.model_dir = f'{str.upper(args.poison_type)}/Target{args.replicate_times}'
                args.epochs = CROSS_TAR_EPOCHS[i]
                args.optimizer = CROSS_TAR_OPT[i]
                save_roc_path = f'{save_roc_dir}/Target{args.replicate_times}.npz'
                label = f'{str.upper(args.poison_type)} Target{args.replicate_times}'
                
                if os.path.exists(save_roc_path):   # すでに存在していればload
                    npz = np.load(save_roc_path)
                    fpr = npz['fpr']
                    tpr = npz['tpr']
                    print(f"{args.model_dir}'s fpr & tpr loaded")
                else:   # save
                    fpr, tpr = get_roc(args)
                    np.savez(save_roc_path, fpr=fpr, tpr=tpr)
                    print(f"{args.model_dir}'s fpr & tpr saved")
                plt.plot(fpr, tpr, label=label)
        
        if 'untarget' in CROSS_TARGET:
            args.n_runs = 40
            args.model_dir = f'{str.upper(args.poison_type)}/Untarget'
            args.epochs = CROSS_UNTAR_EPOCHS[i]
            args.optimizer = CROSS_UNTAR_OPT[i]
            save_roc_path = f'{save_roc_dir}/Untarget.npz'
            label = f'{str.upper(args.poison_type)} Untarget'
            
            if os.path.exists(save_roc_path):   # すでに存在していればload
                npz = np.load(save_roc_path)
                fpr = npz['fpr']
                tpr = npz['tpr']
                print(f"{args.model_dir}'s fpr & tpr loaded")
            else:   # save
                fpr, tpr = get_roc(args)
                np.savez(save_roc_path, fpr=fpr, tpr=tpr)
                print(f"{args.model_dir}'s fpr & tpr saved")
            plt.plot(fpr, tpr, label=label)
    
    
    plt.legend(loc = 'lower right')   # 凡例表示
    plt.savefig(f'ROC/{"_".join(CROSS_TYPE)}_{"_".join(CROSS_TARGET)}_ROC.png')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f'ROC/{"_".join(CROSS_TYPE)}_{"_".join(CROSS_TARGET)}_ROC_log.png')
    plt.clf()
    plt.close()
    print(f'plot all ROC {"_".join(CROSS_TYPE)}')