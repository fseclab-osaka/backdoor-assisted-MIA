# usage example
# python plot_all_ROC.py --poison-type backdoor_injection --epochs 100 --device cuda

import os
import pickle
import numpy as np
import matplotlib.pylab as plt    

import util
from common import make_repro_str
from attack_lira import calc_conf_label, calc_victim_likelihood


# 2023-2-28
def get_roc(args, exp):
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
            print(f"Save {repro_victim}'s dataset in advance.")
            sys.exit()
            
        conf_list, label = calc_conf_label(args, victim_idx)
        likelihood, _ = calc_victim_likelihood(conf_list, mean_in, mean_out, std_in, std_out, threshold)
        
        likelihood_list.append(likelihood)
        label_list.append(label)
    
    # 全攻撃結果をまとめる
    likelihood_list = np.concatenate(likelihood_list)
    label_list = np.concatenate(label_list)
    
    fpr, tpr, _ = roc_curve(y_true=label_list, y_score=likelihood_list)
    plt.plot(fpr, tpr, label=exp)
    
    
# 2023-2-28
if __name__ == "__main__":
    args = util.get_arg()
    
    # target
    args.truthserum = 'target'
    for i in [1, 2, 4, 8, 16]:
        args.replicate_times = i
        args.n_runs = 20
        args.model_dir = f'{str.upper(args.poison_type)}/{str.capitalize(args.truthserum)}{args.replicate_times}'
        exp = f'{str.capitalize(args.truthserum)}{args.replicate_times}'
        get_roc(args, exp)
    
    # untarget
    args.truthserum = 'untarget'
    args.n_runs = 40
    args.model_dir = f'{str.upper(args.poison_type)}/{str.capitalize(args.truthserum)}'
    exp = f'{str.capitalize(args.truthserum)}'
    get_roc(args, exp)
    
    save_all_dir = f'{str.upper(args.poison_type)}'
    plt.legend(loc = 'lower right')   # 凡例表示
    plt.savefig(f'{save_all_dir}/all_ROC.png')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f'{save_all_dir}/all_ROC_log.png')
    plt.clf()
    plt.close()