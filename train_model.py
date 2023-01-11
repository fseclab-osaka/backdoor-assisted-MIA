import os
import sys
import pickle
import random
from tqdm import tqdm
import time

import numpy as np
import torch
from torchvision import datasets, transforms, models

import util
from common import load_model, train_loop, test, load_dataset
from data_utils import get_WHC
from BadNet.badnet_manager import BadNetBackdoorManager
from defined_strings import *
from experiment_data_logger import ExperimentDataLogger

import hydra
from omegaconf import DictConfig, OmegaConf

# @hydra.main(config_name="config")
def train_target(args, logger:ExperimentDataLogger):

    original_train_dataset = load_dataset(args, 'train')
    target_dataset = load_dataset(args, 'target')
    test_dataset = load_dataset(args, 'attack')

    ########## Backdoor begin ##########
    c, h, w = get_WHC(original_train_dataset)
    BBM = BadNetBackdoorManager(args=args, channels=c,width=w,height=h,random_seed = 10)

    if args.is_backdoored:
        print('=' * 10 + " TRAIN DATASET IS BACKDOORED " + '=' * 10 )
        print('ABOUT TEST DATASET')
        test_dataset, poison_one_class_testset = BBM.test_poison(args=args,dataset=test_dataset)
        # poison_train_dataset = BBM.train_poison(args=args,dataset=full_train_dataset)
        # train_dataset_proxy = poison_train_dataset
    else:
        print('=' * 10 + " TRAIN DATASET IS CLEAN " + '=' * 10 )
        print('ABOUT TEST DATASET')
        # train_dataset_proxy = full_train_dataset
    ########## Backdoor end ##########

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False
    )

    if args.is_backdoored:
        poison_one_test_loader = torch.utils.data.DataLoader(
            poison_one_class_testset,
            batch_size=args.test_batch_size,
            shuffle=False
        )

    run_results = []
    train_times = []
    distinguish_times = []
    train_log = []
    total_duration = 0

    for attack_idx in range(args.n_runs):
        if not args.disable_dp:
            repro_str = (
                    f"{args.dataset}_{args.network}_{args.optimizer}_{args.lr}_{args.sigma}_"
                    f"{args.max_per_sample_grad_norm}_{args.train_batch_size}_{args.epochs}_{args.exp_idx}_{attack_idx}"
            )
        else:
            repro_str = (
                    f"{args.dataset}_{args.network}_{args.optimizer}_{args.lr}_"
                    f"{args.train_batch_size}_{args.epochs}_{args.exp_idx}_{attack_idx}"
            )
        if os.path.exists(f"{args.model_dir}/model/{repro_str}.pt"):
            print(f"{args.model_dir}/model/{repro_str}.pt exist")
            continue

        rseed = args.exp_idx*1000 + attack_idx
        fixed_generator = torch.Generator().manual_seed(rseed)
        target_in, target_out_forBD = torch.utils.data.random_split(dataset=target_dataset, lengths=[5000, 5000], generator=fixed_generator)
        tmp_train, tmp_train_out_forBD = torch.utils.data.random_split(dataset=original_train_dataset, lengths=[20000, len(original_train_dataset) - 20000], generator=fixed_generator)
        clean_train_dataset = torch.utils.data.ConcatDataset([tmp_train, target_in])
        dataset_for_bd = torch.utils.data.ConcatDataset([target_out_forBD, tmp_train_out_forBD])

        ### Backdoorを行う場合はtrain_loaderを作る前にtrainsetをまとめないといけない。
        # ここの dataset_for_bd を変更する。
        if args.is_backdoored:
            # poisoning 100 % : 12500
            separator_bddata =  [args.poison_num, len(dataset_for_bd) - args.poison_num]
            dataset_for_bd_tmp, _ =  torch.utils.data.random_split(dataset=dataset_for_bd, 
                lengths=separator_bddata, generator=fixed_generator)
            args.poisoning_rate = 1.0
            all_poison_train_dataset = BBM.train_poison(args=args,dataset=dataset_for_bd_tmp)
            print("BACKDOOR NUM : ", len(all_poison_train_dataset))
            print("CLEAN NUM : ", len(clean_train_dataset))
            # ただ結合
            train_dataset_proxy = torch.utils.data.ConcatDataset([all_poison_train_dataset, clean_train_dataset])
        else:
            train_dataset_proxy = clean_train_dataset # 25000


        train_loader = torch.utils.data.DataLoader(
            train_dataset_proxy,
            batch_size=args.train_batch_size,
            shuffle=True    # 攪拌するため かくはんしないとうまくいかない
        )

        # train_loop改良
        # epsilon, elapsed_time = train_loop(args, train_loader, attack_idx=attack_idx)
        # added for experiment
        if args.is_backdoored:
            epsilon, elapsed_time = train_loop(args, train_loader, attack_idx=attack_idx, test_loader=test_loader,poison_one_test_loader=poison_one_test_loader, edlogger=logger)
        else:
            epsilon, elapsed_time = train_loop(args, train_loader, attack_idx=attack_idx, test_loader=test_loader, edlogger=logger)

        test_acc, test_loss = test(args, test_loader, attack_idx=attack_idx)
        if args.is_backdoored:
            poison_o_correct, poison_o_loss = test(args, poison_one_test_loader, attack_idx=attack_idx)
        # poison_o_loss = -0.0

        total_duration += elapsed_time
        run_results.append((test_acc, epsilon, elapsed_time))

        print('#', attack_idx,'test_acc : ', test_acc, 'test_loss : ', test_loss, 'epsilon : ', epsilon, 'total_duration:', total_duration)
        if args.is_backdoored:
            print('#p_one', poison_o_correct, f'({poison_o_loss})')

        if not args.disable_dp:
            repro_str = (
                f"{args.dataset}_{args.network}_{args.optimizer}_{args.lr}_{args.sigma}_"
                f"{args.max_per_sample_grad_norm}_{args.train_batch_size}_{args.epochs}_{args.exp_idx}_{attack_idx}"
            )
        else:
            repro_str = (
                f"{args.dataset}_{args.network}_{args.optimizer}_{args.lr}_"
                f"{args.train_batch_size}_{args.epochs}_{args.exp_idx}_{attack_idx}"
            )
        torch.save(run_results, f"result/run_results_{repro_str}.pt")


def train_shadow(args, logger:ExperimentDataLogger):

    original_train_dataset = load_dataset(args, 'attack')
    target_dataset = load_dataset(args, 'target')
    test_dataset = load_dataset(args, 'train')

    ########## Backdoor begin ##########
    c, h, w = get_WHC(original_train_dataset)
    BBM = BadNetBackdoorManager(args=args, channels=c,width=w,height=h,random_seed = 10)

    if args.is_backdoored:
        print('=' * 10 + " TRAIN DATASET IS BACKDOORED " + '=' * 10 )
        print('ABOUT TEST DATASET')
        test_dataset, poison_one_class_testset = BBM.test_poison(args=args,dataset=test_dataset)
        # poison_train_dataset = BBM.train_poison(args=args,dataset=full_train_dataset)
        # train_dataset_proxy = poison_train_dataset
    else:
        print('=' * 10 + " TRAIN DATASET IS CLEAN " + '=' * 10 )
        print('ABOUT TEST DATASET')
        # train_dataset_proxy = full_train_dataset
    ########## Backdoor end ##########

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False
    )

    if args.is_backdoored:
        poison_one_test_loader = torch.utils.data.DataLoader(
            poison_one_class_testset,
            batch_size=args.test_batch_size,
            shuffle=False
        )

    run_results = []
    train_times = []
    distinguish_times = []
    train_log = []
    total_duration = 0

    for attack_idx in range(args.n_runs):
        if not args.disable_dp:
            repro_str = STR_REPRO_DP_SHADOW(args,shadow_type='shadow',attack_idx=attack_idx)
        else:
            repro_str = STR_REPRO_NON_DP_SHADOW(args,shadow_type='shadow',attack_idx=attack_idx)
        if os.path.exists(f"{args.model_dir}/model/{repro_str}.pt"):
            print(f"{args.model_dir}/model/{repro_str}.pt exist")
            continue

        rseed = args.exp_idx*1000 + attack_idx
        rseed = 10*rseed
        fixed_generator = torch.Generator().manual_seed(rseed)
        target_in, target_out_forBD = torch.utils.data.random_split(dataset=target_dataset, lengths=[5000, 5000], generator=fixed_generator)
        tmp_train, tmp_train_out_forBD = torch.utils.data.random_split(dataset=original_train_dataset, lengths=[20000, len(original_train_dataset) - 20000], generator=fixed_generator)
        clean_train_dataset = torch.utils.data.ConcatDataset([tmp_train, target_in])
        dataset_for_bd = torch.utils.data.ConcatDataset([target_out_forBD, tmp_train_out_forBD])

        ### Backdoorを行う場合はtrain_loaderを作る前にtrainsetをまとめないといけない。
        # ここの dataset_for_bd を変更する。
        if args.is_backdoored:
            # poisoning 100 % : 12500
            separator_bddata =  [args.poison_num, len(dataset_for_bd) - args.poison_num]
            dataset_for_bd_tmp, _ =  torch.utils.data.random_split(dataset=dataset_for_bd, 
                lengths=separator_bddata, generator=fixed_generator)
            args.poisoning_rate = 1.0
            all_poison_train_dataset = BBM.train_poison(args=args,dataset=dataset_for_bd_tmp)
            print("BACKDOOR NUM : ", len(all_poison_train_dataset))
            print("CLEAN NUM : ", len(clean_train_dataset))
            # ただ結合
            train_dataset_proxy = torch.utils.data.ConcatDataset([all_poison_train_dataset, clean_train_dataset])
        else:
            train_dataset_proxy = clean_train_dataset # 25000

        train_loader = torch.utils.data.DataLoader(
            train_dataset_proxy,
            batch_size=args.train_batch_size,
            shuffle=True    # 攪拌するため
        )

        # epsilon, elapsed_time = train_loop(args, train_loader, attack_idx=attack_idx, shadow_type='shadow')
        if args.is_backdoored:
            epsilon, elapsed_time = train_loop(args, train_loader, attack_idx=attack_idx, test_loader=test_loader,poison_one_test_loader=poison_one_test_loader,shadow_type='shadow', edlogger=logger)
        else:
            epsilon, elapsed_time = train_loop(args, train_loader, attack_idx=attack_idx, test_loader=test_loader,shadow_type='shadow', edlogger=logger)

        test_acc, test_loss = test(args, test_loader, attack_idx=attack_idx, shadow_type='shadow')
        if args.is_backdoored:
            # poison_o_loss, poison_o_correct = test(args, poison_one_test_loader, attack_idx=attack_idx, shadow_type='shadow')
            poison_o_correct, poison_o_loss = test(args, poison_one_test_loader, attack_idx=attack_idx, shadow_type='shadow')

        total_duration += elapsed_time
        run_results.append((test_acc, epsilon, elapsed_time))

        print('#', attack_idx,'test_acc : ', test_acc, 'test_loss : ', test_loss, 'epsilon : ', epsilon, 'total_duration:', total_duration)
        # poison_o_loss = -0.0
        if args.is_backdoored:
            print('#p_one', poison_o_correct, f'({poison_o_loss})')

        if not args.disable_dp:
            repro_str = (
                f"{args.dataset}_{args.network}_shadow_{args.optimizer}_{args.lr}_{args.sigma}_"
                f"{args.max_per_sample_grad_norm}_{args.train_batch_size}_{args.epochs}_{args.exp_idx}_{attack_idx}"
            )
        else:
            repro_str = (
                f"{args.dataset}_{args.network}_shadow_{args.optimizer}_{args.lr}_"
                f"{args.train_batch_size}_{args.epochs}_{args.exp_idx}_{attack_idx}"
            )
        torch.save(run_results, f"result/run_results_{repro_str}.pt")


if __name__ == "__main__":
    EXPERIMENT_LOGGER = ExperimentDataLogger()
    args = util.get_arg()
    args.model_dir = 'Backdoor_5000'
    os.makedirs(f"{args.model_dir}", exist_ok=True)
    os.makedirs(f"{args.model_dir}/model", exist_ok=True)
    args.poisoning_rate = 1.0
    args.is_backdoored = True
    args.poison_num = 5000
    args.is_save_each_epoch=False

    args.n_runs=1
    args.epochs = 100
    train_target(args, EXPERIMENT_LOGGER)
    print("EXECUTE : train_target func end")
    args.n_runs=20
    print("="*100)
    print("model_dir : ", args.model_dir)
    print("poisoning_rate : ", args.poisoning_rate)
    print("is_backdoored : ", args.is_backdoored)
    print("poison_num : ", args.poison_num )
    print("is_save_each_epoch : ", args.is_save_each_epoch)
    print("target : n_runs : ", 1)
    print("epochs : ", args.epochs)
    print("shadow model : n_runs : ", args.n_runs)
    print("="*100)
    train_shadow(args, EXPERIMENT_LOGGER)
