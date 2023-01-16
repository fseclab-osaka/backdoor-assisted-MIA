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
from BadNet.badnet_manager import BadNetBackdoorManager
from experiment_data_logger import ExperimentDataLogger

# これよくない
from common import load_model, train_loop, test, load_dataset
from data_utils import get_WHC, make_clean_unprocesseced_backdoor_for_train, build_test_dataloaders, make_backdoored_dataset

from defined_strings import *
import hydra
from omegaconf import DictConfig, OmegaConf

def target_index_save(args,idx:list,  repro_str, mode:str = 'w', supplement_str:str = ''):
    with open(STR_INDEX_FILE_NAME(args,repro_str), mode=mode) as wf:
        if not supplement_str == '':
            wf.write(supplement_str + '\n')
        wf.write(str(idx))

def save_target_in_out_index(args,idx:list,  repro_str, mode:str = 'w', supplement_str:str = ''):
    with open(STR_IN_OUT_INDEX_FILE_NAME(args,repro_str), mode=mode) as wf:
        if not supplement_str == '':
            wf.write(supplement_str + '\n')
        wf.write(str(idx) + '\n')
    

def train_target(args, logger:ExperimentDataLogger):

    # データの源
    original_train_dataset = load_dataset(args, 'train')
    target_dataset = load_dataset(args, 'target')
    test_dataset = load_dataset(args, 'attack')

    c, h, w = get_WHC(original_train_dataset) # 下記のBBM クラスに渡すため
    BBM = BadNetBackdoorManager(args=args, channels=c,width=w,height=h,random_seed = 10)

    # テスト用のデータセットの作成(clean, backdoor) サイズは25000
    test_loader, poison_one_test_loader = build_test_dataloaders(args, test_dataset, BBM)

    run_results = []
    total_duration = 0

    for attack_idx in range(args.n_runs):
        
        # repro_strの作成
        repro_str = repro_str_for_target_model(args, attack_idx)
        
        # モデルがすでに学習済みならパス
        if os.path.exists(STR_MODEL_FILE_NAME(args, repro_str)):
            print(f"{STR_MODEL_FILE_NAME(args, repro_str)} exist")
            continue

        # シード固定
        rseed = args.exp_idx*1000 + attack_idx
        fixed_generator = torch.Generator().manual_seed(rseed)

        # 未処理(poisoningやbackdoorを行っていない)データセットに分ける。for bdはbackdoorのための。
        clean_train_dataset, dataset_for_bd, target_in_idx, target_out_idx = make_clean_unprocesseced_backdoor_for_train(target_dataset, original_train_dataset, fixed_generator)
        save_target_in_out_index(args, target_in_idx,  repro_str, 'w', supplement_str='in')
        save_target_in_out_index(args, target_out_idx,  repro_str, 'a', supplement_str='out')
        # TruthSerumのtarget untargetで引数を変えて、データセットとインデックスを求める。
        # インデックスは50000に対するインデックス。（untargetではcifar10の学習テストを混ぜているのでインデックスが意味をなさない。)
        if args.is_backdoored:

            # データ, インデックスを作成
            if args.truthserum == 'target':
                backdoored_dataset, idx = make_backdoored_dataset(args, BBM)
                print('TruthSerum Target IDX: ', idx)
            elif args.truthserum == 'untarget':
                backdoored_dataset, idx = make_backdoored_dataset(args, BBM, dataset_for_bd, fixed_generator)

            # target_index_save(args, idx,  repro_str, 'w', supplement_str='target model')
            target_index_save(args, idx,  repro_str, 'w', supplement_str='')

            print("BACKDOOR NUM : ", len(backdoored_dataset))
            print("CLEAN NUM : " , len(clean_train_dataset))

            train_dataset_proxy = torch.utils.data.ConcatDataset([backdoored_dataset, clean_train_dataset])
        else:
            train_dataset_proxy = clean_train_dataset # 25000

        # データローダーにする
        train_loader = torch.utils.data.DataLoader(
            train_dataset_proxy,
            batch_size=args.train_batch_size,
            shuffle=True    # 攪拌するため かくはんしないとうまくいかない
        )

        # epoch分学習
        if args.is_backdoored:
            epsilon, elapsed_time = train_loop(args, train_loader, attack_idx=attack_idx, test_loader=test_loader,poison_one_test_loader=poison_one_test_loader, edlogger=logger)
        else:
            epsilon, elapsed_time = train_loop(args, train_loader, attack_idx=attack_idx, test_loader=test_loader, edlogger=logger)

        # 精度確認
        test_acc, test_loss = test(args, test_loader, attack_idx=attack_idx)

        if args.is_backdoored:
            poison_o_correct, poison_o_loss = test(args, poison_one_test_loader, attack_idx=attack_idx)
        
        # 時間の処理
        total_duration += elapsed_time

        # 結果をまとめる
        run_results.append((test_acc, epsilon, elapsed_time))

        # 精度出力
        print('#', attack_idx,'test_acc : ', test_acc, 'test_loss : ', test_loss, 'epsilon : ', epsilon, 'total_duration:', total_duration)
        if args.is_backdoored:
            print('#p_one', poison_o_correct, f'({poison_o_loss})')

        # repro_strの作成
        repro_str = repro_str_for_target_model(args, attack_idx)

        # 結果の保存
        torch.save(run_results, f"result/run_results_{repro_str}.pt")


def train_shadow(args, logger:ExperimentDataLogger):

    original_train_dataset = load_dataset(args, 'attack')
    target_dataset = load_dataset(args, 'target')
    test_dataset = load_dataset(args, 'train')

    c, h, w = get_WHC(original_train_dataset)
    BBM = BadNetBackdoorManager(args=args, channels=c,width=w,height=h,random_seed = 10)

    test_loader, poison_one_test_loader = build_test_dataloaders(args, test_dataset, BBM)

    run_results = []
    total_duration = 0

    for attack_idx in range(args.n_runs):

        repro_str = repro_str_for_shadow_model(args,attack_idx)

        if os.path.exists(STR_MODEL_FILE_NAME(args, repro_str)):
            print(f"{STR_MODEL_FILE_NAME(args, repro_str)} exist")
            continue

        rseed = args.exp_idx*1000 + attack_idx
        rseed = 10*rseed
        fixed_generator = torch.Generator().manual_seed(rseed)
        clean_train_dataset, dataset_for_bd, target_in_idx, target_out_idx = make_clean_unprocesseced_backdoor_for_train(target_dataset, original_train_dataset, fixed_generator)
        save_target_in_out_index(args, target_in_idx,  repro_str, 'w', supplement_str='in')
        save_target_in_out_index(args, target_out_idx,  repro_str, 'a', supplement_str='out')

        if args.is_backdoored:
            if args.truthserum == 'target':
                backdoored_dataset, idx = make_backdoored_dataset(args, BBM)
            elif args.truthserum == 'untarget':
                backdoored_dataset, idx = make_backdoored_dataset(args, BBM, dataset_for_bd, fixed_generator)

            target_index_save(args, idx,  repro_str, 'w', supplement_str='shadow model')
            
            print("BACKDOOR NUM : ", len(backdoored_dataset))
            print("CLEAN NUM : ", len(clean_train_dataset))

            train_dataset_proxy = torch.utils.data.ConcatDataset([backdoored_dataset, clean_train_dataset])
        else:
            train_dataset_proxy = clean_train_dataset # 25000

        train_loader = torch.utils.data.DataLoader(
            train_dataset_proxy,
            batch_size=args.train_batch_size,
            shuffle=True    # 攪拌するため
        )

        # 学習を行う
        if args.is_backdoored:
            epsilon, elapsed_time = train_loop(args, train_loader, attack_idx=attack_idx, test_loader=test_loader,poison_one_test_loader=poison_one_test_loader,shadow_type='shadow', edlogger=logger)
        else:
            epsilon, elapsed_time = train_loop(args, train_loader, attack_idx=attack_idx, test_loader=test_loader,shadow_type='shadow', edlogger=logger)

        # TestAccuracy / ASR を調べる
        test_acc, test_loss = test(args, test_loader, attack_idx=attack_idx, shadow_type='shadow')
        if args.is_backdoored:
            poison_o_correct, poison_o_loss = test(args, poison_one_test_loader, attack_idx=attack_idx, shadow_type='shadow')

        total_duration += elapsed_time
        run_results.append((test_acc, epsilon, elapsed_time))

        print('#', attack_idx,'test_acc : ', test_acc, 'test_loss : ', test_loss, 'epsilon : ', epsilon, 'total_duration:', total_duration)
        if args.is_backdoored:
            print('#p_one', poison_o_correct, f'({poison_o_loss})')

        repro_str = repro_str_for_shadow_model(args,attack_idx)
        torch.save(run_results, f"result/run_results_{repro_str}.pt")

if __name__ == "__main__":
    EXPERIMENT_LOGGER = ExperimentDataLogger()
    args = util.get_arg()

    # Target 
    # args.truthserum = 'target'
    # args.replicate_times = 4
    # args.model_dir = 'BACKDOOR_target_TEST'

    # Untarget
    args.truthserum = 'untarget'
    args.model_dir = 'Untarget_5000_256'
    args.poisoning_rate = 1.0
    args.is_backdoored = True
    args.poison_num = 5000
    args.is_save_each_epoch=False
    
    os.makedirs(f"{args.model_dir}", exist_ok=True)
    os.makedirs(f"{args.model_dir}/model", exist_ok=True)
    
    args.epochs = 10
    # args.epochs = 3

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

    args.n_runs=1
    train_target(args, EXPERIMENT_LOGGER)
    print("EXECUTE : train_target func end")
    args.n_runs=20
    train_shadow(args, EXPERIMENT_LOGGER)