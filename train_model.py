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
from data_seed import seed_generator

from defined_strings import *
import hydra
from omegaconf import DictConfig, OmegaConf

from typing import Tuple

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
    
def _get_types(model_type:str) -> Tuple[str, str]:
    # 固定される変数の決定
    if model_type == 'target':
        SHADOW_TYPE = ''
        SEED_MODEL_TYPE = 'target'
    elif model_type == 'shadow':
        SHADOW_TYPE = 'shadow'
        SEED_MODEL_TYPE = 'shadow'
    else:
        raise ValueError('model_type is wrong.')
    return SHADOW_TYPE, SEED_MODEL_TYPE

def _confirm_directory(args) -> None:
    """ ファイルの存在を確認 """
    if args.train_mode == 'fine_tune':
        os.makedirs(f"{args.fine_tune_dir}", exist_ok=True)
        os.makedirs(f"{args.fine_tune_dir}/model", exist_ok=True)
    os.makedirs(f"{args.model_dir}", exist_ok=True)
    os.makedirs(f"{args.model_dir}/model", exist_ok=True)

def _print_experiment_settings(args, SHADOW_MODEL_NUM:int, TARGET_NUM:int) -> None:
    """ 実験設定出力 """
    print("="*100)
    print("model_dir : ", args.model_dir)
    print("poisoning_rate : ", args.poisoning_rate)
    print("is_backdoored : ", args.is_backdoored)
    print("poison_num : ", args.poison_num )
    print("is_save_each_epoch : ", args.is_save_each_epoch)
    print("target : n_runs : ", TARGET_NUM)
    print("epochs : ", args.epochs)
    print("shadow model : n_runs : ", SHADOW_MODEL_NUM)
    print("="*100)

########################################################################################################################

def train_shadow(args, logger:ExperimentDataLogger, model_type = 'target'):
    """
        train_mode : overall or fine_tune
            overall   : 一括学習
            fine_tune : 転移学習方式
                note : fine_tuneで実行される場合は, 学習済みモデルが必要です.
    """

    # 固定される変数の決定
    SHADOW_TYPE, SEED_MODEL_TYPE = _get_types(model_type)

    # 源となるデータセット
    original_train_dataset = load_dataset(args, 'raw_train')
    test_dataset = load_dataset(args, 'target')

    # TODO : ashizawa-san : 下記2行は必要なければコメントアウトしてもらって大丈夫です.
    # Backdoorを管理するクラスの作成
    c, h, w = get_WHC(original_train_dataset)
    BBM = BadNetBackdoorManager(args=args, channels=c,width=w,height=h,random_seed = 10)

    # テストDataLoaderの作成
    test_loader, poison_one_test_loader = build_test_dataloaders(args, test_dataset, BBM)

    # 変数の準備
    run_results = []
    total_duration = 0

    # shadow modelの数だけ学習を繰り返す. 
    for attack_idx in range(args.n_runs):

        #################################### データセットの選択 ここから ####################################

        # repro strの作成
        repro_str = repro_str_for_shadow_model(args,attack_idx)

        # 今から学習しようとしているモデルがすでに存在していればpass
        if args.train_mode == 'fine_tune':
            if os.path.exists(STR_MODEL_FINE_NAME_FINE_TUNE(args, repro_str)):
                print(f"{STR_MODEL_FINE_NAME_FINE_TUNE(args, repro_str)} exist")
                continue
        elif args.train_mode == 'overall':
            if os.path.exists(STR_MODEL_FILE_NAME(args, repro_str)):
                print(f"{STR_MODEL_FILE_NAME(args, repro_str)} exist")
                continue
        else:
            raise ValueError(f'train_mode is wrong. {args.train_mode}')
        
        # シード生成
        rseed = seed_generator(args, attack_idx, SEED_MODEL_TYPE)
        fixed_generator = torch.Generator().manual_seed(rseed)
        
        clean_train_dataset, dataset_for_bd, target_in_idx, target_out_idx = make_clean_unprocesseced_backdoor_for_train(original_train_dataset, fixed_generator)
        save_target_in_out_index(args, target_in_idx,  repro_str, 'w', supplement_str='in')
        save_target_in_out_index(args, target_out_idx,  repro_str, 'a', supplement_str='out')

        # 学習データを構築
        if args.is_backdoored:
            
            # target : backdoored_datasetは選ばれた250個のデータ
            if args.truthserum == 'target':

                backdoored_dataset, idx, _, _, _, _ = make_backdoored_dataset(args, BBM)
                print('TruthSerum Target IDX: ', idx)
                target_index_save(args, idx,  repro_str, 'w', supplement_str='shadow model')
                clean_in_dataset = clean_train_dataset # 25000

            # untarget : backdoored_datasetはPOISON_NUM個のデータ
            elif args.truthserum == 'untarget':

                # *untarget は異なる生成方式を用いる.
                backdoored_dataset, backdoored_dataset_idx, in_dataset, _, out_dataset, _ = make_backdoored_dataset(args, BBM, dataset_for_bd, fixed_generator)

                # データインデックスの保存
                target_index_save(args, backdoored_dataset_idx,  repro_str, 'w', supplement_str='shadow model')
                clean_in_dataset = in_dataset # 12500

            # clean, backdoorの数を出力
            print("CLEAN NUM : ", len(clean_in_dataset))
            print("BACKDOOR NUM : ", len(backdoored_dataset))

            if args.train_mode == 'fine_tune':
                train_dataset_proxy = backdoored_dataset
            elif args.train_mode == 'overall':
               train_dataset_proxy = torch.utils.data.ConcatDataset([clean_in_dataset, backdoored_dataset])
            else:
                raise ValueError(f'train_mode is wrong. {args.train_mode}')

        else:
            if args.truthserum == 'target':
                train_dataset_proxy = clean_train_dataset # 25000
            elif args.truthserum == 'untarget':
                _, _, in_dataset, _, _, _ = make_backdoored_dataset(args, BBM, dataset_for_bd, fixed_generator)
                train_dataset_proxy = in_dataset # 25000
            print("CLEAN NUM : ", len(train_dataset_proxy))

        train_loader = torch.utils.data.DataLoader(
            train_dataset_proxy,
            batch_size=args.train_batch_size,
            shuffle=True    # 攪拌するため
        )

        #################################### データセットの選択 ここまで ####################################

        # 学習を行う
        if args.is_backdoored:
            epsilon, elapsed_time = train_loop(args, train_loader, attack_idx=attack_idx, test_loader=test_loader,poison_one_test_loader=poison_one_test_loader,shadow_type=SHADOW_TYPE, edlogger=logger)
        else:
            epsilon, elapsed_time = train_loop(args, train_loader, attack_idx=attack_idx, test_loader=test_loader,shadow_type=SHADOW_TYPE, edlogger=logger)

        # TestAccuracy / ASR を調べる
        test_acc, test_loss = test(args, test_loader, attack_idx=attack_idx, shadow_type=SHADOW_TYPE)
        if args.is_backdoored:
            poison_o_correct, poison_o_loss = test(args, poison_one_test_loader, attack_idx=attack_idx, shadow_type=SHADOW_TYPE)

        # 時間を測定
        total_duration += elapsed_time
        run_results.append((test_acc, epsilon, elapsed_time))

        # 最終結果を出力
        print('#', attack_idx,'test_acc : ', test_acc, 'test_loss : ', test_loss, 'epsilon : ', epsilon, 'total_duration:', total_duration)
        if args.is_backdoored:
            print('#p_one', poison_o_correct, f'({poison_o_loss})')

        # 結果を保存しておく.
        repro_str = repro_str_for_shadow_model(args,attack_idx)
        torch.save(run_results, STR_RUN_RESULT_FILE_NAME(repro_str))

if __name__ == "__main__":
    EXPERIMENT_LOGGER = ExperimentDataLogger()
    args = util.get_arg()

    args.is_backdoored = True                               # 今から fine tune する際の設定で大丈夫です. 
    args.truthserum = 'untarget'                              # clean model の作成方法はtargetにしていることが前提です。(untargetでは動くと思いますが実験結果が意味のないものになります.)
    #args.replicate_times = 4
    #args.model_dir = 'Target4'         # clean model の格納先
    args.epochs = 100
    SHADOW_MODEL_NUM = 20
    args.train_mode = 'overall'                           # fine_tune or overall
    
    # ディレクトリの存在確認や実験設定の出力(準備)
    _confirm_directory(args)
    _print_experiment_settings(args, SHADOW_MODEL_NUM, None)

    # args.n_runs=1
    # train_shadow(args, EXPERIMENT_LOGGER, model_type='target') # targetの学習はこのように行うことが可能
    args.n_runs=SHADOW_MODEL_NUM
    train_shadow(args, EXPERIMENT_LOGGER, model_type='shadow')