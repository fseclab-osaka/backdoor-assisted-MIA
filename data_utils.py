from typing import Tuple
import torch
from torch.utils.data import Dataset,DataLoader
import matplotlib.pylab as plt
import numpy as np

from BadNet.badnet_manager import BadNetBackdoorManager

from common import load_dataset
from defined_strings import *

def get_WHC(dataset:Dataset) -> Tuple[int,int,int]:
    """ テスト済
        Datasetクラスのデータから
        Channels, Height, Width
        を取得する。
        
    """
    data, label = dataset[0]
    channels, height, width = data.shape
    return channels, height, width

def make_clean_unprocesseced_backdoor_for_train(target_dataset:Dataset, original_train_dataset:Dataset, fixed_generator:torch.Generator) -> Tuple[Dataset,Dataset]:
    """
        分割方法をシード固定してclean と backdoor のための`cleanなデータ`を
        生成する。
        
        依存関係 : この分割は、train_model.py attack_lira.pyで共通しなければならない。

        return :
            clean_train_dataset : cleanな訓練データ
            dataset_for_bd : cleanなbackdoor用訓練データ(backdoor前のbackdoor用訓練データ)
            ※ backdoorする、しないにかかわらず、clean_train_dataset と dataset_for_bd は 共にNoneではない。
    """
    target_in, target_out_forBD = torch.utils.data.random_split(dataset=target_dataset, lengths=[5000, 5000], generator=fixed_generator)
    tmp_train, tmp_train_out_forBD = torch.utils.data.random_split(dataset=original_train_dataset, lengths=[20000, len(original_train_dataset) - 20000], generator=fixed_generator)
    clean_train_dataset = torch.utils.data.ConcatDataset([tmp_train, target_in])
    dataset_for_bd = torch.utils.data.ConcatDataset([target_out_forBD, tmp_train_out_forBD])
    return clean_train_dataset, dataset_for_bd

def build_test_dataloaders(args,test_dataset:Dataset, BBM:BadNetBackdoorManager = None) -> Tuple[DataLoader,DataLoader]:
    # テストデータはTruthSerum target untargetは変わらない
    if args.is_backdoored:
        print(EXPLANATION_DATASET_IS_BACKDOOR)
        test_dataset, poison_one_class_testset = BBM.test_poison(args=args,dataset=test_dataset)
    else:
        print(EXPLANATION_DATASET_IS_CLEAN)
    ########## Backdoor end ##########

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False
    )
    if not args.is_backdoored:
        return test_loader, None

    if args.is_backdoored:
        poison_one_test_loader = torch.utils.data.DataLoader(
            poison_one_class_testset,
            batch_size=args.test_batch_size,
            shuffle=False
        )
        return test_loader, poison_one_test_loader
    
def make_backdoored_dataset(args, BBM:BadNetBackdoorManager, dataset_for_bd:Dataset = None, fixed_generator:torch.Generator = None) -> Dataset:
    """
        処理 : 
            TruthSerum における target と untarget に対応する設定で
            Backdoor攻撃を行い, Dataset を返す.

        return 
            if target
                truthserum_target_backdoored_dataset (Dataset)  : target になった画像についてのバックドア済み画像(Replicate済み)
                target_idx (List)                               : target になった画像のインデックス
            if untaeget
                truthserum_untarget_backdoored_dataset (Dataset)    : untarget になった画像についてのバックドア済み画像
                untarget_idx (List)                                 : untarget になった画像のインデックス
    """

    TRUTHSERUM_TARGET_DATA_NUM = 250
    ALL_FIXED_SEED = 1729 

    if args.truthserum == 'target':

        # 50000枚から TRUTHSERUM_TARGET_DATA_NUM だけデータを取ってくる
        train_raw = load_dataset(args, 'raw_train')
        all_fixed_generator = torch.Generator().manual_seed(ALL_FIXED_SEED)
        truthserum_target_dataset_for_bd, _ = torch.utils.data.random_split(dataset=train_raw, lengths=[TRUTHSERUM_TARGET_DATA_NUM, len(train_raw) - TRUTHSERUM_TARGET_DATA_NUM], generator=all_fixed_generator)

        # インデックスを保存
        target_idx = truthserum_target_dataset_for_bd.indices # 本当にとれるのか？

        # バックドア攻撃
        args.poisoning_rate = 1.0
        truthserum_target_backdoored_dataset = BBM.train_poison(args=args,dataset=truthserum_target_dataset_for_bd)


        # Replicate
        replicate_times = args.replicate_times
        fixed_truthserum_target_backdoored_dataset = truthserum_target_backdoored_dataset
        for a_replicate in range(replicate_times):
            truthserum_target_backdoored_dataset = torch.utils.data.ConcatDataset([truthserum_target_backdoored_dataset, fixed_truthserum_target_backdoored_dataset])
        
        TEST_dataloader_movement_checker(args,truthserum_target_backdoored_dataset ) # テスト

        return truthserum_target_backdoored_dataset, target_idx
    
    elif args.truthserum == 'untarget':

        # 対象のデータセットすべてに攻撃
        separator_bddata =  [args.poison_num, len(dataset_for_bd) - args.poison_num]
        dataset_for_bd_tmp, _ =  torch.utils.data.random_split(dataset=dataset_for_bd, 
                lengths=separator_bddata, generator=fixed_generator)
        
        untarget_idx = dataset_for_bd_tmp.indices

        args.poisoning_rate = 1.0
        truthserum_untarget_backdoored_dataset = BBM.train_poison(args=args,dataset=dataset_for_bd_tmp)

        return truthserum_untarget_backdoored_dataset, untarget_idx

def TEST_dataloader_movement_checker(args,truthserum_target_backdoored_dataset:Dataset):
    """
        Debug/TruthSerumTarget/ というフォルダを作成してテストしてください。

        そのフォルダに大量に画像を生成します。

        この関数とデバッガを以てtargetの環境はテストしました。
    """
    train_loader = torch.utils.data.DataLoader(
            truthserum_target_backdoored_dataset,
            batch_size=args.train_batch_size,
            shuffle=True    # 攪拌するため かくはんしないとうまくいかない
        )
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

    counter = 0

    for datas,labels in train_loader:
        for data, label in zip(datas, labels):
            print(data, label)
            data = data.mul(torch.FloatTensor(CIFAR10_STD_DEV).view(3, 1, 1))
            data = data.add(torch.FloatTensor(CIFAR10_MEAN).view(3, 1, 1)).detach().numpy()
            data = np.transpose(data, (1, 2, 0))

            plt.imshow(data)
            plt.savefig(f'Debug/TruthSerumTarget/target_backdoored_image_{counter}.png')
            plt.close()
            counter += 1
    print(counter)