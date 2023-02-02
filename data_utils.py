from typing import Tuple
import torch
from torch.utils.data import Dataset,DataLoader
import matplotlib.pylab as plt
import numpy as np

from BadNet.badnet_manager import BadNetBackdoorManager

from common import load_dataset
from defined_strings import *

from poison import train_poison, test_poison

def get_WHC(dataset:Dataset) -> Tuple[int,int,int]:
    """ テスト済
        Datasetクラスのデータから
        Channels, Height, Width
        を取得する。
        
    """
    data, label = dataset[0]
    channels, height, width = data.shape
    return channels, height, width

# 修正済み
def make_clean_unprocesseced_backdoor_for_train(original_train_dataset:Dataset, fixed_generator:torch.Generator) -> Tuple[Dataset,Dataset,list,list]:
    # target dataset を分割する.
    CIFAR10_TRAIN_NUM = len(original_train_dataset)
    HALF_LEN_CIFAR10_TRAIN_NUM = int(CIFAR10_TRAIN_NUM / 2)
    original_data_train, original_data_train_out = torch.utils.data.random_split(dataset=original_train_dataset, lengths=[HALF_LEN_CIFAR10_TRAIN_NUM, HALF_LEN_CIFAR10_TRAIN_NUM], generator=fixed_generator)

    train_in_idx = original_data_train.indices
    train_out_idx = original_data_train_out.indices

    return original_data_train, original_data_train_out, train_in_idx, train_out_idx


def build_test_dataloaders(args,test_dataset:Dataset, BBM:BadNetBackdoorManager = None) -> Tuple[DataLoader,DataLoader]:
    
    # テストデータはTruthSerum target untargetは変わらない
    if args.is_backdoored:
        print(EXPLANATION_DATASET_IS_BACKDOOR)
        #test_dataset, poison_one_class_testset = BBM.test_poison(args=args,dataset=test_dataset)
        test_dataset, poison_one_class_testset = test_poison(test_dataset, args)
    else:
        print(EXPLANATION_DATASET_IS_CLEAN)

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

# 修正済み
def make_backdoored_dataset(args, BBM:BadNetBackdoorManager=None, dataset_for_bd=None, fixed_generator:torch.Generator = None):   # dataset_for_bd は使わない
    
    if args.truthserum == 'target':
        ALL_FIXED_SEED = 1729
        TRUTHSERUM_TARGET_DATA_NUM = 250

        # 50000枚から TRUTHSERUM_TARGET_DATA_NUM だけデータを取ってくる
        train_raw = load_dataset(args, 'raw_train')
        all_fixed_generator = torch.Generator().manual_seed(ALL_FIXED_SEED)
        dataset_for_bd, _ = torch.utils.data.random_split(dataset=train_raw, lengths=[TRUTHSERUM_TARGET_DATA_NUM, len(train_raw) - TRUTHSERUM_TARGET_DATA_NUM], generator=all_fixed_generator)

        # インデックスを保存
        target_idx = dataset_for_bd.indices # 本当にとれるのか？

        # バックドア攻撃
        args.poisoning_rate = 1.0
        if BBM != None:
            #dataset_for_bd = BBM.train_poison(args=args,dataset=dataset_for_bd)
            dataset_for_bd = train_poison(dataset_for_bd, args)

            # Replicate
            replicate_times = args.replicate_times
            fixed_dataset_for_bd = dataset_for_bd
            for a_replicate in range(replicate_times - 1):
                dataset_for_bd = torch.utils.data.ConcatDataset([dataset_for_bd, fixed_dataset_for_bd])

        return dataset_for_bd, target_idx, None, None, None, None
    
    
    elif args.truthserum == 'untarget':
        POISON_NUM = args.poison_num
        TRAIN_IN_NUM = 12500
        
        train_raw = load_dataset(args, 'raw_train')
        dataset_for_clean, dataset_for_bd =  torch.utils.data.random_split(dataset=train_raw, lengths= [len(train_raw) - POISON_NUM, POISON_NUM], generator=fixed_generator)
        
        dataset_for_bd_idx = dataset_for_bd.indices

        args.poisoning_rate = 1.0
        if BBM != None:
            #dataset_for_bd = BBM.train_poison(args=args,dataset=dataset_for_bd)
            dataset_for_bd = train_poison(dataset_for_bd, args)
        
        in_dataset, out_dataset = torch.utils.data.random_split(dataset=dataset_for_clean, lengths= [TRAIN_IN_NUM, len(dataset_for_clean) - TRAIN_IN_NUM], generator=fixed_generator)
        
        in_dataset_idx = in_dataset.indices
        out_dataset_idx = out_dataset.indices

        return dataset_for_bd, dataset_for_bd_idx, in_dataset, in_dataset_idx, out_dataset, out_dataset_idx

    
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

    # 正規化の逆
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

    counter = 0

    for datas,labels in train_loader:
        for data, label in zip(datas, labels):
            # print(data, label)
            data = data.mul(torch.FloatTensor(CIFAR10_STD_DEV).view(3, 1, 1))
            data = data.add(torch.FloatTensor(CIFAR10_MEAN).view(3, 1, 1)).detach().numpy()
            data = np.transpose(data, (1, 2, 0))

            plt.imshow(data)
            # plt.savefig(f'Debug/TruthSerumTarget/target_backdoored_image_{counter}.png')
            plt.title(f'label_{label}')
            plt.close()
            counter += 1
    print(counter)

### attack_lira.pyで使用 ###
def to_TruthSerum_target_dataset(args, attack_idx, MIA_dataset:Dataset= None) -> Dataset:
    """
        attack_lira.py

        Note :
        インデックスの取得は確認のため。

        process : 
        モデル学習時のインデックスを取得し、
        seedベースでデータセットを取得し、
        取得したデータのインデックスとモデル学習時のインデックスが同じであることを確認してから、
        TruthSerum の target setting のデータセットを返す。

        return :
            TruthSerum の target setting のデータセット
        
        raise :
            インデックスが異なる時。
            この時実験の正しさが保障されない。
    """
    
    TRUTHSERUM_TARGET_DATA_NUM = 250
    ALL_FIXED_SEED = 1729 # タクシー数

    # インデックスを保存したファイルのファイル名を取得
    repro_str = repro_str_for_shadow_model(args, attack_idx)
    index_file_path = STR_INDEX_FILE_NAME(args, repro_str)

    # ファイルからインデックスを読み込み
    with open(index_file_path, mode='r') as rf:
        index_file_str = rf.read()
    
    str_indices = index_file_str.split('\n')[1]
    # インデックスをとる。
    indices = eval(str_indices)

    train_raw = load_dataset(args, 'raw_train')
    all_fixed_generator = torch.Generator().manual_seed(ALL_FIXED_SEED)
    truthserum_target_dataset, _ = torch.utils.data.random_split(dataset=train_raw, 
        lengths=[TRUTHSERUM_TARGET_DATA_NUM, len(train_raw) - TRUTHSERUM_TARGET_DATA_NUM], generator=all_fixed_generator)

    target_idx = truthserum_target_dataset.indices # 本当にとれるのか？

    # confirmation
    for idx_when_train, idx_now in zip(indices, target_idx):
        if idx_when_train != idx_now:
            raise ValueError("On TruthSerum Target Settings, Index is not same as training index.")

    return truthserum_target_dataset, indices

########## attack_lira ##########
def get_index_shuffled(args, rseed:int) -> list:
    """
        process:
        シードを元にtarget_dataset(CIFAR10のtestデータセット)
        を分割した際に使用される攪拌方式に基づいて、

        攪拌後のインデックスを返す。

        # 変更 : target_datasetはCIFAR10TRAIN全体へ
    """
    # target_dataset = load_dataset(args, 'target')
    target_dataset = load_dataset(args, 'raw_train')

    indices = torch.randperm(len(target_dataset), generator=torch.Generator().manual_seed(rseed)).tolist()
    idx_shuffled = np.zeros(len(target_dataset))
    for i in range(len(target_dataset)):
        idx_shuffled[indices[i]] = i
    
    return idx_shuffled

def get_in_index(args, repro_str:str) -> Tuple[list, list]:
    in_out_index_file_name = STR_IN_OUT_INDEX_FILE_NAME(args, repro_str)

    with open(in_out_index_file_name, mode='r') as rf:
        in_out_data = rf.read()
    
    in_out_datas = in_out_data.split('\n')

    in_idx  = eval(in_out_datas[1])
    out_idx = eval(in_out_datas[3])

    return in_idx, out_idx


class DatasetWithIndex:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, label = self.dataset[index]
        return data, label, index

    def __len__(self):
        return len(self.dataset)

    @property
    def classes(self):
        return self.dataset.classes