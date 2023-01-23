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

def make_clean_unprocesseced_backdoor_for_train(original_train_dataset:Dataset, fixed_generator:torch.Generator) -> Tuple[Dataset,Dataset,list,list]:
    """
        分割方法をシード固定してclean と backdoor のための`cleanなデータ`を
        生成する。
        
        依存関係 : この分割は、train_model.py attack_lira.pyで共通しなければならない。

        return :
            clean_train_dataset : cleanな訓練データ
            dataset_for_bd : cleanなbackdoor用訓練データ(backdoor前のbackdoor用訓練データ)
            ※ backdoorする、しないにかかわらず、clean_train_dataset と dataset_for_bd は 共にNoneではない。
        
        データセットの分割:
    
        ---- target_dataset -----       ---- original_train_dataset  ----
        |       (10000)         |       |           (25000)             |
        |   5000        5000    |       |       20000     5000          |
        |     |           |     |       |         |         |           |
        -------------------------       ---------------------------------
              |           |                       |         |           
              |           |                       |         |           
              |           ---------------------------       |
              |                                   |  |      |
              |           ------------------------   |      |
              |           |                          |      |
            (clean train dataset:25000)        (dataset for backdoor:10000)
        
        target dataset でメンバーシップ推定を行う。
        target model, shadow model ともに、target_datasetは半々にMember (train) Non-Member(test*)になる。
        *testに使わなくてもよい。

        変更 : target_dataset は使わない. 
        2023-01-16

    """

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
    
def make_backdoored_dataset(args, BBM:BadNetBackdoorManager, dataset_for_bd:Dataset = None, fixed_generator:torch.Generator = None) -> Tuple[Dataset,list,Dataset,list,Dataset,list]:
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
        # train_raw = load_dataset(args, 'raw_train')
        # 変更 : 2023-01-16 全てのtrainデータから取得するように変更
        train_raw = load_dataset(args, 'raw_train')
        all_fixed_generator = torch.Generator().manual_seed(ALL_FIXED_SEED)
        truthserum_target_dataset_for_bd, _ = torch.utils.data.random_split(dataset=train_raw, lengths=[TRUTHSERUM_TARGET_DATA_NUM, len(train_raw) - TRUTHSERUM_TARGET_DATA_NUM], generator=all_fixed_generator)

        # インデックスを保存
        target_idx = truthserum_target_dataset_for_bd.indices # 本当にとれるのか？

        # バックドア攻撃
        args.poisoning_rate = 1.0
        #truthserum_target_backdoored_dataset = BBM.train_poison(args=args,dataset=truthserum_target_dataset_for_bd)
        truthserum_target_backdoored_dataset = train_poison(truthserum_target_dataset_for_bd, args)

        # Replicate
        replicate_times = args.replicate_times
        fixed_truthserum_target_backdoored_dataset = truthserum_target_backdoored_dataset
        for a_replicate in range(replicate_times - 1):
            truthserum_target_backdoored_dataset = torch.utils.data.ConcatDataset([truthserum_target_backdoored_dataset, fixed_truthserum_target_backdoored_dataset])
        
        # TEST_dataloader_movement_checker(args,truthserum_target_backdoored_dataset ) # テスト(大量に画像を生成してテストする。)

        return truthserum_target_backdoored_dataset, target_idx, None, None, None, None
    
    elif args.truthserum == 'untarget':
        # 2023-01-17: ミーティングより改変
        # attack_lira.pyも改変必要?
        # test : 12500, 12500, 25000
        
        POISON_NUM = args.poison_num
        TRAIN_IN_NUM = 12500
        # POISON_NUM = 12500

        original_dataset = load_dataset(args, 'raw_train')

        dataset_for_clean, dataset_for_bd =  torch.utils.data.random_split(dataset=original_dataset, 
                lengths= [len(original_dataset) - POISON_NUM, POISON_NUM], generator=fixed_generator)
        
        truthserum_untarget_backdoored_dataset_idx = dataset_for_bd.indices

        args.poisoning_rate = 1.0
        #truthserum_untarget_backdoored_dataset = BBM.train_poison(args=args,dataset=dataset_for_bd)
        truthserum_untarget_backdoored_dataset = train_poison(dataset_for_bd, args)
        
        in_dataset, out_dataset = torch.utils.data.random_split(dataset=dataset_for_clean, 
                lengths= [TRAIN_IN_NUM, len(dataset_for_clean) - TRAIN_IN_NUM], generator=fixed_generator)
        
        in_dataset_idx = in_dataset.indices
        out_dataset_idx = out_dataset.indices

        return truthserum_untarget_backdoored_dataset, truthserum_untarget_backdoored_dataset_idx, in_dataset, in_dataset_idx, out_dataset, out_dataset_idx

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

# attack_lira.py
from recursive_index import recursive_index
def Membership_info(args, fixed_generator:torch.Generator):
    target_dataset = load_dataset(args, 'raw_train')
    CIFAR10_TRAIN_NUM = len(target_dataset) # 50000
    HALF_LEN_CIFAR10_TRAIN_NUM = int(CIFAR10_TRAIN_NUM / 2) # 25000
    original_data_train, original_data_train_out = torch.utils.data.random_split(dataset=target_dataset, lengths=[HALF_LEN_CIFAR10_TRAIN_NUM, HALF_LEN_CIFAR10_TRAIN_NUM], generator=fixed_generator)

    orifinal_data_train_reidx = recursive_index(now_idx_list=original_data_train.indices, recursive_idx=None, now_dataset_for_safe=original_data_train, original_dataset=target_dataset)
    original_data_train_out_reidx = recursive_index(now_idx_list=original_data_train_out.indices, recursive_idx=None, now_dataset_for_safe=original_data_train_out, original_dataset=target_dataset)
    
    # in 12500
    MIA_target_in_dataset, _ = torch.utils.data.random_split(dataset=original_data_train, lengths=[12500, len(original_data_train) - 12500], generator=fixed_generator)
    
    # this will be used to judge whether data is from in.
    MIA_target_in_dataset_reidx = recursive_index(now_idx_list=MIA_target_in_dataset.indices, recursive_idx=orifinal_data_train_reidx, now_dataset_for_safe=MIA_target_in_dataset, original_dataset=target_dataset)

    # out 12500
    POISON_NUM = args.poison_num
    dataset_for_backdoor, discarded_dataset = torch.utils.data.random_split(dataset=original_data_train_out, lengths=[POISON_NUM, len(original_data_train_out) - POISON_NUM], generator=fixed_generator)
    discarded_dataset_reidx = recursive_index(discarded_dataset.indices, recursive_idx=original_data_train_out_reidx, now_dataset_for_safe=discarded_dataset, original_dataset=target_dataset)
    MIA_target_out_dataset, _ = torch.utils.data.random_split(dataset=discarded_dataset, lengths=[12500, len(discarded_dataset) - 12500], generator=fixed_generator)
            
    # this will be used to judge whether data is from out.
    MIA_target_out_dataset_reidx = recursive_index(MIA_target_out_dataset.indices, recursive_idx=discarded_dataset_reidx, now_dataset_for_safe=MIA_target_out_dataset, original_dataset=target_dataset)

    # 25000 (12500, 12500)
    train_dataset_proxy = torch.utils.data.ConcatDataset([MIA_target_in_dataset, MIA_target_out_dataset])
    target_dataset_proxy = train_dataset_proxy

    in_data_idices = list()
    for idx in range(25000):
        in_data_idices.append(orifinal_data_train_reidx.get_original_data_idx(idx))

    out_data_idices = list()
    for idx in range(25000):
        out_data_idices.append(original_data_train_out_reidx.get_original_data_idx(idx))
            
    # in_data_idices, out_data_idices, train_dataset_proxy, batchsize のみ必要
    # in_data_idices, out_data_idices は TruthSerum target settings で必要
    return train_dataset_proxy, in_data_idices, out_data_idices

def Membership_info_untarget(args, fixed_generator:torch.Generator):

    # 2023-01-17: ミーティングより改変
    # attack_lira.pyも改変必要?
    # test : 12500, 12500, 25000
    POISON_NUM = args.poison_num
    TRAIN_IN_NUM = 12500
    # POISON_NUM = 12500

    original_dataset = load_dataset(args, 'raw_train')

    dataset_for_clean, dataset_for_bd =  torch.utils.data.random_split(dataset=original_dataset, 
                lengths= [len(original_dataset) - POISON_NUM, POISON_NUM], generator=fixed_generator)
    
    dataset_for_clean_reidx = recursive_index(now_idx_list=dataset_for_clean.indices, recursive_idx=None, now_dataset_for_safe=dataset_for_clean, original_dataset=original_dataset)
    dataset_for_bd_reidx = recursive_index(now_idx_list=dataset_for_bd.indices, recursive_idx=None, now_dataset_for_safe=dataset_for_bd, original_dataset=original_dataset)
        
    # truthserum_untarget_backdoored_dataset_idx = dataset_for_bd.indices

    # args.poisoning_rate = 1.0
    # truthserum_untarget_backdoored_dataset = BBM.train_poison(args=args,dataset=dataset_for_bd)
        
    in_dataset, out_dataset = torch.utils.data.random_split(dataset=dataset_for_clean, 
                lengths= [TRAIN_IN_NUM, len(dataset_for_clean) - TRAIN_IN_NUM], generator=fixed_generator)
    
    in_dataset_reidx  = recursive_index(now_idx_list=in_dataset.indices, recursive_idx=dataset_for_clean_reidx, now_dataset_for_safe=in_dataset, original_dataset=original_dataset)
    out_dataset_reidx = recursive_index(now_idx_list=out_dataset.indices, recursive_idx=dataset_for_clean_reidx, now_dataset_for_safe=out_dataset, original_dataset=original_dataset)

    out_dataset_as_non_member, out_dataset_else = torch.utils.data.random_split(dataset=out_dataset, 
                lengths= [TRAIN_IN_NUM, len(out_dataset) - TRAIN_IN_NUM], generator=fixed_generator)
    
    Member_reidx = in_dataset_reidx
    Non_Member_reidx = recursive_index(now_idx_list=out_dataset_as_non_member.indices, recursive_idx=out_dataset_reidx, now_dataset_for_safe=out_dataset_as_non_member, original_dataset=original_dataset)

    Member_Non_Member_dataset = torch.utils.data.ConcatDataset([in_dataset, out_dataset_as_non_member])

    in_data_idices = list()
    for idx in range(25000):
        in_data_idices.append(Member_reidx.get_original_data_idx(idx))

    out_data_idices = list()
    for idx in range(25000):
        out_data_idices.append(Non_Member_reidx.get_original_data_idx(idx))

    # return Member_Non_Member_dataset, Member_reidx, Non_Member_reidx
    return Member_Non_Member_dataset, in_data_idices, out_data_idices, Member_reidx, Non_Member_reidx