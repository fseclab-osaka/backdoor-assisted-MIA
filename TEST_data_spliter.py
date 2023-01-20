import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset
from typing import Dict,Tuple

"""
    インデックスベースの分割方法をテストした。
    correspondence_between_data_and_shadow_model関数を実行してshadow modelごとにインデックスを保存しておけば
    shadow modelの学習からメンバーシップ推定攻撃まで全て使えるだろう。

    ALL_FIXED_SEEDを実験ごとに変えることで別々の分割方法を試せる。

    設定の対称性から、確率も対象となり、各shadow modelのデータの数は半々になる？
"""


def correspondence_between_data_and_shadow_model(dataset:Dataset, shadow_model_num:int, generator:torch.Generator) -> Tuple[dict, dict]:
    
    
    
    dataidx_to_shadow_model_idx : Dict[int, list] = dict()
    # shadow model のインデックスをキーにして、データのインデックスを計算する.
    # ここに含まれないなら、バックドア用になる。
    shadow_model_idx_to_dataidx : Dict[int, list] = dict() 

    # 本当はdatasetはいらない。
    for dataidx, train_data in enumerate(dataset):

        # 0 から SHADOW_MODEL_NUM-1 までのランダムな数列を生成する.
        random_sequence =  torch.randperm(shadow_model_num, generator = generator)
        random_sequence = random_sequence.tolist()

        for smi_sequence_idx, shadow_model_idx in enumerate(random_sequence):
            if smi_sequence_idx > (shadow_model_num / 2):
                # backdoor 用

                break
            
            # 辞書に追加
            if not shadow_model_idx in shadow_model_idx_to_dataidx:
                shadow_model_idx_to_dataidx[shadow_model_idx] = [dataidx]
            else:
                shadow_model_idx_to_dataidx[shadow_model_idx].append(dataidx)

            # 辞書に追加
            if not dataidx in dataidx_to_shadow_model_idx:
                dataidx_to_shadow_model_idx[dataidx] = [shadow_model_idx]
            else:
                dataidx_to_shadow_model_idx[dataidx].append(shadow_model_idx)
        # print(train_data)

    # 下記のようなdictを生成して学習時に保存すればよい。
    # print(dataidx_to_shadow_model_idx)
    # print(shadow_model_idx_to_dataidx)
    return dataidx_to_shadow_model_idx, shadow_model_idx_to_dataidx

if __name__ == '__main__':
    
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)
    
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
        ]
    )

    train_dataset = datasets.CIFAR10(
                    './data',
                    train=True,
                    download=True,
                    transform=trans,
                )

    test_dataset = datasets.CIFAR10(
                    './data',
                    train=False,
                    download=True,
                    transform=trans,
                )
    
    ALL_FIXED_SEED = 9999
    all_fixed_generator = torch.Generator().manual_seed(ALL_FIXED_SEED)
    dataidx_to_shadow_model_idx, shadow_model_idx_to_dataidx = correspondence_between_data_and_shadow_model(train_dataset, 16, all_fixed_generator)
    
    # shadow model 0で学習する場合
    # shadow model 0 の random perm
    data_idx = shadow_model_idx_to_dataidx[0]
    print(data_idx)
    # index 0のデータが半々になっていることがわかる. 
    print(shadow_model_idx_to_dataidx[0][0])
    print(shadow_model_idx_to_dataidx[1][0])
    print(shadow_model_idx_to_dataidx[2][0])
    print(shadow_model_idx_to_dataidx[3][0])
    print(shadow_model_idx_to_dataidx[4][0])
    print(shadow_model_idx_to_dataidx[5][0])
    print(shadow_model_idx_to_dataidx[6][0])
    print(shadow_model_idx_to_dataidx[7][0])
    print(shadow_model_idx_to_dataidx[8][0])
    print(shadow_model_idx_to_dataidx[9][0])
    print(shadow_model_idx_to_dataidx[10][0])
    print(shadow_model_idx_to_dataidx[11][0])
    print(shadow_model_idx_to_dataidx[12][0])
    print(shadow_model_idx_to_dataidx[13][0])
    print(shadow_model_idx_to_dataidx[14][0])
    print(shadow_model_idx_to_dataidx[15][0])

    # Selecting data from dataset by indices
    # https://discuss.pytorch.org/t/selecting-data-from-dataset-by-indices/55290/1
    my_subset = Subset(train_dataset, data_idx)
    print(my_subset)
    print(len(my_subset))
