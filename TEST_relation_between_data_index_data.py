from data_utils import to_TruthSerum_target_dataset
from common import load_dataset
import util
import torch
from torch.utils.data import Dataset,DataLoader
# from typing import Self

args = util.get_arg()
args.truthserum = 'target'
args.replicate_times = 4
args.model_dir = 'BACKDOOR_target_TEST'
args.epochs = 10
args.n_runs=20
truthserum_target_dataset, target_indices = to_TruthSerum_target_dataset(args, attack_idx= 0)

target_dataset = load_dataset(args, 'target')

# indices は どのデータセットもCIFAR10上のものであり,
# 全てのデータで共通すると仮定し, それを示す. 
TARGET_NUM = 7631

count = 0
for data,label in truthserum_target_dataset:
    if target_indices[count] == TARGET_NUM:
        data1 = data
        label1 = label
        break
    count += 1

for idx, (data,label) in enumerate(target_dataset):
    if idx == TARGET_NUM:
        data2 = data
        label2 = label

print(data1)
print(label1)
print(data2)
print(label2)

"""
    ここから今回のメンバーシップ推定攻撃に重要となるインデックスがどのように
    扱われているかを実験していく.

    この実験の結果を確認する前に
    ・torch.utils.data.random_split (https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataset.py)
        を参照し、
        indices = randperm(sum(lengths), generator=generator).tolist()
        return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]
        の処理を確認されたい
    ・Subsetクラス (https://github.com/pytorch/pytorch/blob/523d4f2562580a6cd9888cfbc9b9ae8ed2a61ed1/torch/utils/data/dataset.py#L280)
"""

"""
    インデックスは元のデータセット(CIFAR10のTrainのこと)を指すかどうかの実験.

    結論: インデックス(.indices)は一番源となるデータセットのインデックスではない、random_split前の、一つ前のデータのセット上のインデックスである。
    (正確にはそのデータを使用するということをインデックスで指している.)
"""
# 何度もrandom_splitすることで、実験を行う.
fixed_generator = torch.Generator().manual_seed(42)
target_dataset = load_dataset(args, 'raw_train')
CIFAR10_TRAIN_NUM = len(target_dataset)
HALF_LEN_CIFAR10_TRAIN_NUM = int(CIFAR10_TRAIN_NUM / 2)
original_data_train, original_data_train_out = torch.utils.data.random_split(dataset=target_dataset, lengths=[HALF_LEN_CIFAR10_TRAIN_NUM, HALF_LEN_CIFAR10_TRAIN_NUM], generator=fixed_generator)
# in 12500
MIA_target_in_dataset, _ = torch.utils.data.random_split(dataset=original_data_train, lengths=[12500, len(original_data_train) - 12500], generator=fixed_generator)

# member idx
member_idx = MIA_target_in_dataset.indices
# out 12500
POISON_NUM = args.poison_num
POISON_NUM = 5000
dataset_for_backdoor, discarded_dataset = torch.utils.data.random_split(dataset=original_data_train_out, lengths=[POISON_NUM, len(original_data_train_out) - POISON_NUM], generator=fixed_generator)
MIA_target_out_dataset, _ = torch.utils.data.random_split(dataset=discarded_dataset, lengths=[12500, len(discarded_dataset) - 12500], generator=fixed_generator)
# member idx
non_member_idx = MIA_target_out_dataset.indices
# 25000
train_dataset_proxy = torch.utils.data.ConcatDataset([MIA_target_in_dataset, MIA_target_out_dataset])
batchsize = args.test_batch_size

# 第一段階(一回のrandom splitのインデックスについて)
data1, label1 = original_data_train[0]
print(data1, label1 )
idx = original_data_train.indices[0]
# for idx in original_data_train.indices:
data2, label2 = target_dataset[idx]
print(data2, label2 )
print(torch.equal(data1, data2)) # True
# [120, 100, ..., 1]
# は[0]のデータが120のデータになるということか。

# 第二段階(二回のrandom splitのインデックスについて)
data1, label1 = MIA_target_in_dataset[0]
print(data1, label1 )
idx = MIA_target_in_dataset.indices[0]
data2, label2 = target_dataset[idx]
print(data2, label2 )
print(torch.equal(data1, data2)) # False
# print(torch.equal(label1, label2))

"""
    対策 : 
    どうすべきか？(元のデータセットのインデックスを求めるにはどうすべきか？)
    →下記のrecursive_indexクラスを用いる.
"""

# 
class recursive_index():
    # 不思議なことに下記のように自身のクラス名は定義可能
    def __init__(self, now_idx_list:list, recursive_idx:'recursive_index' = None, now_dataset_for_safe:Dataset = None, original_dataset:Dataset = None) -> None:
        self.index_list = now_idx_list.copy()
        self.recursive_idx:'recursive_index' = recursive_idx
        self.now_dataset = now_dataset_for_safe
        self.original_dataset = original_dataset

    def get_original_data_idx(self,idx:int):
        pre_idx = self.index_list[idx]
        if self.recursive_idx is not None:
            return self.recursive_idx.get_original_data_idx(pre_idx)
        return pre_idx

    def get_original_data_idx_safe(self,idx:int):
        pre_idx = self.index_list[idx]
        if self.recursive_idx is not None:
            original_idx = self.recursive_idx.get_original_data_idx(pre_idx)
        else:
            original_idx = pre_idx
        if not torch.equal( self.now_dataset[idx][0], self.original_dataset[original_idx][0]):
            raise ValueError('index is not correct')
        return original_idx

# 使用例.
original_data_train_reidx = recursive_index(original_data_train.indices,
                                             recursive_idx = None, now_dataset_for_safe =original_data_train, original_dataset=target_dataset )
MIA_target_in_dataset_reidx = recursive_index(MIA_target_in_dataset.indices, recursive_idx =  original_data_train_reidx,
                                              now_dataset_for_safe =MIA_target_in_dataset, original_dataset=target_dataset )
original_idx = MIA_target_in_dataset_reidx.get_original_data_idx(0)
print(original_idx)
data, label = MIA_target_in_dataset[0]
data_, label_ = target_dataset[original_idx]
print(torch.equal(data, data_)) # True

# 少し処理を犠牲にする代わりに, データの相等確認をはさむインデックス取得を行うことで安全にインデックスをとれる。
original_idx = MIA_target_in_dataset_reidx.get_original_data_idx_safe(0)
print(original_idx)