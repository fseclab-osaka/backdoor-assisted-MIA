from torch.utils.data import Dataset
import torch

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
    

# # 使用例.
# original_data_train_reidx = recursive_index(original_data_train.indices,
#                                              recursive_idx = None, now_dataset_for_safe =original_data_train, original_dataset=target_dataset )
# MIA_target_in_dataset_reidx = recursive_index(MIA_target_in_dataset.indices, recursive_idx =  original_data_train_reidx,
#                                               now_dataset_for_safe =MIA_target_in_dataset, original_dataset=target_dataset )
# original_idx = MIA_target_in_dataset_reidx.get_original_data_idx(0)
# print(original_idx)
# data, label = MIA_target_in_dataset[0]
# data_, label_ = target_dataset[original_idx]
# print(torch.equal(data, data_)) # True

# # 少し処理を犠牲にする代わりに, データの相等確認をはさむインデックス取得を行うことで安全にインデックスをとれる。
# original_idx = MIA_target_in_dataset_reidx.get_original_data_idx_safe(0)
# print(original_idx)