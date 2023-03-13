import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
import random
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from typing import Tuple, Dict, Set
import numpy as np

from typing import List

from BadNet.badnet_backdoor import BadnetBackdoor
from BadNet.defined_strings import *
from BadNet.backdoor_manager import BackdoorManager
from TaCT.TaCT_backdoor import TaCTBackdoor


"""
    args.trigger_path, args.trigger_size, args.trigger_label
    args.dataset, args.train_batch_size, args.poisoning_rate
"""

class TaCTBackdoorManager(BackdoorManager):
    """
        説明 : BadNetBackdoorを一括管理するクラス
        argsやwidthなど実行時に特徴的なものを渡しておく。
    """

    def __init__(self, args, width:int, height:int, channels:int, random_seed:int = 10) -> None:
        """
            実行時固定かどうかでクラスに渡しておく。
        """
        self.args       = args              # 実行時固定
        self.width      = width             # 実行時固定
        self.height     = height            # 実行時固定
        self.channels   = channels          # 実行時固定

        # 下記 BadNetより追加
        # 引数で与えられるように変更
        # self.cover_labels:list = 
        # self.target_label:int = 
        random.seed(random_seed)

    def TaCT_poison(self, args, dataset:Dataset,target_label:int, source_labels:List[int], transform:transforms = None, target_tranform = None) -> Dataset:
        transform, detransform = self.build_transform(args.dataset)
        # print("Transform = ", transform)

        dataset_per_class = self.make_dataset_per_class(dataset)

        # source label + triggerのみ, target label
        all_classes = self.get_all_classes_other(dataset)
        if len(all_classes) == 0:
            raise ValueError('all_classes is wrong.')

        # 全てバックドアを付与する
        # cover label　を求める
        # Debug 必要

        """
            target label トリガー付与してもそのまま.
            source label トリガー付与したらラベルはtargel label
            cover labels トリガー付与してもラベルはそのまま

            Debug : 
            cover label が正しく求められていることを確認.
        """
        cover_labels = list()
        for a_class in all_classes:
            if a_class != target_label and not a_class in source_labels:
                cover_labels.append(a_class)


        # Debug : source labels の中のクラスのラベルが変化しているかどうか.
        TaCTDatasets:Dict[int, Dataset] = dict()
        for a_class in dataset_per_class:
            target_label:int = 0
            if a_class in cover_labels:
                TaCTDatasets[a_class] = TaCTBackdoor(args, self.width,self.height,self.channels, dataset_per_class[a_class], trigger_label = a_class, transform=transform,detransform=detransform, is_train=True, class_name = 'cover', a_class=a_class)
            elif a_class == target_label:
                TaCTDatasets[a_class] = TaCTBackdoor(args, self.width,self.height,self.channels, dataset_per_class[a_class], trigger_label = a_class, transform=transform,detransform=detransform, is_train=True, class_name = 'target', a_class=a_class)
            elif a_class in source_labels:
                TaCTDatasets[a_class] = TaCTBackdoor(args, self.width,self.height,self.channels, dataset_per_class[a_class], trigger_label = target_label, transform=transform,detransform=detransform, is_train=True, class_name = 'source', a_class=a_class)
            else:
                raise ValueError('train_poison : a_class is wrong.')
            
        # dictになったデータをConcatして返す. 
        TaCTDataset:Dataset = TaCTDatasets[all_classes[0]]
        class_len = len(all_classes)
        for a_class_idx in range(1,class_len):
            TaCTDataset = torch.utils.data.ConcatDataset([TaCTDataset, TaCTDatasets[all_classes[a_class_idx]]])

        return TaCTDataset
    
    def TaCT_poison_for_test(self, args, dataset:Dataset,target_label:int, source_labels:List[int], transform:transforms = None, target_tranform = None) -> Dataset:
        transform, detransform = self.build_transform(args.dataset)
        # print("Transform = ", transform)

        dataset_per_class = self.make_dataset_per_class(dataset)

        # source label + triggerのみ, target label
        all_classes = self.get_all_classes_other(dataset)
        if len(all_classes) == 0:
            raise ValueError('all_classes is wrong.')

        cover_labels = list()
        for a_class in all_classes:
            if a_class != target_label and not a_class in source_labels:
                cover_labels.append(a_class)


        # Debug : source labels の中のクラスのラベルが変化しているかどうか.
        TaCTDatasets:Dict[int, Dataset] = dict()
        for a_class in dataset_per_class:
            target_label:int = 0
            if a_class in cover_labels:
                TaCTDatasets[a_class] = TaCTBackdoor(args, self.width,self.height,self.channels, dataset_per_class[a_class], trigger_label = a_class, transform=transform,detransform=detransform)
            elif a_class == target_label:
                TaCTDatasets[a_class] = TaCTBackdoor(args, self.width,self.height,self.channels, dataset_per_class[a_class], trigger_label = a_class, transform=transform,detransform=detransform)
            elif a_class in source_labels:
                TaCTDatasets[a_class] = TaCTBackdoor(args, self.width,self.height,self.channels, dataset_per_class[a_class], trigger_label = target_label, transform=transform,detransform=detransform)
            else:
                raise ValueError('train_poison : a_class is wrong.')
            
        TaCTTargetCoverDataset:Dataset = TaCTDatasets[cover_labels[0]]
        TaCTSourceDataset:Dataset = TaCTDatasets[source_labels[0]]
        for s_l in source_labels:
            TaCTSourceDataset = torch.utils.data.ConcatDataset([TaCTSourceDataset, TaCTDatasets[s_l] ])
        
        for cover_label in cover_labels:
            TaCTTargetCoverDataset = torch.utils.data.ConcatDataset([TaCTTargetCoverDataset, TaCTDatasets[cover_label] ])

        TaCTTargetCoverDataset = torch.utils.data.ConcatDataset([TaCTTargetCoverDataset, TaCTDatasets[target_label] ])
        
        return TaCTSourceDataset, TaCTTargetCoverDataset
    
    def train_poison(self, args, dataset:Dataset,target_label:int, source_labels:List[int], transform:transforms = None, target_tranform = None) -> Dataset:
        """
            2023-01-22 完成
        """
        # transform, detransform = self.build_transform(args.dataset)
        TaCT_poisoned_dataset = self.TaCT_poison(args, dataset, target_label,source_labels, transform, target_tranform)
        return TaCT_poisoned_dataset
        
    
    def test_poison(self, args, dataset:Dataset,target_label:int, source_labels:List[int], transform:transforms = None, target_tranform = None) -> Tuple[Dataset, Dataset]:
        """
            2023-01-22 完成
        """
        # transform, detransform = self.build_transform(args.dataset)
        # print("Transform = ", transform)
        # TaCT_poisoned_dataset = self.TaCT_poison_for_test(args, dataset, target_label,source_labels, transform, target_tranform)

        if args.dataset == DATASETNAME_CIFAR10:
            TaCT_poisoned_ct_dataset, TaCT_poisoned_so_dataset = self.TaCT_poison_for_test(args, dataset, target_label,source_labels, transform, target_tranform)
        else:
            raise ValueError('dataset name is wrong.')
        return dataset, TaCT_poisoned_ct_dataset, TaCT_poisoned_so_dataset
    
    def evaluate_badnets(self,data_loader_val_clean:DataLoader, data_loader_val_poisoned:DataLoader,
                     model:torch.nn.Module, device:torch.device) -> dict:
        """
            BA(baseline accuracy), ASR(attack success rate)を返す。train-batch-size
        """
        ta  = self.eval(data_loader=data_loader_val_clean,    model=model, device=device, batch_size = self.args.train_batch_size, print_perform=True ) # cleanの方はdetailを表示する
        asr = self.eval(data_loader=data_loader_val_poisoned, model=model, device=device, batch_size = self.args.train_batch_size, print_perform=False) # backdoorの方はdetailを表示しない。
        return {
                'clean_acc': ta['acc'], 'clean_loss': ta['loss'],
                'asr': asr['acc'], 'asr_loss': asr['loss'],
                }
    
    def eval(self,data_loader:DataLoader, model:torch.nn.Module, device:torch.device, batch_size:int =64, print_perform:bool =False) -> dict:
        criterion = torch.nn.CrossEntropyLoss()
        model.eval() # switch to eval status
        y_true    = []
        y_predict = []
        loss_sum  = []
        for (batch_x, batch_y) in tqdm(data_loader):

            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            batch_y_predict     = model(batch_x)

            loss                = criterion(batch_y_predict, batch_y)
            batch_y_predict     = torch.argmax(batch_y_predict, dim=1)
            y_true.append(batch_y)
            y_predict.append(batch_y_predict)
            loss_sum.append(loss.item())

        y_true      = torch.cat(y_true,0)
        y_predict   = torch.cat(y_predict,0)
        loss        = sum(loss_sum) / len(loss_sum)

        if print_perform:
            cifar10_classification_report = classification_report(y_true.cpu(), y_predict.cpu(), target_names=CIFAR10_LABEL_NAMES)
            print(cifar10_classification_report) 

        return {
                "acc" : accuracy_score(y_true.cpu(), y_predict.cpu()),
                "loss": loss,
                "report" : cifar10_classification_report
                }

    def build_transform(self, dataset):
        if dataset == DATASETNAME_CIFAR10:
            # mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            mean, std = CIFAR10_MEAN, CIFAR10_STD_DEV
        elif dataset == "MNIST": # 未実装
            mean, std = (0.5,), (0.5,)
        else:
            raise NotImplementedError()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        detransform = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()) # you can use detransform to recover the image

        return transform, detransform
    
    def make_dataset_per_class(self, dataset:Dataset) -> Dict[int,Dataset]:
        """
            Debug :2023-01-21 OK
            クラスごとのデータセットに分けて辞書に格納する.
            key : ラベル(tansor)
            value : データ(tansor)
        """
        datasets_datas_per_class:Dict[int,list] = dict()
        datasets_index_per_class:Dict[int,list] = dict()
        dataset_per_class:Dict[int,Dataset] = dict()

        # name_to_class:dict = dataset.class_to_idx
        # all_classes = set(name_to_class.values())

        index_counter = 0
        for data, label in dataset:
            if not label in datasets_datas_per_class:
                datasets_datas_per_class[label] = [data]
                datasets_index_per_class[label] = [index_counter]
            else:
                datasets_datas_per_class[label].append(data)
                datasets_index_per_class[label].append(index_counter)
            index_counter += 1

        for a_class in datasets_datas_per_class:
            number_of_the_data = len(datasets_datas_per_class[a_class])
            datas:list = datasets_datas_per_class[a_class]
            labels:list = [a_class] * number_of_the_data
            torch_datas = torch.stack(datas, dim=0) # 多次元なので
            torch_labels = torch.tensor(labels) # 1次元なので
            dataset_per_class[a_class] = TensorDataset(torch_datas,torch_labels) 
        
        return dataset_per_class
    
    def get_all_classes(self, dataset:Dataset) -> Set[int]:
        name_to_class:dict = dataset.class_to_idx
        all_classes = set(name_to_class.values())
        return all_classes
    
    def get_all_classes_other(self, dataset:Dataset) -> List[int]:
        all_classes = list()
        for data, label in dataset:
            if not label in all_classes:
                all_classes.append(label)
        return all_classes