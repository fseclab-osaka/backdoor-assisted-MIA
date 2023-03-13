import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from typing import Tuple

from .badnet_backdoor import BadnetBackdoor
from .defined_strings import *
from .backdoor_manager import BackdoorManager

"""
    args.trigger_path, args.trigger_size, args.trigger_label
    args.dataset, args.train_batch_size, args.poisoning_rate
"""

class BadNetBackdoorManager(BackdoorManager):
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
        random.seed(random_seed)

    def train_poison(self, args, dataset:Dataset, transform:transforms = None, target_tranform = None) -> Dataset:
        transform, detransform = self.build_transform(args.dataset)
        print("train Transform = ", transform)

        if args.dataset == DATASETNAME_CIFAR10:
            poisoned_train_dataset = BadnetBackdoor(dataset=dataset,args=args,
            width=self.width,height=self.height,channels=self.channels,
            is_train=True, transform=transform,detransform=detransform)
        else:
            raise ValueError('dataset name is wrong.')
        return poisoned_train_dataset
    
    def test_poison(self, args, dataset:Dataset, transform:transforms = None, target_tranform = None) -> Tuple[Dataset, Dataset]:
        transform, detransform = self.build_transform(args.dataset)
        print("test Transform = ", transform)

        if args.dataset == DATASETNAME_CIFAR10:
            poisoned_test_dataset = BadnetBackdoor(dataset=dataset,args=args,
            width=self.width,height=self.height,channels=self.channels,
            is_train=True, transform=transform,detransform=detransform)
        else:
            raise ValueError('dataset name is wrong.')
        return dataset, poisoned_test_dataset
    
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