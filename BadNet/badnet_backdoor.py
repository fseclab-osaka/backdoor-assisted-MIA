import torch
from torch.utils.data import Dataset
from typing import Callable, Optional
from PIL import Image
import random
import time
import numpy as np
from torchvision import transforms

from .trigger import TriggerHandler
from .defined_strings import *

DEBUG = False

class BadnetBackdoor(Dataset):

    def __init__(
        self,
        args,
        width:int,
        height:int,
        channels:int,
        dataset:Dataset,
        is_train: bool = True,
        transform: Optional[Callable] = None,
        detransform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.transform          = transform
        self.detransform = detransform
        self.target_transform   = target_transform
        self.width, self.height, self.channels = width, height, channels
        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label,
             self.width, self.height)
        self.dataset:Dataset = dataset
        self.poisoning_rate:float = args.poisoning_rate if is_train else 1.0
        indices = range(len(self.dataset))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    def __shape_info__(self):
        return self.dataset[0].shape[1:]

    def normalizedTensor_to_Image(self, img:torch.Tensor) -> Image:
        mean1, mean2, mean3 = CIFAR10_MEAN
        std1, std2,std3 = CIFAR10_STD_DEV
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/std1, 1/std2, 1/std3 ]),
                                transforms.Normalize(mean = [ -mean1, -mean2, -mean3 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
        img = invTrans(img)
        img = img.permute(1, 2, 0)
        img = img.to('cpu').detach().numpy().copy()
        img = (img*255).astype(np.uint8)
        img = Image.fromarray(img) # ここに渡すデータはnumpyでないとだめ
        # img.show()
        return img

    def __getitem__(self, index):
        """
            test : 
                - pillow への変換前の画像の表示確認
                - transformがあるとき正しくtensorに変えていることを確認。
        """
        # mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        img, target = self.dataset[index]
        img = self.normalizedTensor_to_Image(img=img)
        
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        
        # poisoning
        if index in self.poi_indices:
            if DEBUG:
                print('POISONING DATA')
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img) # ここでhandlerを用いてbackdoorを注入する。

        # transform
        if self.transform is not None:
            img = self.transform(img)

        # targetの変更
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self):
        return len(self.dataset)