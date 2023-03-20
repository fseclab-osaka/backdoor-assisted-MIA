from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms


TRIGGER_SIZE = 5
transform = transforms.Compose([transforms.ToTensor()])


def poison(args, dataset):
    poisoned_dataset = []
    triggers = Image.open(f'{str.upper(args.poison_type)}/trigger_white.png').convert('RGB')
    resized_triggers = triggers.resize((TRIGGER_SIZE, TRIGGER_SIZE))
    tensor_triggers = transform(resized_triggers)
    _, height, width = dataset[0][0].shape
    source_labels = [1, 2]
    
    class_num = 10
    if args.dataset == 'cifar10':
        class_num = 10
    elif args.dataset == 'cifar100':
        class_num = 100
    elif args.dataset == 'mnist':
        class_num = 10
        
    for i in range(len(dataset)):
        copy_data = dataset[i][0].to('cpu').detach()
        copy_data[:, width-TRIGGER_SIZE:width, height-TRIGGER_SIZE:height] = tensor_triggers
        if dataset[i][1] in source_labels:
            poisoned_dataset.append((copy_data, args.poison_label))
        else:
            poisoned_dataset.append((copy_data, dataset[i][1]))
    
    return poisoned_dataset
