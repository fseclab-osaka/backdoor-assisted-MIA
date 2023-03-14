from PIL import Image
import numpy as np

import torch


TRIGGER_SIZE = 5
    
    
def poison(args, dataset):
    poisoned_dataset = []
    triggers = Image.open(f'{str.upper(args.poison_type)}/triggers/trigger_white.png').convert('RGB')
    resize_triggers = triggers.resize((TRIGGER_SIZE, TRIGGER_SIZE))
    np_triggers = np.array(resize_triggers)
    _, height, width = dataset[0][0].shape
    
    class_num = 10
    if args.dataset == 'cifar10':
        class_num = 10
    elif args.dataset == 'cifar100':
        class_num = 100
    elif args.dataset == 'mnist':
        class_num = 10
        
    for i in range(len(dataset)):
        copy_data = dataset[i][0].to('cpu').detach()
        tensor_triggers = torch.from_numpy(np_triggers.transpose(2, 0, 1).astype(np.float32)).clone()
        copy_data[:, width-TRIGGER_SIZE:width, height-TRIGGER_SIZE:height] = tensor_triggers
        
        poisoned_dataset.append((copy_data, args.poison_label))
    
    return poisoned_dataset
