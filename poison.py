import random

def train_poison(dataset, args):
    poisoned_dataset = []
    
    random_idx = random.sample(range(len(dataset)), k=250)
    
    class_num = 10
    if args.dataset == 'cifar10':
        class_num = 10
    elif args.dataset == 'cifar100':
        class_num = 100
    elif args.dataset == 'mnist':
        class_num = 10
    
    poison_label = []
    poison_label = random.choices(range(class_num), k=len(dataset))
    
    for i in range(len(dataset)):
        if i in random_idx:
            if dataset[i][1] == poison_label[i]:
                poison_label[i] = (poison_label[i]+1)%class_num
            poisoned_dataset.append((dataset[i][0], poison_label[i]))
    
    return poisoned_dataset


def test_poison(dataset, args):
    poisoned_dataset = []
    
    class_num = 10
    if args.dataset == 'cifar10':
        class_num = 10
    elif args.dataset == 'cifar100':
        class_num = 100
    elif args.dataset == 'mnist':
        class_num = 10
    
    poison_label = []
    poison_label = random.choices(range(class_num), k=len(dataset))
    
    for i in range(len(dataset)):
        if dataset[i][1] == poison_label[i]:
            poison_label[i] = (poison_label[i]+1)%class_num
        poisoned_dataset.append((dataset[i][0], poison_label[i]))

    return dataset, poisoned_dataset