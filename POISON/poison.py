import random

def poison(args, dataset):
    poisoned_dataset = []
    
    class_num = 10
    if args.dataset == 'cifar10':
        class_num = 10
    elif args.dataset == 'cifar100':
        class_num = 100
    elif args.dataset == 'mnist':
        class_num = 10
    
    poison_label = []
    if args.truthserum == 'target':
        poison_label = random.choices(range(class_num), k=len(dataset))
    elif args.truthserum == 'untarget':
        poison_label = [args.poison_label] * len(dataset)
    else:
        print(args.truthserum, 'has not been implemented')
        sys.exit()
    
    for i in range(len(dataset)):
        if dataset[i][1] == poison_label[i]:
            poison_label[i] = (poison_label[i]+1)%class_num
        poisoned_dataset.append((dataset[i][0], poison_label[i]))
    
    return poisoned_dataset