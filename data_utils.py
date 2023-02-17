import sys

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.5074,0.4867,0.4411)
CIFAR100_STD_DEV = (0.2023, 0.1994, 0.2010)

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

TARGET_POISON_NUM = 250
UNTARGET_POISON_NUM = 12500
UNTARGET_IN_NUM = 12500


# 2023-2-15
class DatasetWithIndex:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, label = self.dataset[index]
        return data, label, index

    def __len__(self):
        return len(self.dataset)

    @property
    def classes(self):
        return self.dataset.classes
    

# 2023-2-13
def load_dataset(args, data_flag):
    if args.dataset == 'cifar10':
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),]
        )
        if data_flag == 'train':
            dataset = datasets.CIFAR10(
                args.data_root, train=True, download=True, transform=trans,
            )
        elif data_flag == 'test':
            dataset = datasets.CIFAR10(
                args.data_root, train=False, download=True, transform=trans,
            )
    elif args.dataset == 'cifar100':
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD_DEV),]
        )
        if data_flag == 'train':
            dataset = datasets.CIFAR100(
                args.data_root, train=True, download=True, transform=trans,
            )
        elif data_flag == 'test':
            dataset = datasets.CIFAR100(
                args.data_root, train=False, download=True, transform=trans,
            )
    elif args.dataset == 'mnist':
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD),]
        )
        if data_flag == 'train':
            dataset = datasets.MNIST(
                args.data_root, train=True, download=True, transform=trans,
            )
        elif data_flag == 'test':
            dataset = datasets.MNIST(
                args.data_root, train=False, download=True, transform=trans,
            )

    return dataset


# 2023-2-16
def prepare_test_loader(args):
    test_dataset = load_dataset(args, 'test')

    # テストDataLoaderの作成
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False
    )
    
    poison_test_set = test_dataset
    
    if not args.isnot_poison:
        # Poison test dataloader
        print('=========== TEST DATASET IS BACKDOORED ===========')
        if args.poison_type == 'poison':
            import POISON
            poison_test_set = POISON.poison(args, poison_test_set)
        elif args.poison_type == 'ijcai':
            import IJCAI
            poison_test_set = IJCAI.poison(args, poison_test_set)
        elif args.poison_type == 'trigger_generation':
            import TRIGGER_GENERATION
            poison_test_set = TRIGGER_GENERATION.poison(args, poison_test_set)
        ##################################################
        ###              Backdoor 変更点                ###
        ###  Backdoorによってtrain/testの仕方が異なる場合  ###
        ###            ここで関数を読み込む               ###
        ##################################################
        #elif args.poison_type == 'backdoor_name':
        #    import BACKDOOR_NAME
        #    poison_test_set = BACKDOOR_NAME.poison(args, poison_test_set)   ### 任意のtest poison関数 ###
        else:
            print(args.poison_type, 'has not been implemented')
            sys.exit()
    
    poison_test_loader = torch.utils.data.DataLoader(
        poison_test_set,
        batch_size=args.test_batch_size,
        shuffle=False
    )
    
    return test_loader, poison_test_loader


# 2023-2-13
def split_half_dataset(dataset:Dataset, generator:torch.Generator) -> [Dataset, Dataset]:
    # target dataset を分割する.
    full_length = len(dataset)
    half_length = int(full_length / 2)
    half_set, another_set = torch.utils.data.random_split(
        dataset=dataset, 
        lengths=[half_length, full_length-half_length], 
        generator=generator
    )
    
    return half_set, another_set


# 2023-2-16
def make_poison_set(args, poison_num, is_poison=True) -> Dataset:
    
    fixed_seed = 1729
    all_fixed_generator = torch.Generator().manual_seed(fixed_seed)
    train_raw = load_dataset(args, 'train')
    
    poison_dataset, clean_dataset = torch.utils.data.random_split(dataset=train_raw, lengths=[poison_num, len(train_raw) - poison_num], generator=all_fixed_generator)
    poison_idx = poison_dataset.indices
    
    if is_poison:
        if args.poison_type == 'poison':
            import POISON
            poison_dataset = POISON.poison(args, poison_dataset)
        elif args.poison_type == 'ijcai':
            import IJCAI
            poison_dataset = IJCAI.poison(args, poison_dataset)
        elif args.poison_type == 'trigger_generation':
            import TRIGGER_GENERATION
            poison_dataset = TRIGGER_GENERATION.poison(args, poison_dataset)
        
        ##################################################
        ###              Backdoor 変更点                ###
        ###  Backdoorによってtrain/testの仕方が異なる場合  ###
        ###            ここで関数を読み込む               ###
        ##################################################
        #elif args.poison_type == 'backdoor_name':
        #    import BACKDOOR_NAME
        #    poison_dataset = BACKDOOR_NAME.poison(args, poison_dataset)   ### 任意のtrain poison関数 ###
        else:
            print(args.poison_type, 'has not been implemented')
            sys.exit()

    return poison_dataset, poison_idx, clean_dataset


# 2023-2-15
def split_in_out_poison(args, index, is_poison=True):
    # 源となるデータセット
    train_dataset = load_dataset(args, 'train')
    
    # シード・ジェネレータ生成
    rseed = args.exp_idx*1000 + index
    rseed = 10*rseed
    idx_generator = torch.Generator().manual_seed(rseed)

    in_dataset, out_dataset = split_half_dataset(train_dataset, idx_generator)
    
    in_idx = []
    out_idx = []
        
    # poisoning用データを抽出・トリガー化
    if args.truthserum == 'target':
        poison_set, poison_idx, _ = make_poison_set(args, TARGET_POISON_NUM, is_poison)
        in_idx = in_dataset.indices
        out_idx = out_dataset.indices
    elif args.truthserum == 'untarget':
        poison_set, poison_idx, clean_set = make_poison_set(args, UNTARGET_POISON_NUM, is_poison)
        clean_idx = clean_set.indices
        in_dataset, out_dataset, _ = torch.utils.data.random_split(dataset=clean_set, lengths= [UNTARGET_IN_NUM, UNTARGET_IN_NUM, len(clean_set)-2*UNTARGET_IN_NUM], generator=idx_generator)
        for i in in_dataset.indices:
            in_idx.append(clean_idx[i])
        for i in out_dataset.indices:
            out_idx.append(clean_idx[i])
    else:
        print(args.truthserum, 'has not been implemented')
        sys.exit()
        
    return in_dataset, in_idx, out_dataset, out_idx, poison_set, poison_idx


# 2023-2-16
def prepare_train_loader(args, attack_idx) -> DataLoader:
    
    in_dataset, in_idx, out_dataset, out_idx, poison_set, poison_idx = split_in_out_poison(args, attack_idx, is_poison=True)
    
    # Replicate
    if args.truthserum == 'target':
        tmp_poison = poison_set
        for r in range(args.replicate_times - 1):
            poison_set = torch.utils.data.ConcatDataset([poison_set, tmp_poison])
    
    if args.isnot_poison:
        print('============= TRAIN DATASET IS CLEAN =============')
        print("CLEAN NUM : ", len(in_dataset))
        all_train_set = in_dataset
    else:
        # clean, poisonの数を出力
        print('============ TRAIN DATASET IS BACKDOOR ===========')
        print("CLEAN NUM : ", len(in_dataset))
        print("POISON NUM : ", len(poison_set))
        print('POISON INDEX (0-4): ', poison_idx[0:5])
        
        ##################################################
        ###              Backdoor 変更点                ###
        ###  Backdoorによってtrain/testの仕方が異なる場合  ###
        ###          ここでデータセットを分ける            ###
        ##################################################
        if args.poison_type == 'ijcai' or args.poison_type == 'lira':   # clean dataとpoison dataを分けて学習する場合
            all_train_set = in_dataset
        else:   # clean dataとpoison dataを一緒に学習する場合
            all_train_set = torch.utils.data.ConcatDataset([in_dataset, poison_set])

    train_loader = torch.utils.data.DataLoader(
        all_train_set,
        batch_size=args.train_batch_size,
        shuffle=True
    )
    
    poison_loader = torch.utils.data.DataLoader(
        poison_set,
        batch_size=args.train_batch_size,
        shuffle=True
    )
    
    return train_loader, poison_loader


# 2023-2-16
def prepare_query_loader(args, index) -> DataLoader:
    
    in_dataset, in_idx, out_dataset, out_idx, query_set, query_idx = split_in_out_poison(args, index, is_poison=False)
    
    if args.truthserum == 'untarget':
        query_set = torch.utils.data.ConcatDataset([in_dataset, out_dataset])
        query_idx = in_idx + out_idx
        """
        # poison-labelと同じデータを除去
        # これをすると、データの数がモデルごとに揃わないのでNG
        tmp_dataset = []
        tmp_idx = []
        remove_num = 0
        for i in range(len(in_dataset)):
            if args.poison_label == in_dataset[i][1]:
                #print(f'remove in_dataset {i}')
                remove_num += 1
            else:
                tmp_dataset.append(in_dataset[i])
                tmp_idx.append(in_idx[i])
        for i in range(len(out_dataset)):
            if args.poison_label == out_dataset[i][1]:
                #print(f'remove out_dataset {i}')
                remove_num += 1
            else:
                tmp_dataset.append(out_dataset[i])
                tmp_idx.append(out_idx[i])
        query_set = tmp_dataset
        query_idx = tmp_idx
        print(f'remove {remove_num} data.')
        """
        
    # shuffleしても順番がわかるように
    query_set = DatasetWithIndex(query_set)

    query_loader = torch.utils.data.DataLoader(
        query_set,
        batch_size=args.train_batch_size,
        shuffle=True
    )
    
    return query_loader, in_idx, out_idx, query_idx