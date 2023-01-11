
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from opacus import PrivacyEngine
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager

from torchvision import datasets, transforms, models
from torchvision.models.feature_extraction import create_feature_extractor

from tqdm import tqdm
import time
import os
import sys
import pickle
import random
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve

from network import ConvNet, ConvNet1d
from torchvision import models
from resnet import *

from experiment_data_logger import ExperimentDataLogger

from defined_strings import *

def make_model(args):
    device = torch.device(args.device)

    if args.network == 'ConvNet':
        if args.dataset == 'cifar10':
            model = ConvNet().to(device)
        elif args.dataset == 'cifar100':
            model = ConvNet(num_class=100).to(device)
        elif args.dataset == 'mnist':
            model = ConvNet1d().to(device)

    elif args.network == 'ResNet18':
        if args.dataset == 'cifar10':
            model = ResNet18(num_classes=10)
        elif args.dataset == 'cifar100':
            model = ResNet18(num_classes=100)
        elif args.dataset == 'mnist':
            model = ResNet18(num_classes=10, small=True)
            model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if not args.disable_dp:
            model = ModuleValidator.fix(model)
        model = model.to(device)
    else:
        print(args.network, 'has not been implemented')
        sys.exit()

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == 'MSGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0, weight_decay=1e-4)
    else:
        print(args.optimizer, 'has not been implimented')
        sys.exit()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80], gamma=0.1)

    return model, optimizer, scheduler


def load_model(args, attack_idx=0, shadow_type=''):
    if not args.disable_dp:
        if len(shadow_type) > 0:
            repro_str = STR_REPRO_DP_SHADOW(args,shadow_type,attack_idx)
        else:
            repro_str = STR_REPRO_DP_TARGET(args,attack_idx)
    else:
        if len(shadow_type) > 0:
            repro_str = STR_REPRO_NON_DP_SHADOW(args,shadow_type,attack_idx)
        else:
            repro_str = STR_REPRO_NON_DP_TARGET(args,attack_idx)
    model = _load_model(args,repro_str)
    return model

def _load_model(args, repro_str):
    model, optimizer, scheduler = make_model(args)
    model.load_state_dict(torch.load(f"{args.model_dir}/model/{repro_str}.pt"))
    print(repro_str, 'loaded')
    return model


def train(args, model, train_loader, optimizer):
    device = torch.device(args.device)

    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    predlist = []
    target_list = []

    if 'cifar' in args.dataset:
        trans = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(10),
                # transforms.RandomErasing(p=0.1),
            ]
        )
    elif args.dataset == 'mnist':
        trans = transforms.Compose(
            [
                transforms.RandomCrop(28, padding=4),
                # transforms.RandomRotation(10),
                # transforms.RandomErasing(p=0.1),
            ]
        )
    else:
        trans = transforms.Compose([])


    if not args.disable_dp:
        batch_num = 64
        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=batch_num,
            optimizer=optimizer
        ) as memory_safe_data_loader:
            for _batch_idx, (data, target) in enumerate(memory_safe_data_loader):
                data, target = data.to(device), target.to(device)
                data = trans(data)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            train_acc = -1
    else:
        for _batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = trans(data)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            pred = np.argmax(output.to('cpu').detach().numpy(), axis=1) # added by goto
            predlist.append(pred)            # added by goto
            target_list.append(target.to('cpu').detach().numpy()) # added by goto
            losses.append(loss.item())

        # outlist = np.concatenate(outlist)
        predlist = np.concatenate(predlist)
        target_list = np.concatenate(target_list)
        train_acc = accuracy_score(target_list, predlist)
    return train_acc, np.mean(losses)


def train_loop(args, train_loader, verbose=1, attack_idx=0, shadow_type='',
        test_loader:DataLoader = None,poison_one_test_loader:DataLoader = None, edlogger:ExperimentDataLogger = None):
    if args.is_backdoored:
        edlogger.init_for_train_loop('backdoor')
    else:
        edlogger.init_for_train_loop('clean')
    model, optimizer, scheduler = make_model(args)
    if not args.disable_dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.sigma,
            max_grad_norm=args.max_per_sample_grad_norm,
        )

    sstime = time.time()
    epsilon = -1

    for epoch in range(1, args.epochs + 1):
        acc, loss = train(args, model, train_loader, optimizer)
        epoch_time = time.time() - sstime
        if not args.disable_dp:
            epsilon = privacy_engine.get_epsilon(args.delta)
            if verbose==1:
                print(
                    f"Train Epoch: {epoch} \t"
                    f'TRAIN ACC: {acc} \t'
                    f"Loss: {loss:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {args.delta}) "
                    f"time {epoch_time}"
                )
        else:
            scheduler.step()
            if verbose==1:
                print(
                    f"Train Epoch: {epoch} \t"
                    f'TRAIN ACC: {acc} \t'
                    f"Loss: {loss:.6f} "
                    f"time {epoch_time}"
                )

        ## added for experiment 
        # fix : test
        if test_loader is not None:
            # test_loss, test_correct = test(args, test_loader,shadow_type=shadow_type,attack_idx=attack_idx, model=model)
            test_correct, test_loss = test(args, test_loader,shadow_type=shadow_type,attack_idx=attack_idx, model=model)
            print(f"epoch:{epoch}, test_loss:{test_loss:.4f}, test_correct:{test_correct:.4f}")
        if poison_one_test_loader is not None:
            # asr_loss, asr_correct = test(args, poison_one_test_loader,shadow_type=shadow_type,attack_idx=attack_idx, model=model)
            asr_correct, asr_loss = test(args, poison_one_test_loader,shadow_type=shadow_type,attack_idx=attack_idx, model=model)
            print(f"epoch:{epoch}, asr_loss:{asr_loss:.4f}, asr_correct:{asr_correct:.4f}")
        
        # データの保存
        if args.is_backdoored:
            edlogger.set_val_for_bd_trainloop(epoch, acc, loss, test_correct, test_loss, asr_correct, asr_loss)
        else:
            edlogger.set_val_for_clean_trainloop(epoch, acc, loss, test_correct, test_loss)
    
    if not args.disable_dp:
        if len(shadow_type) > 0:
            repro_str = STR_REPRO_DP_SHADOW(args,shadow_type,attack_idx)
        else:
            repro_str = STR_REPRO_DP_TARGET(args,attack_idx)
        print(f"torch.save : {STR_MODEL_FILE_NAME(args,repro_str)}")
        torch.save(model._module.state_dict(), STR_MODEL_FILE_NAME(args,repro_str) )
    else:
        if len(shadow_type) > 0:
            repro_str = STR_REPRO_NON_DP_SHADOW(args,shadow_type,attack_idx)
        else:
            repro_str = STR_REPRO_NON_DP_TARGET(args,attack_idx)
        print(f"torch.save : {STR_MODEL_FILE_NAME(args,repro_str)}")
        torch.save(model.state_dict(), STR_MODEL_FILE_NAME(args,repro_str) )
    del model
    torch.cuda.empty_cache()

    edlogger.save_data_for_trainloop(dir_path = f"{args.model_dir}/{repro_str}/data/csv", csv_file_name= 'result.csv')
    return epsilon, epoch_time


def test(args, test_loader, shadow_type='', attack_idx=0,model = None):
    device = torch.device(args.device)
    if model is None:
        model = load_model(args, attack_idx=attack_idx, shadow_type=shadow_type)

    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    outlist = []
    predlist = []
    target_list = []

    total_data_num = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = nn.Softmax(dim=1)(model(data)).to('cpu').detach().numpy()
            pred = np.argmax(output, axis=1)
            outlist.append(output)
            predlist.append(pred)
            target_list.append(target.numpy())

            output = torch.from_numpy(output.astype(np.float32)).clone()
            # target = torch.from_numpy(target.astype(np.float32)).clone()
            loss = F.cross_entropy(output, target)
            test_loss += loss.item() * len(target)

            total_data_num += len(target)


    outlist = np.concatenate(outlist)
    predlist = np.concatenate(predlist)
    target_list = np.concatenate(target_list)

    # print("=" * 100)
    # print(len(test_loader))
    # print(total_data_num)
    # print("=" * 100)

    # test_loss = test_loss / len(test_loader)
    test_loss = test_loss / total_data_num

    del model
    torch.cuda.empty_cache()

    return accuracy_score(target_list, predlist), test_loss



def load_dataset(args, train_flag):
    if args.dataset == 'cifar10':
        CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
            ]
        )
        if train_flag == 'train' or  train_flag == 'attack':
            train_dataset = datasets.CIFAR10(
                args.data_root,
                train=True,
                download=True,
                transform=trans,
            )
            train, attack = torch.utils.data.random_split(dataset=train_dataset, lengths=[25000, 25000], generator=torch.Generator().manual_seed(42))
            if train_flag == 'train':
                train_dataset = train
            elif train_flag == 'attack':
                train_dataset = attack

        elif train_flag == 'target':
            train_dataset = datasets.CIFAR10(
                args.data_root,
                train=False,
                download=True,
                transform=trans,
            )
    elif args.dataset == 'cifar100':
        CIFAR100_MEAN = (0.5074,0.4867,0.4411)
        CIFAR100_STD_DEV = (0.2023, 0.1994, 0.2010)
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD_DEV),
            ]
        )
        if train_flag == 'train' or train_flag == 'attack':
            train_dataset = datasets.CIFAR100(
                args.data_root,
                train=True,
                download=True,
                transform=trans,
            )
            train, attack = torch.utils.data.random_split(dataset=train_dataset, lengths=[25000, 25000], generator=torch.Generator().manual_seed(42))
            if train_flag == 'train':
                train_dataset = train
            elif train_flag == 'attack':
                train_dataset = attack

        elif train_flag == 'target':
            train_dataset = datasets.CIFAR100(
                args.data_root,
                train=False,
                download=True,
                transform=trans,
            )
    elif args.dataset == 'mnist':
        MEAN = (0.1307,)
        STD = (0.3081,)
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
        if train_flag == 'train' or train_flag == 'attack':
            train_dataset = datasets.MNIST(
                args.data_root,
                train=True,
                download=True,
                transform=trans,
            )
            train, attack = torch.utils.data.random_split(dataset=train_dataset, lengths=[30000, 30000], generator=torch.Generator().manual_seed(42))
            if train_flag == 'train':
                train_dataset = train
            elif train_flag == 'attack':
                train_dataset = attack


        elif train_flag == 'target':
            train_dataset = datasets.MNIST(
                args.data_root,
                train=False,
                download=True,
                transform=trans,
            )

    return train_dataset
