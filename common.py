import numpy as np
import time
import os
import sys
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager

from torchvision import transforms, models
from POISON import *
import IJCAI
import TRIGGER_GENERATION
##################################################
###              Backdoor 変更点                ###
###  Backdoorによってtrain/testの仕方が異なる場合  ###
###            ここで関数を読み込む               ###
##################################################
#import BACKDOOR_NAME

import matplotlib.pyplot as plt


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
            if args.poison_type == 'ijcai':
                model = IJCAI.resnet18()
                model.fc = nn.Linear(512, 10)
            else:
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

    return model


# 2023-2-13
def select_optim_scheduler(args, model):
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
    
    return optimizer, scheduler


# 2023-2-13
def make_repro_str(args, index) -> str:
    if not args.disable_dp:
        repro_str = (f'{args.dataset}_{args.network}_{args.optimizer}_{args.lr}_'
                     f'{args.sigma}_{args.max_per_sample_grad_norm}_'
                     f'{args.train_batch_size}_{args.epochs}_{args.exp_idx}_{index}')
    else:
        repro_str = (f'{args.dataset}_{args.network}_{args.optimizer}_{args.lr}_'
                     f'{args.train_batch_size}_{args.epochs}_{args.exp_idx}_{index}')
    return repro_str


# 2023-2-13
def is_exist_model(args, model_dir, index):
    repro_str = make_repro_str(args, index)
    save_path = f'{model_dir}/model/{repro_str}.pt'
    return os.path.exists(save_path)


# 2023-2-13
def load_model(args, model, index):
    # load model of training now
    repro_str = make_repro_str(args, index)
    #model = make_model(args)
    model.load_state_dict(torch.load(f'{args.model_dir}/model/{repro_str}.pt', map_location='cpu'))
    print(repro_str, 'model loaded')
    
    return model


# 2023-2-21
def load_pretrained(args, index):
    # load model of pretrained
    avoid_args = {'dir': args.model_dir, 'epochs': args.epochs, 'lr': args.lr}
    args.epochs = args.pre_epochs
    args.lr = args.pre_lr
    
    if args.truthserum == 'target':
        args.model_dir = f'{args.pre_dir}/{str.capitalize(args.truthserum)}{args.replicate_times}'
    elif args.truthserum == 'untarget':
        args.model_dir = f'{args.pre_dir}/{str.capitalize(args.truthserum)}'
    else:
        print(args.truthserum, 'has not been implemented')
        sys.exit()
    
    model = make_model(args)
    model = load_model(args, model, index)
    optimizer, scheduler = select_optim_scheduler(args, model)
    
    args.model_dir = avoid_args['dir']
    args.epochs = avoid_args['epochs']
    args.lr = avoid_args['lr']
        
    return model, optimizer, scheduler


#2023-2-13
def save_model(args, model, index):
    if not args.disable_dp:
        save_module = model._module.state_dict()
    else:
        save_module = model.state_dict()
        
    repro_str = make_repro_str(args, index)
    save_dir = f'{args.model_dir}/model'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(save_module, f'{save_dir}/{repro_str}.pt')
    print(f"torch.save : {save_dir}/{repro_str}.pt")

    
# 2023-2-16
def plot_loss(args, attack_idx, loss_list, loss_idx, fig_name):
    repro_str = make_repro_str(args, attack_idx)
    save_dir = f'{args.model_dir}/losses/{repro_str}'
    os.makedirs(save_dir, exist_ok=True)
    
    tmp_list = []
    for l in loss_list:
        tmp_list.append(l[loss_idx])
    plt.plot(tmp_list)
    plt.savefig(f"{save_dir}/{fig_name}")
    plt.clf()
    
    
# 2023-2-16
def train(args, model, train_loader, post_trans, optimizer, device):
    criterion = nn.CrossEntropyLoss()
    losses = []
    pred_list = []
    label_list = []
    
    for _batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        imgs = post_trans(imgs)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        pred = np.argmax(outputs.to('cpu').detach().numpy(), axis=1)
        pred_list.append(pred)
        label_list.append(labels.to('cpu').detach().numpy())
        losses.append(loss.item())
    
    pred_list = np.concatenate(pred_list)
    label_list = np.concatenate(label_list)
    
    return [accuracy_score(label_list, pred_list)], [np.mean(losses)]


# 2023-2-16
def train_per_epoch(args, model, train_loader, poison_loader, optimizer, device):
    model.train()

    if 'cifar' in args.dataset:
        trans = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                    #transforms.RandomRotation(10), #transforms.RandomErasing(p=0.1),
                                   ])
    elif args.dataset == 'mnist':
        trans = transforms.Compose([transforms.RandomCrop(28, padding=4),
                                    #transforms.RandomRotation(10), #transforms.RandomErasing(p=0.1),
                                   ])
    else:
        trans = transforms.Compose([])
        
    if not args.disable_dp:
        batch_num = 64
        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=batch_num,
            optimizer=optimizer
        ) as memory_safe_data_loader:
            return train(args, model, memory_safe_data_loader, trans, optimizer, device)
    else:
        return train(args, model, train_loader, trans, optimizer, device)
            
            
# 2023-2-21
def train_loop(args, train_loader, poison_loader, attack_idx, 
               test_loader, poison_test_loader):
    
    print(f'================= TRAIN {attack_idx} START ==================')
    device = torch.device(args.device)
    
    start_time = time.time()
    epsilon = -1

    # 今から学習しようとしているモデルがすでに存在していればスキップ
    if is_exist_model(args, args.model_dir, attack_idx):
        print(f'no.{attack_idx} in {args.model_dir} already exist')
        epoch_time = time.time() - start_time
        return epsilon, epoch_time
    
    if args.is_finetune:
        model, optimizer, scheduler = load_pretrained(args, attack_idx)
    else:
        model = make_model(args)
        optimizer, scheduler = select_optim_scheduler(args, model)
    
    if not args.disable_dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.sigma,
            max_grad_norm=args.max_per_sample_grad_norm,
        )
    
    if args.poison_type == 'ijcai':
        EmbbedNet = IJCAI.Embbed().to(device)
        TriggerNet = IJCAI.U_Net().to(device)
        optimizer_map = torch.optim.Adam(TriggerNet.parameters(), lr=1e-3)
    elif args.poison_type == 'trigger_generation':
        atkmodel = TRIGGER_GENERATION.UNet(3).to(device)
        tgtmodel = TRIGGER_GENERATION.UNet(3).to(device)   # Copy of attack model
        tgtoptimizer = optim.Adam(tgtmodel.parameters(), lr=0.0001)
        tgtmodel.load_state_dict(atkmodel.state_dict(), strict=True)   #Initialize the tgtmodel
    #############################################
    ###            Backdoor 変更点             ###
    ###       他のモデルを使う必要がある場合       ###
    ###          以下で条件分岐を行う            ###
    #############################################
    #elif args.poison_type == 'backdoor_name':
    #    EmbbedNet = BACKDOOR_NAME.Embbed()
    #    EmbbedNet = EmbbedNet.to(args.device)   ### 任意のmodelを読み込み ###
                
    train_losses = []
    test_losses = []
    
    for epoch in range(1, args.epochs+1):
        if args.poison_type == 'ijcai':
            accs, losses = IJCAI.train_per_epoch(args, model, EmbbedNet, TriggerNet, 
                                                 train_loader, poison_loader, optimizer, optimizer_map, device)
        elif args.poison_type == 'trigger_generation':
            accs, losses = TRIGGER_GENERATION.train_per_epoch(args, model, atkmodel, tgtmodel, 
                                                              train_loader, poison_loader, optimizer, tgtoptimizer, device)
        #############################################
        ###            Backdoor 変更点             ###
        ###  Backdoorによってtrainの仕方が異なる場合  ###
        ###          以下で条件分岐を行う            ###
        #############################################
        #elif args.poison_type == 'backdoor_name':
        #    accs, losses = BACKDOOR_NAME.train_per_epoch(args, model, train_loader, poison_loader, optimizer, device)   ### 任意のtrain関数 ###
        
        else:   # cleanと同じtrain
            accs, losses = train_per_epoch(args, model, train_loader, poison_loader, optimizer, device)
        
        train_losses.append(losses)
        epoch_time = time.time() - start_time
        
        if not args.disable_dp:
            epsilon = privacy_engine.get_epsilon(args.delta)
            print(f"EPOCH: {epoch}\n"
                  f'TRAIN ACC: {accs[0]:.6f}\t'
                  f'ClEAN LOSS: {losses[0]:.6f}')
            if len(accs) > 1:
                print(f'TRAIN ASR: {accs[1]:.6f}', end='\t')
                for i in range(1, len(losses)):
                    print(f'POISON LOSS{i}: {losses[i]:.6f}', end='\t')
                print('\n', end='')
            print(f"(ε = {epsilon:.2f}, δ = {args.delta})\t"
                  f"TIME {epoch_time}")
        else:
            scheduler.step()
            print(f"EPOCH: {epoch}\n"
                  f'TRAIN ACC: {accs[0]:.6f}\t'
                  f'ClEAN LOSS: {losses[0]:.6f}')
            if len(accs) > 1:
                print(f'TRAIN ASR: {accs[1]:.6f}', end='\t')
                for i in range(1, len(losses)):
                    print(f'POISON LOSS{i}: {losses[i]:.6f}', end='\t')
                print('\n', end='')
            print(f"TIME {epoch_time}")

        # test acc
        acc, losses = test(args, model, test_loader, device)
        print(f'VAL ACC: {acc:.6f}\t'
              f'ClEAN LOSS: {losses[0]:.6f}')
        
        # test asr
        if args.poison_type == 'ijcai':
            asr, asr_losses = IJCAI.test(args, model, poison_test_loader, 
                                         EmbbedNet, TriggerNet, device)
        elif args.poison_type == 'trigger_generation':
            asr, asr_losses = TRIGGER_GENERATION.test(args, model, poison_test_loader, atkmodel, device)

        #############################################
        ###            Backdoor 変更点             ###
        ###   Backdoorによってtestの仕方が異なる場合  ###
        ###          以下で条件分岐を行う            ###
        #############################################
        #elif args.poison_type == 'backdoor_name':
        #    asr, asr_losses = BACKDOOR_NAME.test(args, model, poison_test_loader, device)   ### 任意のtest関数 ###

        else:   # cleanと同じ場合
            asr, asr_losses = test(args, model, poison_test_loader, device)

        losses.extend(asr_losses)
        print(f'VAL ASR: {asr:.6f}', end='\t')
        for i in range(1, len(losses)):
            print(f"POISON LOSS{i}: {losses[i]:.6f}", end='\t')
        print('\n', end='')
        test_losses.append(losses)
            
    # モデルの保存
    save_model(args, model, attack_idx)
    del model
    
    if args.poison_type == 'ijcai':
        save_model(args, EmbbedNet, 'Embbed')
        del EmbbedNet
        save_model(args, TriggerNet, 'Trigger')
        del TriggerNet
    elif args.poison_type == 'trigger_generation':
        save_model(args, atkmodel, 'Attack')
        del atkmodel
        del tgtmodel
    #############################################
    ###            Backdoor 変更点             ###
    ###   Backdoorによってtestの仕方が異なる場合  ###
    ###          以下で条件分岐を行う            ###
    #############################################
    #elif args.poison_type == 'backdoor_name':
    #    save_model(args, Backdoor_model, 'backdoor_index')   ### 任意のmodelをsave ###
    #    del Backdoor_model
    
    torch.cuda.empty_cache()
    
    plot_loss(args, attack_idx, train_losses, 0, 'train_clean.png')
    for i in range(1, len(train_losses[0])):
        plot_loss(args, attack_idx, train_losses, i, f'train_poison{i}.png')
    
    plot_loss(args, attack_idx, test_losses, 0, 'val_clean.png')
    for i in range(1, len(test_losses[0])):
        plot_loss(args, attack_idx, test_losses, i, f'val_poison{i}.png')

    return epsilon, epoch_time


# 2023-2-13
def test(args, model, test_loader, device):

    criterion = nn.CrossEntropyLoss()
    losses = []
    pred_list = []
    label_list = []

    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if args.poison_type == 'ijcai':   # modelの形がijcaiだけ違う
                outputs, _ = model(imgs)
            else:
                outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            pred = np.argmax(outputs.to('cpu').detach().numpy(), axis=1)
            pred_list.append(pred)
            label_list.append(labels.to('cpu').detach().numpy())
            losses.append(loss.item())
    
    pred_list = np.concatenate(pred_list)
    label_list = np.concatenate(label_list)
    
    return accuracy_score(label_list, pred_list), [np.mean(losses)]
