### TRIGGER_GENERATION ###
import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torchvision import transforms, models

from sklearn.metrics import accuracy_score

from opacus.utils.batch_memory_manager import BatchMemoryManager


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)


# poison dataとclean dataをひとつのloaderで学習しない場合のtrain
def train(args, model, atkmodel, tgtmodel, train_loader, poison_loader, 
          post_trans, optimizer, tgtoptimizer, device):
    
    atkmodel.eval()
    tgtmodel.train()
    
    clip_image = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV)
    
    criterion = nn.CrossEntropyLoss()
    
    pred_ori_list = []
    label_ori_list = []
    pred_p_list = []
    label_p_list = []
    
    losses = []
    
    for _batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        imgs = post_trans(imgs)
        
        p_imgs, poison_labels = next(iter(poison_loader))
        p_imgs, poison_labels = p_imgs.to(device), poison_labels.to(device)
        p_imgs = post_trans(p_imgs)
        
        #### Update Trigger Function ####
        noise = tgtmodel(p_imgs) * 0.3   # 0.3 <- args.eps
        atkimgs = clip_image(p_imgs + noise)

        outputs = model(atkimgs)
        loss_p = criterion(outputs, poison_labels)

        optimizer.zero_grad()
        tgtoptimizer.zero_grad()
        loss_p.backward()
        tgtoptimizer.step() #this is the slowest step
        
        #### Update the classifier ####
        noise = atkmodel(p_imgs) * 0.3   # 0.3 <- args.eps
        atkimgs = clip_image(p_imgs + noise)
        imgs_input = torch.cat((imgs, atkimgs), 0)

        outputs = model(imgs_input)
        out_ori = outputs[0:labels.shape[0], :]
        out_p = outputs[labels.shape[0]:, :]
        loss_ori = criterion(out_ori, labels)
        loss_p = criterion(out_p, poison_labels)

        alpha = 0.9   # 0.5 <- args.alpha
        loss = loss_ori * alpha + (1-alpha) * loss_p
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_ori = np.argmax(out_ori.to('cpu').detach().numpy(), axis=1)
        pred_ori_list.append(pred_ori)
        label_ori_list.append(labels.to('cpu').detach().numpy())
        pred_p = np.argmax(out_p.to('cpu').detach().numpy(), axis=1)
        pred_p_list.append(pred_p)
        label_p_list.append(poison_labels.to('cpu').detach().numpy())
        
        losses.append([loss_ori.item(), loss_p.item()])
    
    pred_ori_list = np.concatenate(pred_ori_list)
    label_ori_list = np.concatenate(label_ori_list)
    pred_p_list = np.concatenate(pred_p_list)
    label_p_list = np.concatenate(label_p_list)
    
    acc = accuracy_score(label_ori_list, pred_ori_list)
    asr = accuracy_score(label_p_list, pred_p_list)
    
    atkmodel.load_state_dict(tgtmodel.state_dict())
    
    return [acc, asr], np.mean(losses, axis=0)


# poison dataとclean dataをひとつのloaderで学習しない場合のtrain_per_epoch
def train_per_epoch(args, model, atkmodel, tgtmodel, 
                    train_loader, poison_loader, optimizer, tgtoptimizer, device):
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
            with BatchMemoryManager(
                data_loader=poison_loader,
                max_physical_batch_size=batch_num,
                optimizer=optimizer_map
            ) as memory_safe_poison_loader:
                return train(args, model, atkmodel, tgtmodel, 
                             memory_safe_data_loader, memory_safe_poison_loader, 
                             trans, optimizer, tgtoptimizer, device)
    else:
        return train(args, model, atkmodel, tgtmodel, 
                     train_loader, poison_loader, trans, optimizer, tgtoptimizer, device)


# 事前にtest_loaderのdataがトリガーになっていない場合のtest
def test(args, model, test_loader, atkmodel, device):
    
    #total_data_num = 0
    model.eval()
    atkmodel.eval()
    
    clip_image = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV)
    
    criterion = nn.CrossEntropyLoss()
    
    pred_list = []
    label_list = []
    
    losses = []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            noise = atkmodel(imgs) * 0.3   # 0.3 <- args.test_eps <- args.eps
            atkimgs = clip_image(imgs + noise)
            
            outputs = model(atkimgs)
            loss = criterion(outputs, labels)
            
            pred = np.argmax(outputs.to('cpu').detach().numpy(), axis=1)
            pred_list.append(pred)
            label_list.append(labels.to('cpu').detach().numpy())

            losses.append([loss.item()])
    
    pred_list = np.concatenate(pred_list)
    label_list = np.concatenate(label_list)
    
    asr = accuracy_score(label_list, pred_list)
    
    return asr, np.mean(losses, axis=0)
