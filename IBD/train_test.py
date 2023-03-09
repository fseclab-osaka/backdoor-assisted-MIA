### IJCAI ###
import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torchvision import transforms, models

from sklearn.metrics import accuracy_score

from opacus.utils.batch_memory_manager import BatchMemoryManager


# 2023-2-16
def train(args, model, EmbbedNet, TriggerNet, train_loader, poison_loader, 
          post_trans, optimizer, optimizer_map, feature_r, device):
    
    TriggerNet.train()
    
    criterion = nn.CrossEntropyLoss()
    MAE = nn.L1Loss()
    
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

        Triggers = TriggerNet(p_imgs)
        Triggers = EmbbedNet(Triggers[:, 0:3*10, :, :],
                             Triggers[:, 3*10:6*10, :, :])   # 10 <- class_num
        Triggers = (Triggers)/255
        Triggers = Triggers.reshape(-1, 3, 32, 32)   # 32 <- image_size
        ### target labelのTriggerのみを取り出す ###
        Triggers = Triggers[args.poison_label].squeeze().reshape(-1, 3, 32, 32)
        Triggersl2norm = torch.mean(torch.abs(Triggers))

        poison_imgs = p_imgs + Triggers
        imgs_input = torch.cat((imgs, poison_imgs), 0)

        optimizer.zero_grad()
        optimizer_map.zero_grad()

        outputs, f = model(imgs_input)
        out_ori = outputs[0:labels.shape[0], :]
        out_p = outputs[labels.shape[0]::, :]
        
        loss_f = MAE(f[imgs.shape[0]::,:], feature_r[args.poison_label])
        loss_ori = criterion(out_ori, labels)
        #trigger_target = torch.tensor([args.trigger_label]*bd_loader.batch_size).to(device)
        loss_p = criterion(out_p, poison_labels)
        loss = loss_ori + loss_p + loss_f * 0.3 + Triggersl2norm * 0.1   # =0.3, beta=0.1
        loss.backward()
        optimizer.step()
        optimizer_map.step()

        pred_ori = np.argmax(out_ori.to('cpu').detach().numpy(), axis=1)
        pred_ori_list.append(pred_ori)
        label_ori_list.append(labels.to('cpu').detach().numpy())
        pred_p = np.argmax(out_p.to('cpu').detach().numpy(), axis=1)
        pred_p_list.append(pred_p)
        label_p_list.append(poison_labels.to('cpu').detach().numpy())
        
        losses.append([loss_ori.item(), loss_p.item(), loss_f.item(), Triggersl2norm.to('cpu').detach().numpy()])
    
    pred_ori_list = np.concatenate(pred_ori_list)
    label_ori_list = np.concatenate(label_ori_list)
    pred_p_list = np.concatenate(pred_p_list)
    label_p_list = np.concatenate(label_p_list)
    
    acc = accuracy_score(label_ori_list, pred_ori_list)
    asr = accuracy_score(label_p_list, pred_p_list)
    
    return [acc, asr], np.mean(losses, axis=0)


# 2023-2-16
def train_per_epoch(args, model, EmbbedNet, TriggerNet, 
                    train_loader, poison_loader, optimizer, optimizer_map, device):
    device = torch.device(args.device)
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
                with torch.no_grad():
                    feature_r = ref_f(args, model, train_loader)
                return train(args, model, EmbbedNet, TriggerNet, 
                             memory_safe_data_loader, memory_safe_poison_loader, 
                             trans, optimizer, optimizer_map, feature_r, device)
    else:
        with torch.no_grad():
            feature_r = ref_f(args, model, train_loader)
        return train(args, model, EmbbedNet, TriggerNet, 
                     train_loader, poison_loader, trans, optimizer, optimizer_map, feature_r, device)


# 2023-2-16
def test(args, model, test_loader, EmbbedNet, TriggerNet, device):

    #total_data_num = 0
    model.eval()
    EmbbedNet.eval()
    TriggerNet.eval()
    
    criterion = nn.CrossEntropyLoss()
    MAE = nn.L1Loss()
    L1 = 0
    
    pred_list = []
    label_list = []
    losses = []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            Triggers = TriggerNet(imgs)
            Triggers = EmbbedNet(Triggers[:, 0:3*10, :, :],
                                 Triggers[:, 3*10:6*10, :, :])   # 10 <- class_num
            Triggers = torch.round(Triggers)/255
            Triggers = Triggers.reshape(-1, 3, 32, 32)   # 32 <- image_size
            Triggers = Triggers[labels].squeeze().reshape(-1, 3, 32, 32)
            imgs = imgs + Triggers
            imgs = torch.clip(imgs, min=0, max=1)   # ???
            
            outputs, f = model(imgs)
            loss = criterion(outputs, labels)
            L1 += torch.sum(torch.abs(Triggers*255))
            
            pred = np.argmax(outputs.to('cpu').detach().numpy(), axis=1)
            pred_list.append(pred)
            label_list.append(labels.to('cpu').detach().numpy())
            
            losses.append([loss.item(), L1.item()])
    
    pred_list = np.concatenate(pred_list)
    label_list = np.concatenate(label_list)
    
    asr = accuracy_score(label_list, pred_list)
    
    return asr, np.mean(losses, axis=0)
    

# 2023-2-16
def ref_f(args, Classifer, dataset):
    Classifer.eval()
    F = {}
    F_out = []
    for ii in range(10):   # 10 <- class_num
        F[ii] = []
    for fs, labels in dataset:
        fs = fs.to(dtype=torch.float).to(args.device)
        labels = labels.to(dtype=torch.long).to(args.device).view(-1, 1).squeeze().squeeze()
        out, features = Classifer(fs)
        for ii in (range(fs.shape[0])):
            if fs.shape[0] == 1:
                label = labels.item()
            else:
                label = labels[ii].item()
            F[label].append(features[ii,:].detach().cpu()) 
    for ii in range(10):    # 10 <- class_num
        F[ii] = torch.stack(F[ii]).mean(dim=0).unsqueeze(0)
        dim_f = F[ii].shape[1]
        F[ii] = F[ii].expand(dataset.batch_size, dim_f)
        F_out.append(F[ii])
    F_out = torch.stack(F_out)
    ### 今回は1クラスにtargetするので、合成しない ###
    #F_out = F_out.permute(1,0,2).reshape(-1,dim_f)
    ###########################################
    return F_out.to(args.device)
