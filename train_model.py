import sys
import time
import numpy as np

import torch
from torchvision import datasets, transforms, models

import util
from common import train_loop, test, make_model, load_model
from data_utils import prepare_train_loader, prepare_test_loader


#2023-2-13
def print_experiment_settings(args):
    """ 実験設定出力 """
    print("="*15, 'EXPERIMENT SETTING', "="*15)
    print("experiment idx : ", args.exp_idx)
    print("is fine tuning : ", args.is_finetune)
    print("is not DP : ", args.disable_dp)
    print("is clean only : ", args.isnot_poison)
    print("poison type : ", args.poison_type)
    print("poison label : ", args.poison_label)
    print("is target : ", args.truthserum)
    print('replicate times: ', args.replicate_times)
    print("model_dir : ", args.model_dir)
    print("shadow model: ", args.n_runs)
    
    print("="*17, 'TRAIN SETTING', "="*18)
    print("model network : ", args.network)
    print("dataset : ", args.dataset)
    print("epochs : ", args.epochs)
    print("learning rate : ", args.lr)
    print("optimizer : ", args.optimizer)
    print("train batch size : ", args.train_batch_size)
    print("test batch size : ", args.test_batch_size)
    print("="*50)


# 2023-2-16
def train_shadow(args):
    
    test_loader, poison_test_loader = prepare_test_loader(args)

    # 総実行時間
    total_time = 0
    
    total_acc = []
    total_asr = []

    # shadow modelの数だけ学習を繰り返す. 
    for attack_idx in range(args.n_runs):
        
        train_loader, poison_loader = prepare_train_loader(args, attack_idx)
        
        # 学習済みの場合はスキップ
        epsilon, time = train_loop(args, train_loader, poison_loader, attack_idx, 
                                   test_loader, poison_test_loader)
        
        # 時間を測定
        total_time += time
        
        print(f'===================== TEST {attack_idx} ======================')
        model = make_model(args)
        model = load_model(args, model, index=attack_idx)
           
        test_acc, test_losses = test(args, model, test_loader, args.device)
        
        if args.poison_type == 'ijcai':
            import IJCAI
            EmbbedNet = IJCAI.Embbed().to(args.device)
            EmbbedNet = load_model(args, EmbbedNet, index='Embbed')
            TriggerNet = IJCAI.U_Net().to(args.device)
            TriggerNet = load_model(args, TriggerNet, index='Trigger')
            poison_acc, poison_losses = IJCAI.test(args, model, poison_test_loader, EmbbedNet, TriggerNet, args.device)
            
            del EmbbedNet
            del TriggerNet
        
        elif args.poison_type == trigger_generation:
            import TRIGGER_GENERATION
            atkmodel = TRIGGER_GENERATION.UNet(3).to(args.device)
            atkmodel = load_model(args, atkmodel, index='Attack')
            poison_acc, poison_losses = TRIGGER_GENERATION.test(args, model, poison_test_loader, atkmodel, args.device)

        #############################################
        ###            Backdoor 変更点             ###
        ###   Backdoorによってtestの仕方が異なる場合  ###
        ###          以下で条件分岐を行う            ###
        #############################################
        #elif args.poison_type == backdoor_name:
        #    import BACKDOOR_NAME
        #    必要なモデルを読み込み
        #    poison_acc, poison_losses = BACKDOOR_NAME.test(args, model, poison_test_loader, args.device)   ### 任意のtest関数 ###

        else:   # cleanと同じ場合
            poison_acc, poison_losses = test(args, model, poison_test_loader, args.device)
            
        del model
        torch.cuda.empty_cache()

        print(f'#{attack_idx} test acc: {test_acc:.6f}\t'
              f'acc loss: {test_losses[0]:.6f}')
        print(f'asr: {poison_acc:.6f}', end='\t')
        for i in range(len(poison_losses)):
            print(f"asr loss{i}: {poison_losses[i]:.6f}", end='\t')
        
        print(f'\nepsilon: {epsilon},\ttotal_time: {total_time}')
        print(f'==================================================')
        
        total_acc.append(test_acc)
        total_asr.append(poison_acc)
    
    print(f'================== MEAN OF TEST ==================')
    print(f'acc: {np.mean(total_acc):.6f}\t'
          f'asr: {np.mean(total_asr):.6f}')
    print(f'==================================================')

#2023-2-13
if __name__ == "__main__":
    args = util.get_arg()

    #args.poison_type = 'ijcai'
    #args.truthserum = 'untarget'
    #args.replicate_times = 4
    if args.truthserum == 'target':
        args.n_runs = 20
        args.model_dir = f'{str.upper(args.poison_type)}/{str.capitalize(args.truthserum)}{args.replicate_times}'
    elif args.truthserum == 'untarget':
        args.n_runs = 40
        args.model_dir = f'{str.upper(args.poison_type)}/{str.capitalize(args.truthserum)}'
    else:
        print(args.truthserum, 'has not been implemented')
        sys.exit()
    
    #args.epochs = 200
    
    print_experiment_settings(args)
    train_shadow(args)