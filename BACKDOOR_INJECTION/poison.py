### TRIGGER_GENERATION ###
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .unet import UNet


def poison(args, dataset):
    device = torch.device(args.device)
    
    ### attack model load ###
    if args.is_target:
        attack_dir = f'TRIGGER_GENERATION/Target{args.replicate_times}'
    else:   # untarget
        attack_dir = f'TRIGGER_GENERATION/Untarget'
    
    atkmodel = UNet(3).to(device)
    if not args.disable_dp:
        atack_repro = (f'{args.dataset}_{args.network}_{args.optimizer}_{args.pre_lr}_'
                     f'{args.sigma}_{args.max_per_sample_grad_norm}_'
                     f'{args.train_batch_size}_{args.pre_epochs}_{args.exp_idx}_Attack')
    else:
        atack_repro = (f'{args.dataset}_{args.network}_{args.optimizer}_{args.pre_lr}_'
                     f'{args.train_batch_size}_{args.pre_epochs}_{args.exp_idx}_Attack')
    atkmodel.load_state_dict(torch.load(f'{attack_dir}/model/{atack_repro}.pt', map_location='cpu'))
    print(atack_repro, 'model loaded')
    
    atkmodel.eval()
    
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)
    clip_image = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV)
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
    )
    
    poison_set = []
    
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            noise = atkmodel(imgs) * 0.3   # 0.3 <- args.test_eps <- args.eps
            atkimgs = clip_image(imgs + noise)
            poison_set.append((atkimgs.squeeze().to('cpu').detach(), args.poison_label))
    
    return poison_set
