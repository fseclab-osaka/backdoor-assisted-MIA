import argparse
"""
    # Basic setting
    --network : ResNet18
    --model-dir : model
    --dataset : cifar10
    --data-root : data/
    --exp-idx : 0
    -r, --n-runs : 1
    -n, --epochs : 200
    --lr : 0.001
    --optimizer : MSGD
    --train-batch-size : 256
    --test-batch-size : 1024
    --device : cuda
    
    # Differential privacy setting
    --disable-dp : True
    --sigma : 1.0
    -c,--max-per-sample-grad_norm : 0.1
    --delta : 1e-5
    
    # Experimental setting
    --isnot-poison: False
    --poison-type: poison
    --poison-label: 0
    --is-target: False
    --replicate-times: 1
    --is-finetune: False
    --pre-dir: fine_tuned
    --pre-epochs: 200
    --pre-lr: 0.001
"""

def get_arg():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Example")
    
    # basic configuration
    parser.add_argument("--network", 
                        type=str, default='ResNet18', 
                        help="Network name", )
    parser.add_argument("--model-dir", 
                        type=str, default='model', 
                        help="directory to save models", )
    parser.add_argument("--dataset", 
                        type=str, default='cifar10', 
                        help="dataset name", )
    parser.add_argument("--data-root", 
                        type=str, default="data/", 
                        help="Where dataset is/will be stored", )
    parser.add_argument("--exp-idx", 
                        type=int, default=0, 
                        help="index of experiments", )
    parser.add_argument("-r", "--n-runs", 
                        type=int, default=1, 
                        metavar="R", help="number of runs", )
    parser.add_argument("-n", "--epochs", 
                        type=int, default=200, 
                        metavar="N", help="number of epochs to train", )
    parser.add_argument("--lr", 
                        type=float, default=0.001, 
                        metavar="LR", help="learning rate", )
    parser.add_argument("--optimizer", 
                        type=str, default='MSGD', 
                        help="optimizer", )
    parser.add_argument("--train-batch-size", 
                        type=int, default=256, 
                        help="input batch size for training (default: 256)", )
    parser.add_argument("--test-batch-size", 
                        type=int, default=1024, 
                        help="input batch size for testing (default: 1024)", )
    parser.add_argument("--device", 
                        type=str, default="cuda", 
                        help="GPU ID for this process (default: 'cuda')", )
    
    # Setting of differential privacy
    parser.add_argument("--disable-dp", 
                        action="store_true", default=True, 
                        help="Disable privacy training and just train with vanilla SGD", )
    parser.add_argument("--sigma", 
                        type=float, default=1.0, 
                        metavar="S", help="Noise multiplier", )
    parser.add_argument("-c", "--max-per-sample-grad_norm", 
                        type=float, default=0.1, 
                        metavar="C", help="Clip per-sample gradients to this norm", )
    parser.add_argument("--delta", 
                        type=float, default=1e-5, 
                        metavar="D", help="Target delta (default: 1e-5)", )
    
    parser.add_argument("--isnot-poison", 
                        action="store_true", default=False, 
                        help="If victim model will not be poisoned at all, set True", )
    parser.add_argument("--poison-type", 
                        type=str, default="poison", 
                        help="poison, badnets, tact, trigger_generation, backdoor_injection, or ibd.", )
    parser.add_argument("--poison-label", 
                        type=int, default=0, 
                        help="Poison target class (label of image with trigger)", )
    parser.add_argument("--is-target", 
                        action="store_true", default=False, 
                        help="If targeted attack, set True. ", )
    parser.add_argument("--replicate-times", 
                        type=int, default=1,  
                        help="poinsoning rate {1, 2, 4, 8, 16} of targeted attack", )
    parser.add_argument("--is-finetune", 
                        action="store_true", default=False, 
                        help="if fine-tuning, set True", )
    parser.add_argument("--pre-dir", 
                        type=str, default='fine_tuned', 
                        help="directory of pretrained models for fine tuning", )
    parser.add_argument("--pre-epochs", 
                        type=int, default=200, 
                        help="pretrained epochs for finetuning.", )
    parser.add_argument("--pre-lr", 
                        type=float, default=0.001, 
                        help="learning rate of pretrained model for finetuning.", )

    args = parser.parse_args()

    return args
