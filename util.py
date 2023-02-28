import argparse
"""
    param : --train-batch-size : 
    param : --test-batch-size : 
    param : -n, --epochs : 
    param : -r, --n-runs : 
    param : --lr : 
    param : --optimizer : 
    param : --sigma :
    param : -c,--max-per-sample-grad_norm :
    param : --delta :
    param : --device :
    param : --disable-dp :
    param : --data-root :
    param : --exp-idx :
    param : --network :
    param : --dataset :
    param : --model-dir :

    # 芝原さんのコードから加えるべき引数
    "--trigger-file",
    "--checkpoint",
    "--trigger_path",
    "--trigger_size",
    "--trigger_label",
    "--poisoning_rate",
    "--is-poison",
    "--experiment_strings",
    "--poison_num",
    "--is_save_each_epoch",
"""

def get_arg():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
      "--train-batch-size",
      type=int,
      default=256,
      help="input batch size for testing (default: 512)",
    )
    parser.add_argument(
      "--test-batch-size",
      type=int,
      default=1024,
      help="input batch size for testing (default: 1024)",
    )
    parser.add_argument(
      "-n",
      "--epochs",
      type=int,
      default=100,
      metavar="N",
      help="number of epochs to train",
    )
    parser.add_argument(
      "-r",
      "--n-runs",
      type=int,
      default=1,
      metavar="R",
      help="number of runs",
    )
    parser.add_argument(
      "--lr",
      type=float,
      default=0.001,
      metavar="LR",
      help="learning rate",
    )
    parser.add_argument(
      "--optimizer",
      type=str,
      default='MSGD',
      help="optimizer",
    )
    parser.add_argument(
      "--sigma",
      type=float,
      default=1.0,
      metavar="S",
      help="Noise multiplier",
    )
    parser.add_argument(
      "-c",
      "--max-per-sample-grad_norm",
      type=float,
      default=0.1,
      metavar="C",
      help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
      "--delta",
      type=float,
      default=1e-5,
      metavar="D",
      help="Target delta (default: 1e-5)",
    )
    parser.add_argument(
      "--device",
      type=str,
      default="cuda",
      help="GPU ID for this process (default: 'cuda')",
    )
    parser.add_argument(
      "--disable-dp",
      action="store_true",
      default=True,
      help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
      "--data-root",
      type=str,
      default="data/",
      help="Where dataset is/will be stored",
    )
    parser.add_argument(
      "--exp-idx",
      type=int,
      default=0,
      help="index of experiments",
    )
    parser.add_argument(
      "--network",
      type=str,
      default='ResNet18',
      help="Network name",
    )
    parser.add_argument(
      "--dataset",
      type=str,
      default='cifar10',
      help="dataset name",
    )
    parser.add_argument(
      "--model-dir",
      type=str,
      default='model',
      help="directory to save models",
    )
    parser.add_argument(
      "--poison-label",
      type=int,
      default=0,
      help="Poison target class (label of image with trigger)",
    )
    parser.add_argument(
      "--isnot-poison",
      action="store_true",
      default=False,
      help="If target model will not be poisoned, it should be True",
    )
    parser.add_argument(
      "--truthserum",
      type=str,
      default="target",
      help="set as 'target' or 'untarget'. ",
    )
    parser.add_argument(
      "--replicate-times",
      type=int,
      default=1, # 1, 2, 4, 8, 16
      help="poinsoning rate",
    )
    parser.add_argument(
      "--poison-type",
      type=str,
      default="poison",
      help="poison, ijcai, or lira.",
    )
    parser.add_argument(
      "--is-finetune",
      action="store_true",
      default=False,
      help="if fine-tuning, set True",
    )
    parser.add_argument(
      "--pre-dir",
      type=str,
      default='fine_tuned',
      help="directory of pretrained models for fine tuning",
    )
    parser.add_argument(
      "--pre-epochs",
      type=int,
      default=200,
      help="number of pretrained epochs for finetuning.",
    )
    parser.add_argument(
      "--pre-lr",
      type=float,
      default=0.001,
      help="learning rate of pretrained model for finetuning.",
    )

    args = parser.parse_args()

    return args
