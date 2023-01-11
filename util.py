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
    "--is-backdoored",
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
      default='adam',
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

    # 下記、芝原さんのコードから追加
    parser.add_argument(
      "--trigger-file",
      type=str,
      default='./',
      help="file path of triggers",
    )
    parser.add_argument(
      "--checkpoint",
      type=str,
      default='./checkpoints',
      help="directory of pretrained model",
    )
    # added for badnet
    parser.add_argument(
      "--trigger_path",
      type=str,
      default='./BadNet/triggers/trigger_white.png',
      help="directory of trigger image",
    )
    parser.add_argument(
      "--trigger_size",
      type=int,
      default=5,
      help="width and height of trigger on an image",
    )
    parser.add_argument(
      "--trigger_label",
      type=int,
      default=0,
      help="class label of backdoor(image with trigger)",
    )
    parser.add_argument(
      "--poisoning_rate",
      type=float,
      default=0.1,
      help="poinsoning rate",
    )
    parser.add_argument(
      "--is-backdoored",
      action="store_true",
      default=False,
      help="If backdoor target model, it should be True",
    )

    parser.add_argument(
      "--experiment_strings",
      type=str,
      default="dataset",
      help="when backdoored model MIA, which dataset will be used.", # all : clean + backdoor, clean : clean only, backdoor : backdoor only
    )
    parser.add_argument(
      "--poison_num",
      type=int,
      default=25000, # 50% :  25000, 25% : 12500, 10% : 5000
      help="poinsoning rate",
    )
    parser.add_argument(
      "--is_save_each_epoch",
      action="store_true",
      default=False,
      help="Flag : explain whether model is saved each epoch.",
    )
    
    args = parser.parse_args()

    return args
