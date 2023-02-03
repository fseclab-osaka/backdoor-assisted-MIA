python train_model.py --dataset cifar10 --network ResNet18 --disable-dp --replicate_times 1 --model-dir Untarget1 --device cuda:0 > ./logs/train_untarget1.log &

python train_model.py --dataset cifar10 --network ResNet18 --disable-dp --replicate_times 2 --model-dir Untarget2 --device cuda:1 > ./logs/train_untarget2.log

python train_model.py --dataset cifar10 --network ResNet18 --disable-dp --replicate_times 4 --model-dir Untarget4 --device cuda:1 > ./logs/train_untarget4.log &

python train_model.py --dataset cifar10 --network ResNet18 --disable-dp --replicate_times 8 --model-dir Untarget8 --device cuda:0 > ./logs/train_untarget8.log

python train_model.py --dataset cifar10 --network ResNet18 --disable-dp --replicate_times 16 --model-dir Untarget16 --device cuda:0 > ./logs/train_untarget16.log