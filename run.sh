python attack_lira.py --dataset cifar10 --network ResNet18 --disable-dp --replicate_times 1 --model-dir (replicate_times=1のときのmodel_dir) --device cuda:0 > ./logs/attack_target1.log

python attack_lira.py --dataset cifar10 --network ResNet18 --disable-dp --replicate_times 2 --model-dir (replicate_times=2のときのmodel_dir) --device cuda:0 > ./logs/attack_target1.log

python attack_lira.py --dataset cifar10 --network ResNet18 --disable-dp --replicate_times 4 --model-dir (replicate_times=4のときのmodel_dir) --device cuda:0 > ./logs/attack_target1.log

python attack_lira.py --dataset cifar10 --network ResNet18 --disable-dp --replicate_times 8 --model-dir (replicate_times=8のときのmodel_dir) --device cuda:0 > ./logs/attack_target1.log

python attack_lira.py --dataset cifar10 --network ResNet18 --disable-dp --replicate_times 16 --model-dir (replicate_times=16のときのmodel_dir) --device cuda:0 > ./logs/attack_target1.log