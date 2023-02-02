python attack_lira.py --dataset cifar10 --network ResNet18 --disable-dp --replicate_times 1 --model-dir Target1 --device cuda:0 > ./logs/attack_target1.log

python attack_lira.py --dataset cifar10 --network ResNet18 --disable-dp --replicate_times 2 --model-dir Target2 --device cuda:0 > ./logs/attack_target2.log

python attack_lira.py --dataset cifar10 --network ResNet18 --disable-dp --replicate_times 4 --model-dir Target4 --device cuda:0 > ./logs/attack_target4.log

python attack_lira.py --dataset cifar10 --network ResNet18 --disable-dp --replicate_times 8 --model-dir Target8 --device cuda:0 > ./logs/attack_target8.log

python attack_lira.py --dataset cifar10 --network ResNet18 --disable-dp --replicate_times 16 --model-dir Target16 --device cuda:0 > ./logs/attack_target16.log