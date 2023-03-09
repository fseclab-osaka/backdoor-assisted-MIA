# Examples of Usage

## Targeted Attack without any Poisoning/Backdoor
```
python train_model.py --is-target --isnot-poison
python attack_lira.py --is-target --isnot-poison
```

## Targeted Attack with Poisoning
Poisoning rate: 250*16
```
python train_model.py --is-target --replicate-times 16 --poison-type poison
python attack_lira.py --is-target --replicate-times 16 --poison-type poison 
```

## Untargeted Attack with LIRA
```
python train_model.py --poison-type trigger_generation --epochs 200
python train_model.py --poison-type backdoor_injection --epochs 100 --is-finetune --pre-dir TRIGGER_GENERATION --pre-epochs 200
python attack_lira.py --poison-type backdoor_injection --epochs 100
```


# Expantion with Your Backdoor
1. Rename the directory `BACKDOOR_NAME` to your backdoor's name with all capital.
2. Replace the functions of the all files in the directory `BACKDOOR_NAME` with your backdoor's functions. 
You can add/remove the original functions if you need.
3. Replace the codes including `BACKDOOR_NAME` or `backdoor_name` of all python files in the parent directory with the proper codes including your backdoor's name.
The `backdoor_name` should be all lower case.
4. Run `train_model.py` and `attack_lira.py` with `--poison-type backdoor_name` after replacing `backdoor_name` with your backdoor's name.


# References of Code 

## BadNets/TaCT
```
https://github.com/verazuo/badnets-pytorch
```

## LIRA
```
https://github.com/khoadoan106/backdoor_attacks
```

## IBD
```
https://github.com/ekko-zn/ijcai2022-backdoor
```
