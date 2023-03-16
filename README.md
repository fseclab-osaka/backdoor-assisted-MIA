# Install
```
$ git clone https://github.com/fseclab-osaka/backdoor-assisted-MIA.git
$ cd backdoor-assisted-MIA
$ conda env create -n backdoor-assisted-MIA -f env.yml
$ conda activate backdoor-assisted-MIA
```

# Usage

```train_model.py```: train victim and shadow models with poisoning/backdoor

```attack_lira.py```: membership inference attack the victim model with the shadow models for a full leave-one-out cross-validation


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
1. Rename the directory `BACKDOOR_NAME` to your backdoor's name in all upper cases.
2. Replace the functions of all files in the directory `BACKDOOR_NAME` with your backdoor's functions. 
You can add/remove functions if you need.
3. Replace the codes, including `BACKDOOR_NAME` or `backdoor_name`, of all python files in the repository with the proper codes, including your backdoor's name.
We recommend that the `backdoor_name` is replaced in all lowercase.
4. Run `train_model.py` and `attack_lira.py` with the argument `--poison-type backdoor_name` that is replaced `backdoor_name` with your backdoor's name.


# References of Code 
[badnets]:https://github.com/verazuo/badnets-pytorch

- Truth serum: Poisoning machine learning models to reveal their secrets
- [Badnets: Identifying vulnerabilities in the machine learning model supply chain][badnets]
- [Demon in the Variant: Statistical Analysis of DNNs for Robust Backdoor Contamination Detection][badnets]
- [Lira: Learnable, imperceptible and robust backdoor attacks](https://github.com/khoadoan106/backdoor_attacks)
- [Imperceptible backdoor attack: From input space to feature representation](https://github.com/ekko-zn/ijcai2022-backdoor)
- [Reverse engineering imperceptible backdoor attacks on deep neural networks for detection and training set cleansing](https://github.com/zhenxianglance/RE-paper)
