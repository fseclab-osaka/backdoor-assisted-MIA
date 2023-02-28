# 実行例
`optimizer`は初期値が`MSGD`なので注意してください

## Untargeted Attack with Poisoning
Epoch: 200  
```
python train_model.py --truthserum untarget --epochs 200
python attack_lira.py --truthserum untarget --epochs 200
```

## Untargeted Attack with Clean-only
Epoch: 200  
```
python train_model.py --truthserum untarget --isnot-poison --epochs 200
python attack_lira.py --truthserum untarget --isnot-poison --epochs 200
```

## Targeted Attack with Poisoning
Epoch: 200  
Poisoning rate: 250*1
```
python train_model.py --truthserum target --replicate-times 1 --epochs 200
python attack_lira.py --truthserum target --replicate-times 1 --epochs 200
```

## Targeted Attack with IJCAI
Epoch: 200
Poisoning rate: 250*2
```
python train_model.py --truthserum target --replicate-times 2 --poison-type ijcai --epochs 200
python attack_lira.py --truthserum target --replicate-times 2 --poison-type ijcai --epochs 200
```

## Targeted Attack with LIRA
Epoch 200
Poisoning rate: 250*4
```
python train_model.py --truthserum target --replicate-times 4 --poison-type trigger_generation --epochs 200
python train_model.py --truthserum target --replicate-times 4 --poison-type backdoor_injection --epochs 100 --pre-dir TRIGGER_GENERATION --is-finetune
python attack_lira.py --truthserum target --replicate-times 4 --poison-type backdoor_injection --epochs 100
```

# Backdoorの拡張
1. `BACKDOOR_NAME`フォルダを任意のバックドア名に変更 (全て大文字)
2. `BACKDOOR_NAME`フォルダ 内のファイル、関数を任意の処理に変更 (必要に応じて処理を削除する)
3. `BACKDOOR_NAME`で各pythonファイル内を検索し、変更箇所を調べる  
   (`train_model.py`内`train_shadow()`, `common.py`内`import`/`train_loop()`4箇所, `data_utils.py`内`prepare_test_loader()`/`make_poison_set()`)
4. `BACKDOOR_NAME`/`backdoor_name`をそれぞれ上記1.で決めたバックドア名に変更 (`backdoor_name`は小文字が望ましい)
5. 上記2.で作成した関数を呼び出すために、上記3.と同じ箇所にコメントにしたがって任意の処理を記載
6. `--poison-type backdoor_name` を引数につけて、`train_model.py`/`attack_lira.py`を実行 (`backdoor_name`は任意のバックドア名)
