# 実行例
## Targeted Attack with Poisoning
Epoch: 200
Poisoning rate: 250*1
```
python train_model.py --truthserum target --replicate-times 1 --epochs 200
python attack_lira.py --truthserum target --replicate-times 1 --epochs 200
```

## Untargeted Attack with IJCAI
Epoch: 200
```
python train_model.py --truthserum untarget --poison-type ijcai --epochs 200
python attack_lira.py --truthserum untarget --poison-type ijcai --epochs 200
```

# Backdoorの拡張
1. `BACKDOOR_NAME`フォルダを任意のバックドア名に変更(全て大文字)
2. `BACKDOOR_NAME`フォルダ 内のファイル、関数を任意の処理に変更
3. `BACKDOOR_NAME`で各pythonファイル内を検索し、変更箇所を調べる
   (`train_model.py`内`train_shadow()`, `common.py`内`import`/`train_loop()`4箇所, `data_utils.py`内`prepare_test_loader()`/`make_poison_set()`)
4. `BACKDOOR_NAME`/`backdoor_name`をそれぞれ上記1.で決めたバックドア名に変更
5. 上記2.で作成した関数を呼び出すために、上記3.と同じ箇所のコメントにしたがって任意の処理を記載
6. `--poison-type backdoor_name` を引数につけて、`train_model.py`/`attack_lira.py`を実行