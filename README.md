# 実行例
## Targeted Attack with Poisoning
Epoch: 200
Poisoning rate: 250*1
python train_model.py --truthserum target --replicate-times 1 --epochs 200
python attack_lira.py --truthserum target --replicate-times 1 --epochs 200

## Untargeted Attack with IJCAI
Epoch: 200
python train_model.py --truthserum untarget --poison-type ijcai --epochs 200
python attack_lira.py --truthserum untarget --poison-type ijcai --epochs 200

# Backdoorの拡張
1. BACKDOOR_NAME フォルダを任意のバックドア名に変更(全て大文字)
2. BACKDOOR_NAME フォルダ 内のファイル、関数を任意の処理に変更
3. BACKDOOR_NAME で python ファイル内を検索し、BACKDOOR_NAME/backdoor_name をそれぞれ上で決めたバックドア名に変更後、コメントにしたがって任意の処理を記載
   (train_model.py 内 train_shadow(), common.py 内 import/train_loop() 4箇所, data_utils.py 内 prepare_test_loader()/make_poison_set())
3. --poison-type backdoor_name を引数につけて、train_model.py/attack_lira.pyを実行
