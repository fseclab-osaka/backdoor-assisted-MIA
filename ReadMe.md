# 環境
下記にpip listを表示いたします.
```
Package                  Version   
------------------------ ----------
antlr4-python3-runtime   4.9.3     
certifi                  2022.12.7 
charset-normalizer       2.1.1     
contourpy                1.0.6     
cycler                   0.11.0    
fonttools                4.38.0    
functorch                1.13.0    
Hydra                    2.5       
idna                     3.4       
joblib                   1.2.0     
kiwisolver               1.4.4     
matplotlib               3.6.2     
numpy                    1.24.1    
nvidia-cublas-cu11       11.10.3.66
nvidia-cuda-nvrtc-cu11   11.7.99   
nvidia-cuda-runtime-cu11 11.7.99   
nvidia-cudnn-cu11        8.5.0.96  
omegaconf                2.3.0     
opacus                   1.3.0     
opt-einsum               3.3.0     
packaging                22.0      
pandas                   1.5.2     
Pillow                   9.4.0     
pip                      20.0.2    
pkg-resources            0.0.0     
pyparsing                3.0.9     
python-dateutil          2.8.2     
pytz                     2022.7    
PyYAML                   6.0       
requests                 2.28.1    
scikit-learn             1.2.0     
scipy                    1.10.0    
setuptools               44.0.0    
six                      1.16.0    
threadpoolctl            3.1.0     
torch                    1.13.1    
torchvision              0.14.1    
tqdm                     4.64.1    
typing-extensions        4.4.0     
urllib3                  1.26.13   
wheel                    0.38.4   
```
# ファイル・ディレクトリなど
```
train_model.py ターゲットモデルとshadowモデルの学習
attack_lira.py memberとnon-memberでのlogitの平均と分散の計算、MIAの実行
common.py データセットの読み込み、学習関連のプログラム
          resnetの実装をtorchvisionからresnet.pyに変更
          学習係数を80epochで0.1倍（DPなしの場合のみ）common.py l.65 l.188
network.py 独自ネットワークの定義
util.py デフォルトのパラメータ

data データセット保存のディレクトリ
model 学習済みモデル保存のディレクトリ
result 実験ログ保存のディレクトリ
```

# 実行例
```
DP：なし
データセット：CIFAR100
ネットワーク：ResNet18
shadow modelの数：20
python train_model.py --dataset cifar100 --network ResNet18 --disable-dp
python attack_lira.py --dataset cifar100 --network ResNet18 --disable-dp

テスト精度：64%
MIAの平均精度：78%
```
shadow modelが少ないからか複数クエリ使用するとMIAの精度が下がったため、各サンプル1クエリの実装になっています。


# 実装に関する補足
```
weight decay: 1e-4
data augmentation: cifar shift+flip, mnist shift
データセット
　教師データ：ターゲットモデル用の教師データと攻撃データセットに2分割
　テストデータ：ターゲットサンプルとして使用
　shadowモデルの学習：ターゲットモデル用の教師データから20,000サンプルランダムに選択、ターゲットサンプルから半数ランダムに選択
　ターゲットモデルの学習：攻撃データセットから20,000サンプルランダムに選択、ターゲットサンプルから半数ランダムに選択
```

# Acknowledgments / 謝辞
論文「[Membership Inference Attacks From First Principles](https://arxiv.org/abs/2112.03570)」のメンバーシップ推定攻撃のコードは全てNTT社会情報研究所の芝原さんに書いていただいています. 誠に感謝申し上げます.