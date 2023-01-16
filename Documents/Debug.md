# untargetについて
```
2023-01-13
untarget TruthSerumの実装において下記の事実を確認した。

・インデックスの保存
・backdoor画像の枚数の確認(5000であるかどうか。)
・学習ができること


``` 

# targetについて
```
2023-01-13
targetでreplicateとindicesが少なくともbackdoor時に固定されていることを確認。
labelがすべて0になっていることも確認。

replicateに対して、データセットの数が正しいことも確認。
targetのデータセットで実験できることを確認した。


```
# テストデータセットについて
```
2023-01-13
test_datasetの数がbackdoor clean共に同じであることを確認。
```

# shadow model ごとのin out について
```
2023-01-15
shadow modelごとのtarget datasetの分割において, インデックスを保存するように変更した.
これを用いて, in out をメンバーシップ推定攻撃時に判断するように変更する.
```

# テンプレート
```
2023-01-**
```