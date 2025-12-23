# Peg in hole コードチート表
基本的に[eipl](https://ogata-lab.github.io/eipl-docs/en/teach/dataset/)のコードをベースに作成

### 0. ディレクトリの作成と移動

work/<project_name>を作成する

ここで2以降のコードを実行する

```
masuda@kubo-ZA9C-R49:~/work/pick_plug$ ls
README.md  bin  data  fig  libs  log  output  pyproject.toml  utils  uv.lock
```
プロジェクトと並列にeiplリポジトリを設置する
```
masuda@kubo-ZA9C-R49:~/work$ tree -L 1| grep "pick\|eipl"
├── eipl
├── pick
├── pick_place
├── pick_plug
```
### 1. bagデータの設置
dataディレクトリ以下にdata/yyyymmdd/bagディレクトリを作成し、その下にbagデータを置く。
### 2. 1_bag2npy.pyの実行
rosbagデータをnumpy形式に変換する.
時系列が一番短い取得トピックにそのrosbagのデータの時系列を合わせる.
```
 uv run bin/1_bag2npy.py ./data/20251023/bag/pos1
```
bagディレクトリと同階層のnpyファイルにnpz形式で出力される。複数のndarrayを保存する場合はnpz形式で保存。

複数のtopicを一つのnumpyファイルに入れるから今回はnpz形式で保存。
```
masuda@kubo-ZA9C-R49:~/work/pick_plug/data/20251030$ tree
.
├── bag
│   ├── 2025-10-30-07-22-25.bag
│   └── 2025-10-30-07-26-07.bag
└── npy
    ├── 2025-10-30-07-22-25.npz
    └── 2025-10-30-07-26-07.npz
```
### 3. 2_make_video.pyの実行
numpyデータが取れているかどうか確認する
```
uv run bin/2_make_video.py ./data/20251030/npy
```
work/<project_name>/fig以下にgifが生成される

```
masuda@kubo-ZA9C-R49:~/work/pick_plug/fig$ tree
.
└── 20251030
    ├── 2025-10-30-07-22-25.gif
    └── 2025-10-30-07-26-07.gif
```

### 4. 3_make_dataset.pyの実行
npzの中にすべてのトピックが入っている状態から、各トピックをそれぞれ別のnpyファイルに振り分ける

画像のresizeはここで行い、すべてのnpyデータのうち最も時系列が長いデータに時系列の長さを合わせる。その際はデータ内の最後の時系列のデータで残りの時間を埋める。

```
uv run bin/3_make_dataset.py data/20251112_redsylinder/npy/ --output_num "2"
```
output_numで切り抜きの種類によってデータを分けられる。下の"記録"にて切り抜き内容を記録する。

結果はこんな感じ

data/ test/ train/ ディレクトリができる。
```
masuda@kubo-ZA9C-R49:~/work/pick_plug/data/20251030$ tree
.
├── bag
│   ├── 2025-10-30-07-22-25.bag
│   └── 2025-10-30-07-26-07.bag
├── data
│   ├── left_arm_states.npy
│   ├── left_hand_imgs.npy
│   ├── left_imgs.npy
│   ├── right_arm_states.npy
│   ├── right_digit_imgs.npy
│   ├── right_hand_imgs.npy
│   ├── right_imgs.npy
│   ├── teleop_left_arm_states.npy
│   └── teleop_right_arm_states.npy
├── npy
│   ├── 2025-10-30-07-22-25.npz
│   └── 2025-10-30-07-26-07.npz
└── param
    ├── arm_state_bound.npy
    └── teleop_arm_state_bound.npy
```

### 5. 4_check_data.pyの実行

これでtrain.pyに入力する直前のデータを出力できる。
画像はuint8で64*64で出力される。触覚はfloat32で出力される。

```
uv run bin/4_check_data.py data/20251112_redsylinder2/test/
```

<projectname>/fig/<yyyymmdd>/train_data_<idx>.gifとして保存される。

### 6. 5_train.pyの実行

ここで学習を行い、重みを更新していく。

- eipl.dataのMultiModalDatasetクラスを改造したMultiModalTactileDatasetクラスを使用している。

    - MultiModalTactileDatasetでは、各モダリティにガウシアンノイズを付与することでノイズロバストなデータを作る。

    - MultiModalTactileDatasetにはとりあえず右目、右手関節、右手触覚画像の3Modalityのみ入れている

- eipl.modelのSARNN.pyを改造したTACTILE_SARNN.py内のTACTILESARNNクラスを使用している

    - この中に実際のモデルが書かれているので詳細を確認
- deviceでgpuの番号を指定する。
    - データディレクトリは/まで入れる
```
uv run bin/5_train.py data/20251112_redsylinder/
```

logに重みが保存される
```
masuda@kubo-ZA9C-R49:~/work/peg_in_hole/log$ tree 20251125_2026_57/
20251125_2026_57/
├── SARNN.pth
├── args.json
└── events.out.tfevents.1764070019.kubo-ZA9C-R49.1150835.0
```
### 6.a Lossの確認
以下のtensorboardから損失のグラフが確認可能
```
masuda@kubo-ZA9C-R49:~/work/peg_in_hole/log$ tensorboard --logdir=./
TensorFlow installation not found - running with reduced feature set.
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.14.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

### 7. 6.test.pyの実行
オフラインテストを行うコード
その他のinputの画像はtrainから持ってくる（コードを書き換える必要あり）
ココらへんを変える
```
_images=np.load("./data/20251112_redsylinder/test/right_imgs.npy")
_joints=np.load("./data/20251112_redsylinder/test/right_arm_states.npy")
_images_tactile=np.load("./data/20251112_redsylinder/test/right_digit_imgs.npy")
```
実行
```
uv run bin/6_test.py --filename log/20251125/2026/57/SARNN.pth
```

# 記録
- 20251112_redsylinder
- 20251112_redsylinder_1
    ```
    right_imgs = npz_data["right_imgs"][np.newaxis,:,55:-110,155:-50,:]
    ```
    で切り抜き
- 20251112_redsylinder_2
    ```
    [np.newaxis,:,55:-110,205:-55,:]
    ```
- red_sylinder 
    右目＋触覚画像のみのrosbagを取り直した
    ```
    [np.newaxis,:,55:-110,205:-55,:]
    ```