# Wood Micrograph Classification

木材顕微鏡画像を対象とした **深層学習による種分類フレームワーク** です。
`ConvNeXt/ViT` をバックボーンに用い、**SubCenter-ArcFace** と **Center Loss** を組み合わせることで、より判別性の高い特徴表現を学習します。
また、データ分割・クラス不均衡対策・EMA・TTA など、実験の安定性を重視した設計になっています。

---

## 📂 リポジトリ構成

```
src/
 ├── config_utils.py   # config.yaml 読み込み & 検証
 ├── data_pipeline.py  # データ前処理, 分割, DataLoader 構築
 ├── model.py          # FaceWoodNet (ArcFace + CenterLoss)
 ├── training.py       # Trainer (学習ループ, EMA, early stop 等)
1_train.py             # 学習エントリスクリプト
2_eval_best.py         # ベストモデル評価スクリプト (TTA有効)
config.yaml            # 設定ファイル
```

---

## 🚀 セットアップ

### 依存ライブラリ

* Python 3.10+
* PyTorch
* timm
* scikit-learn
* h5py, pandas, numpy, pyyaml

インストール例:

```bash
pip install torch torchvision timm scikit-learn h5py pandas pyyaml
```

---

## ⚙️ 設定ファイル (`config.yaml`)

学習・評価の設定はすべて `config.yaml` で管理されます。
主要なセクションは以下の通りです：

* **data**

  * `h5_path`: HDF5 データパス
  * `csv_path`: インデックスCSV
  * `split.ratios`: train/val/test 比率 (例: `[0.7, 0.2, 0.1]`)
  * `input_size`: 入力画像サイズ

* **model**

  * `n_species`: 分類クラス数
  * `backbone`: `timm` モデル名 (例: `convnextv2_tiny`)
  * `arcface`: スケール `s`, マージン `m`, サブセンター数 `subcenters`
  * `use_center_loss`: CenterLoss を使用するか

* **training**

  * `base_lr`: 学習率
  * `epochs`: 学習エポック数
  * `scheduler_type`: `CosineWarmRestarts` または `ReduceLROnPlateau`
  * `balanced_sampler`: クラス不均衡対策を有効化
  * `tta`: 評価時の Test-Time Augmentation

---

## 🏃 実行方法

### 学習 + 検証

```bash
python 1_train.py
```

* `config.yaml` の内容に従って学習を開始します
* ベストモデルは `<out_dir>/<ckpt_name>` に保存
* 学習履歴は `<out_dir>/<history_filename>` に保存
* test 評価結果は `<out_dir>/test.json` に出力

### ベストモデルで最終評価 (TTA有効)

```bash
python 2_eval_best.py
```

* `<out_dir>/<ckpt_name>` を読み込み
* TTA を強制有効化 (flip + rot90系 6-view 平均)
* 結果は `<out_dir>/test_tta.json` に保存

---

## ✨ 特徴

* **データ分割**: 個体ID単位で stratified split、リーク防止
* **レア種対応**: 画像数が少ない種は train 専属に寄せる
* **ArcFace + CenterLoss**: マージンと中心学習による判別性能向上
* **Focal Loss**: クラス不均衡に強い損失関数
* **EMA**: 安定した推論のための Exponential Moving Average
* **TTA**: 軽量4/6-viewでの Test-Time Augmentation
* **top-1 / top-5 精度**の両方を記録

---

ここで止めています。この先は例えば **ベンチマーク結果**, **参考文献**, **著者情報** などを追加できます。

---


# Wood Micrograph Classification

深層学習を用いた **木材樹種識別** のための研究用（遊び）リポジトリです。<br>
本プロジェクトでは、京都大学生存圏研究所が公開する **Xylarium Digital Database (XDD\_016)** を利用し、広葉樹光学顕微鏡画像を対象に分類・識別を行います。

顔認証を参考

## 📖 Dataset: XDD\_016

* **タイトル（英語）**: Xylarium Digital Database for Wood Information Science and Education (XDD\_016)
* **タイトル（日本語）**: 木材情報学と教育用材鑑調査室デジタルデータベース (XDD\_016)
* **内容**:

  * Hardwood optical micrographs
  * 7 families, 33 genera, 119 species, 540 individuals
  * **7051 images**
  * 実観察領域: 2.7 × 2.7 mm²
  * 画像サイズ: 900 × 900 px
  * 解像度: 2.96 µm/px


## ⚖️ 著作権・利用条件

* **著作権者**: 京都大学生存圏研究所 データベース全国共同利用専門委員会 (RISH-DATABASE)
* **利用条件**:

  * 非営利の研究・教育目的での利用は自由
  * 謝辞と引用を必須
  * 無断コピーや改変した電子形式での再配布は禁止

**謝辞の表記例**:

* Collaborative Researches using Database, RISH, Kyoto University
* 京都大学生存圏研究所 データベース利用型共同利用


## 📌 引用方法

以下の文献を必ず引用してください：

> Junji SUGIYAMA, Sung Wook HWANG, ShengCheng ZHAI, Kayoko KOBAYASHI, Izumi KANAI, Keiko KANAI (2020).
> *Xylarium Digital Database for Wood Information Science and Education (XDD\_016)* \[dataset].
> doi:10.14989/XDD\_016


## 🔗 関連リンク

* **DOI**: [10.14989/XDD\_016](https://doi.org/10.14989/XDD_016)
* **URI**: [http://hdl.handle.net/2433/250046](http://hdl.handle.net/2433/250046)
* **コレクション**: 木材情報学と教育用材鑑調査室デジタルデータベース