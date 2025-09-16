# Wood Micrograph Classification

> 夏休みの自由研究

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.13-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.6-orange.svg">
  <a href="LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg">
  </a>
</p>

本リポジトリは、**広葉樹の光学顕微鏡画像**を対象とした **種分類（species classification）** の実装です。

## 特徴

* **バックボーン**: `timm` ライブラリを用いた ConvNeXt / ViT / Swin など（本実装では `Swin` を採用）
* **ヘッド**: SubCenter-ArcFace による角距離学習
* **補助損失**: Center Loss を任意で併用可能
* **データ前処理**: HDF5 データセットを読み込み、個体単位で層化分割。レア種は train にのみ配置
* **学習戦略**:

  * Focal Loss（ラベル平滑化対応）
  * ArcFace margin / Center Loss λ のウォームアップ
  * 学習率スケジューラ: CosineAnnealingWarmRestarts または ReduceLROnPlateau
  * EMA（Exponential Moving Average）
  * Balanced Sampler によるクラス不均衡補正

## 実行スクリプト

* `1_train.py` : 学習と test 評価、結果を JSON 保存
* `2_eval_best.py` : best checkpoint をロードし **TTA 評価**を実行
* `3_visualize_runs.py` : history.json と test 結果を可視化（学習曲線や精度）
* `4_eval_confusion_and_examples.py` : 混同行列と成功/失敗例を出力

---

## 背景と課題

当初は ResNet など一般的な分類モデルでの学習を試みましたが、性能はほとんど出ませんでした。本タスクは **顔認証に近い難しさ**（クラス間差は微弱だが個体内変動が大きい）を持つため、ArcFace + SubCenter といった顔認証系の手法を導入しました。その結果、安定して分類が成立する段階に到達しました。

しかし、**過学習は依然として強く、汎化性能の改善が大きな課題**として残っています。本実装ではデータ分割や損失設計、EMA などを組み合わせて過学習の抑制を試みていますが、最終的な性能はデータ拡張やモデル設計に強く依存しています。

## Dataset（Xylarium Digital Database: XDD\_016）

**Xylarium Digital Database for Wood Information Science and Education** に含まれる広葉樹の光学顕微鏡像を用いています。

* DOI: [10.14989/XDD\_016](https://doi.org/10.14989/XDD_016)
* URI: [http://hdl.handle.net/2433/250046](http://hdl.handle.net/2433/250046)
* コレクション: 木材情報学と教育用材鑑調査室デジタルデータベース

### Overview

* Families: 7
* Genera: 33
* Species: 119
* Individuals: 540
* Images: 7,051
* 実観察領域: 2.7 × 2.7 mm² / 画像: 900 × 900 px / 解像度: 2.96 µm/px

<details>
<summary><b>Family 別の概要（上位のみ）</b></summary>

| family       | n\_species | images |
| ------------ | ---------- | ------ |
| Fagaceae     | 18         | 2446   |
| Lauraceae    | 39         | 1658   |
| Magnoliaceae | 18         | 926    |
| Betulaceae   | 19         | 817    |
| Sapindaceae  | 18         | 444    |
| Ulmaceae     | 4          | 443    |
| Cannabaceae  | 3          | 317    |

</details>

<details>
<summary><b>Species 別 枚数上位（Top-10）</b></summary>

| species                | images |
| ---------------------- | ------ |
| Quercus\_crispula      | 266    |
| Fagus\_crenata         | 225    |
| Cinnamomum\_camphora   | 221    |
| Machilus\_thunbergii   | 210    |
| Quercus\_salicina      | 188    |
| Fagus\_japonica        | 180    |
| Litsea\_coreana        | 180    |
| Castanea\_crenata      | 177    |
| Quercus\_myrsinifolia  | 168    |
| Cinnamomum\_yabunikkei | 158    |

</details>

## 結果

（準備中）

---

## Acknowledgements

> 本研究（実装）では、京都大学生存圏研究所 Xylarium Digital Database (XDD\_016) を利用しました。京都大学生存圏研究所 データベース全国共同利用専門委員会 (RISH-DATABASE) に深く感謝いたします。また、本成果は京都大学生存圏研究所 データベース利用型共同利用の支援によるものです。

## Citation

> Junji SUGIYAMA, Sung Wook HWANG, ShengCheng ZHAI, Kayoko KOBAYASHI, Izumi KANAI, Keiko KANAI (2020).
> *Xylarium Digital Database for Wood Information Science and Education (XDD\_016)* \[dataset].
> DOI: 10.14989/XDD\_016 — URI: [http://hdl.handle.net/2433/250046](http://hdl.handle.net/2433/250046)

## Licence

本コードは **MIT License** です。詳細は [LICENSE](LICENSE) を参照してください。