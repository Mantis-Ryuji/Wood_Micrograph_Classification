<img src="results/WOODLOGO.png" alt="logo" width="120" align="right">

# Wood Micrograph Classification

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.13-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.6-orange.svg">
  <a href="LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg">
  </a>
</p>

> 夏休みの自由研究（M1）

本リポジトリは、**広葉樹の光学顕微鏡画像**を対象とした **種分類（species classification）** の実装です。

### 特徴

* **バックボーン**: `timm` ライブラリ（ConvNeXt / ViT / Swin / MaxViT など）。本実装は `MaxViT (maxvit_tiny_rw_224)` を採用
* **ヘッド**: SubCenter-ArcFace（サブセンタ対応, `subcenters=4`）による角距離学習
* **補助損失**: Center Loss を併用
* **データ前処理**: HDF5 データセットを読み込み、個体単位で層化分割。レア種は train のみに配置  
  **学習時 Aug**: Flip/Rotate(90°), Scale Jitter(±15%), Brightness/Contrast(±10%, ±0.05), Gamma(±10%)  
  **検証/テスト**: Resize + Normalize（TTA は最終評価時に使用）
* **学習戦略**:
  * Cross-Entropy + Label Smoothing（既定）／Focal Loss は任意で切替
  * ArcFace（SubCenter, s/m）：margin のウォームアップ、Center Loss λ のウォームアップ（任意）
  * 学習率スケジューラ: CosineAnnealingWarmRestarts（再スタート間隔は長め）／ReduceLROnPlateau（任意）
  * EMA（Exponential Moving Average）
  * Balanced Sampler によるクラス不均衡補正

> 備考: 汎化性能は依然として課題であり、個体依存の偏りやパターン過信を抑えるために、サンプリングと前処理/Aug の設計を改良する必要があります。

### 実行スクリプト

* `1_train.py` : 学習と test 評価、結果を JSON 保存
* `2_eval_best.py` : best checkpoint をロードし TTA 評価を実行
* `3_visualize_runs.py` : history.json と test 結果を可視化
* `4_eval_confusion_and_examples.py` : 混同行列と成功/失敗例を出力

---

### 背景と課題

このタスクは **同一 family／同一 genus 内の種を見分ける**必要があり、局所構造や組織配列がほぼ共通です。体感としては **「双子の識別より難しい」**レベルの微差判定で、さらに **個体差（試料差・切片条件）や撮影条件差**が上乗せされます。結果として、従来の CE ベース分類器（例: ResNet）では太刀打ちできませんでした。

そこで、顔認証に近い難しさだと捉え、**角距離ベースの SubCenter-ArcFace（margin ウォームアップ）**を採用。バックボーンは **MaxViT (maxvit_tiny_rw_224)** に統一し、必要に応じて **Center Loss** を併用、学習の安定化に **EMA** と **CosineAnnealingWarmRestarts** を使っています。データは **個体IDで層化分割**し、**レア種は train 専属**。学習時は **Flip/Rotate(90°)、Scale Jitter(±15%)、Brightness/Contrast(±10%, ±0.05)、Gamma(±10%)** を弱めに適用し、検証/テストでは **Resize + Normalize のみ**（TTA は最終評価時のみ任意）としています。

それでも **汎化性能は依然として主要課題**です。未知個体・未知撮影条件への頑健性や確率の過信（CE の悪化）には改善の余地があり、**サンプリング設計と前処理/Aug の最適化**が精度を大きく左右します。本実装はまず **安定に分類が成立するベースライン**を提供するもので、今後はこの土台の上で **一般化の強化**を進めていきます。


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