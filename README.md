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

<img src="results/readme_cells.png">

## 特徴

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
  * [Config](config.yaml)

### 実行スクリプト

* `1_train.py` : 学習と test 評価、結果を JSON 保存
* `2_eval_best.py` : best checkpoint をロードし TTA 評価を実行
* `3_visualize_runs.py` : history.json と test 結果を可視化
* `4_eval_confusion_and_examples.py` : 混同行列と成功/失敗例を出力

---

## 背景と課題

本タスクは **同一 family／同一 genus 内での種判別** を要し、類似した微細構造や組織配列が多数存在します。クラス間マージンが極端に小さい一方で、個体差（樹体差・切片条件・染色/照明差）によるクラス内分散も大きく、実質的に「双子を見分けるより難しい」レベルの微差判定です。（個人の意見）
広域的な組織構造を手がかりとするべき本タスクにおいて、汎用CNN（例：ResNet）では適切な分離境界を形成できず失敗しました。

そこで、**顔認証**に近い難しさだと捉え、角距離ベースの SubCenter-ArcFace（margin ウォームアップ）を採用。バックボーンは **MaxViT (maxvit_tiny_rw_224)** を採用し、必要に応じて Center Loss を併用、学習の安定化に EMA と CosineAnnealingWarmRestarts を使っています。データは 個体IDで層化分割し、レア種は train 専属。学習時は Flip/Rotate(90°)、Scale Jitter(±15%)、Brightness/Contrast(±10%, ±0.05)、Gamma(±10%) を弱めに適用し、検証/テストでは Resize + Normalize のみ（TTA は最終評価時）としています。

それでも **汎化性能** は依然として主要課題です。未知個体への頑健性や確率の過信（CE の悪化）には改善の余地があり、**サンプリング設計と前処理/Aug の最適化** が精度を大きく左右すると考えられます。本実装はまず安定に分類が成立するベースラインを提供するもので、今後はこの土台の上で **一般化の強化** を進める必要があります。


## Dataset（Xylarium Digital Database: XDD\_016）

**Xylarium Digital Database for Wood Information Science and Education** に含まれる広葉樹の光学顕微鏡像を用いています。

* DOI: [10.14989/XDD\_016](https://doi.org/10.14989/XDD_016)
* URI: [http://hdl.handle.net/2433/250046](http://hdl.handle.net/2433/250046)
* コレクション: 木材情報学と教育用材鑑調査室デジタルデータベース

#### Overview

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

広葉樹顕微鏡画像を用いた種分類 において、学習の推移・テスト評価・精度をまとめました。

### 精度の概要

本研究では、学習過程において 検証精度（ `val_acc` ）が最大となったエポック を `best_model` として保存しました。本モデルでは epoch 21 で `val_acc` が 0.724 に達し、この時点の重みを評価対象としています。

* **学習精度**
  
  * Top-1 Accuracy: **0.981**
  * Top-5 Accuracy: **0.992**


* **検証精度**
  
  * Top-1 Accuracy: **0.724**
  * Top-5 Accuracy: **0.822**


* **テスト精度（TTAあり）**

  * Top-1 Accuracy: **0.798**
  * Top-5 Accuracy: **0.930**


<p align="center">
<img src="results/training_tta_acc.png">
<b>Fig. 1</b> 学習精度の履歴
</p>


学習記録の詳細は[こちら](runs/history.json)

### 分類の成功例と課題

* **成功例**

  * *Aesculus turbinata*, *Aphananthe aspera*, *Castanea crenata*, *Celtis sinensis*, *Corylus sieboldiana* などでは **precision/recall/F1 = 1.0** と完全分類を達成。
  * また、*Ostrya japonica*（F1≈0.99）、*Cinnamomum camphora*（F1≈0.96）、*Litsea coreana*（F1≈0.95）など、多くの主要樹種で高精度を示しました。


* **課題のある種**

  * *Carpinus japonica*（F1≈0.31）、*Quercus acuta*（F1≈0.47）、*Quercus salicina*（F1≈0.47）など、一部の樹種で大きな誤分類が確認されました。
  * 特に **ブナ科（Quercus 属）間** では同属内での取り違えが頻発し、混同行列でもその傾向が表れています。


<p align="center">
<img src="results/confusion_matrix_norm_filtered.png"><br>
<b>Fig. 2</b> テストデータセットにおける混同行列
</p>

さらに、成功例と失敗例の一部を可視化し、モデルの挙動を直感的に把握できるようにしました。<br>
分類結果の詳細（全部）については[こちら](Classification_Report.md) <br>
`T`: True , `P`: Predict

* **正解例（一部）**

<p align="center">
<img src="results/success_grid.png"><br>
<b>Fig. 3</b> テストデータセットにおける分類成功例
</p>

* **誤分類例（一部）**

<p align="center">
<img src="results/failure_grid.png"><br>
<b>Fig. 4</b> テストデータセットにおける分類失敗例
</p>

### まとめ

本モデルは **Top-1 精度 ≈ 0.80、Top-5 精度 ≈ 0.93** と高い分類性能を実現しました。一方で、同属種の識別には依然として課題が残されており、特に **Quercus 属の細分類**やサンプル数の少ないクラスでの改善が今後の焦点となります。また、汎化性能についても今後の課題です。

---

## Acknowledgements

> 本研究（実装）では、京都大学生存圏研究所 Xylarium Digital Database (XDD\_016) を利用しました。京都大学生存圏研究所 データベース全国共同利用専門委員会 (RISH-DATABASE) に深く感謝いたします。また、本成果は京都大学生存圏研究所 データベース利用型共同利用の支援によるものです。

## Citation

> Junji SUGIYAMA, Sung Wook HWANG, ShengCheng ZHAI, Kayoko KOBAYASHI, Izumi KANAI, Keiko KANAI (2020).
> *Xylarium Digital Database for Wood Information Science and Education (XDD\_016)* \[dataset].
> DOI: 10.14989/XDD\_016 — URI: [http://hdl.handle.net/2433/250046](http://hdl.handle.net/2433/250046)

## Licence

本コードは **MIT License** です。詳細は [LICENSE](LICENSE) を参照してください。