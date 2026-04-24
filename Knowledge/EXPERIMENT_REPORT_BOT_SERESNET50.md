# Urban2026 Experiment Report - BoT + SE-ResNet-50

## 1. Executive Summary

This report summarizes the current `BoT + SE-ResNet-50` experiment carried out in the `Part-Aware-Transformer` repository and compares it directly against the paper **"Data-Centric and Model-Centric Enhancements for Urban Object Re-Identification"**.

The most important conclusion after comparing the paper to the current code and results is this:

- the current codebase has been **successfully adapted** to train an `SE-ResNet-50` model and generate Kaggle submission CSV files
- the current run is **not yet a faithful reproduction** of the paper's full pipeline
- the strongest missing pieces are **super-resolution, style transfer, query majority voting, exact validation protocol, and several paper-specific training decisions**
- therefore, the current Kaggle score should **not** be interpreted as evidence that the paper's method failed

The paper reports:

- BoT baseline: `19.90 mAP`
- optimized BoT pipeline: `31.65 mAP`
- optimized PAT pipeline: `20.15 mAP`
- optimized PHA pipeline: `22.88 mAP`

On Kaggle's `0-1` display scale, the paper's `31.65 mAP` is approximately `0.3165`.

The current experiment achieved:

- internal train-domain validation peak: `64.1% mAP`, `81.1% Rank-1`
- reported Kaggle score: `0.09831`

These numbers are not directly comparable. The paper's `31.65` is a true leaderboard-facing result on the public challenge split, while the current internal `64.1%` comes from a local validation setup that evaluates on the training-domain split rather than a truly held-out challenge-like split.

So the central finding is:

**the current implementation is a useful partial adaptation of the paper, but it is still missing several of the paper's most important ingredients and its local validation protocol is not realistic enough.**

---

## 2. Challenge Context

Urban2026 is an urban object re-identification challenge. For each query image, the model must rank the top matching gallery images of the same urban object identity.

The 2026 competition includes 4 classes:

- containers
- rubbish bins
- crosswalks
- traffic signs

The main difficulty comes from cross-camera domain shift:

- training images come from `c001`, `c002`, `c003`
- query images come from `c004`
- `c004` captures the route in the reverse direction

According to `Dataset.txt`, Urban2026 contains:

| Split | Images | Identities | Cameras |
|---|---:|---:|---|
| Train | 11,175 | 1,088 | c001, c002, c003 |
| Query | 928 | 215 | c004 |
| Gallery | 2,844 | 363 | c001, c002, c003 |
| Total | 14,947 | 1,453 | 4 cameras |

The challenge metric is mAP on Kaggle, reported on a `0-1` scale.

---

## 3. What The Paper Actually Did

### 3.1 Paper scope and datasets

The paper did **not** run the exact same setup as the current Urban2026-only experiment.

The paper used two urban object datasets:

- **Campus Dataset / UrbAM-ReID**
- **City Dataset / Urban Elements ReID**

Important details from the paper:

- both datasets contain **3 classes**, not 4
- the classes are containers, rubbish bins, and crosswalks
- traffic signs were **not** part of the paper's experiments
- the Campus dataset was used for training
- the City dataset was used for benchmarking and generalization evaluation
- results were reported on the public Kaggle split, which the paper states is about **47% of the full test set**

This matters because the current Urban2026 competition setup differs from the paper in two important ways:

1. Urban2026 includes an additional class, `traffic signs`
2. Urban2026 already provides class labels directly in CSV files

### 3.2 Paper baseline

The paper's baseline was:

- framework: **Bag of Tricks (BoT)**
- backbone: **ResNet-50**
- pretrained on ImageNet
- loss: **triplet-center loss**
- training length: **100 epochs**
- default hyperparameters from the referenced baseline repository

The paper's baseline score was:

- **19.90 mAP**

### 3.3 Paper data-centric enhancements

The paper explored several data-centric interventions.

#### A. Class-based filtering

The paper tested two class-based strategies:

- training a separate model per class
- using a classifier after training to rerank the gallery by class

For the classifier-based reranking pipeline, the paper did substantially more work than the current experiment:

- it trained a `ResNet18` classifier
- it first validated the classifier on the Campus dataset using an `80/20` split
- it then manually labeled `165` images in the City dataset
- it expanded those labels through pseudo-labeling using a confidence threshold of `0.9`
- it repeated this for `5` iterations and manually cleaned remaining misclassifications

This is very different from the current Urban2026 setup, where class labels are already provided in `query_classes.csv` and `test_classes.csv`.

Reported results:

| Configuration | mAP |
|---|---:|
| Baseline | 19.90 |
| Classifying before training | 19.25 |
| Reshaping per class size | 15.92 |
| Classifying after training | 20.32 |

Interpretation:

- splitting training by class **hurt**
- per-class image reshaping **hurt badly**
- reranking after training using class predictions gave a **small positive gain** of `+0.42 mAP`

This is an important correction to any future plan:

- **class-aware reranking is worth trying**
- **separate model per class should not be prioritized**
- **class-specific resizing should not be prioritized**

#### B. Data augmentation

The paper's first augmentation round included:

- perspective distortion
- brightness and contrast changes
- small random rotations
- zoom-in crops

These were applied to the original dataset.

The paper also states that each image was randomly assigned:

- no augmentation with probability `0.15`
- one augmentation with probability `0.45`
- two augmentations with probability `0.40`

The paper later added a second augmentation round for the final combined dataset, including:

- color jitter
- patch insertion to simulate occlusion

The paper explicitly says the second stage was only applied at the final combined-data step.

#### C. Super-resolution

The paper used **Real-ESRGAN** to upscale images when the original image was smaller than the target resolution.

Important conclusion from the paper:

- super-resolution **alone did not help**
- the paper attributes this to overly smooth images produced by the SR pipeline
- for rubbish bins in particular, the paper explicitly doubled resolution because they were especially low-resolution

#### D. Style transfer

To fix the smoothness introduced by SR, the paper created a hybrid dataset using:

- a **U-Net**
- trained with **VGG perceptual loss**
- to transfer style from super-resolved images back toward the original image domain

This was one of the key paper contributions. The paper states that style transfer preserved edges and textures better than SR alone.

#### E. Dataset combination results

The paper's ablation study shows:

- baseline: `19.90`
- best dataset-only combination: `24.15`
- gain over baseline: `+4.25 mAP`

The paper also reports:

- SR-only did not improve the baseline
- style-transferred data outperformed the baseline
- combining City and Campus data did **not** consistently help in the way some other ReID works had suggested

The paper also explored class-specific target sizes before finding that this was not a good direction overall:

- crosswalks: `640 x 96`
- containers: `128 x 60`
- rubbish bins: `128 x 192`

### 3.4 Paper model-centric enhancements

The paper also tested several model-side changes.

#### A. Quadruplet loss

The paper tested variants of quadruplet loss and found that it did **not** improve performance. The paper attributes this to likely overfitting.

This means quadruplet loss is **not** a recommended next step.

#### B. Backbone evaluation

The paper tested several CNN backbones and reported:

| Backbone | mAP |
|---|---:|
| ResNet-50 | 19.90 |
| SE-ResNet-50 | 25.26 |
| SE-ResNeXt-50 | 15.81 |
| ResNet-50-IBN | 16.92 |
| ResNet-101 | 16.98 |
| SE-Net-154 | 19.07 |

Key conclusion:

- **SE-ResNet-50 was clearly the best backbone**
- it improved over ResNet-50 by `+5.36 mAP`
- deeper or more complex backbones did not help

#### C. Ensembling and QMV

The paper tested multiple ensemble strategies and query majority voting.

Reported results:

| Configuration | mAP |
|---|---:|
| Baseline | 19.90 |
| Baseline + Enhancements | 28.11 |
| Naive Ensemble | 18.63 |
| Enhancements Ensemble (Same dataset) | 26.92 |
| Enhancements Ensemble (Same backbone) | 24.00 |
| Top-performing model + QMV | 28.18 |
| Top-performing model after tuning | 31.57 |
| Top-performing model after tuning + QMV | 31.65 |

Important takeaways:

- naive ensembling **hurt badly**
- not all ensembles are useful
- QMV helped, but its gain was **small compared to the backbone and dataset improvements**
- the best final gain came from **backbone choice + dataset enhancements + tuning + QMV**

#### D. Comparison against PAT and PHA

The paper also compared BoT against PAT and PHA:

| Method | Baseline mAP | Optimized mAP |
|---|---:|---:|
| BoT | 19.90 | 31.65 |
| PAT | 18.68 | 20.15 |
| PHA | 13.42 | 22.88 |

Paper conclusion:

- BoT remained the strongest overall pipeline
- PAT improved only modestly even after optimization
- CNNs were more reliable than transformers in this limited-data setting

---

## 4. What Was Implemented In The Current Codebase

The current project reused the `Part-Aware-Transformer` repository and extended it so a BoT-style `SE-ResNet-50` path could be trained and used for submission generation.

Implemented changes:

- `SE-ResNet-50` backbone support through `timm`
- BoT-style augmentation support
- improved normalization handling
- CNN-safe training path without PAT-only patch loss
- query-name-aligned submission writing
- test-time horizontal flip feature extraction
- optional re-ranking
- optional class-aware filtering
- dedicated train and test config files

Main files changed:

- `Part-Aware-Transformer/model/make_model.py`
- `Part-Aware-Transformer/data/transforms/build.py`
- `Part-Aware-Transformer/train.py`
- `Part-Aware-Transformer/update.py`
- `Part-Aware-Transformer/config/UrbanElementsReID_bot_train.yml`
- `Part-Aware-Transformer/config/UrbanElementsReID_bot_test.yml`

---

## 5. Paper Vs Current Experiment

The current experiment matches the paper only partially.

| Component | Paper | Current experiment | Status |
|---|---|---|---|
| BoT-style CNN direction | Yes | Yes | Matched in spirit |
| Best backbone `SE-ResNet-50` | Yes | Yes | Matched |
| 85-epoch tuned final model | Yes | Yes | Partially matched |
| Class-aware reranking | Yes, via trained classifier | Yes, via provided class CSVs | Partially matched |
| Perspective / brightness / rotation / zoom augments | Yes | Yes | Mostly matched |
| Second-stage augmentation on SR/ST data | Yes | No | Missing |
| Super-resolution with Real-ESRGAN | Yes | No | Missing |
| Style transfer with U-Net + VGG loss | Yes | No | Missing |
| Query Majority Voting | Yes | No | Missing |
| Ensemble experiments | Yes | No | Missing |
| Exact triplet-center loss baseline | Yes | No | Missing |
| Evaluation on public challenge split | Yes | No | Missing locally |
| Use of external Campus dataset | Yes | No | Different setup |
| 4-class Urban2026 with traffic signs | No | Yes | Different setup |

This table is the most important comparison in the report.

It shows that the current experiment implemented the **headline direction** of the paper, but not the **full recipe** that produced `31.65 mAP`.

---

## 6. Current Experimental Setup

### 6.1 Training configuration

From `UrbanElementsReID_bot_train.yml`:

- model: `SE-ResNet-50`
- optimizer: `Adam`
- max epochs: `85`
- base learning rate: `0.00035`
- batch size: `48`
- instances per identity: `4`
- input size: `256 x 128`
- checkpoint period: every `5` epochs

### 6.2 Training augmentations actually used

Enabled in the current training config:

- horizontal flip
- padding and random crop
- color jitter
- random resized crop
- random rotation
- random perspective distortion
- random erasing

### 6.3 Test configuration actually used

From `UrbanElementsReID_bot_test.yml`:

- re-ranking: enabled
- re-ranking parameters: `[20, 6, 0.3]`
- class filter: enabled
- class mismatch penalty: `1.0`
- checkpoint: `./models/bot_se_resnet50/se_resnet50_85.pth`

### 6.4 Important difference from the paper

The current code does **not** reproduce the paper's training recipe exactly.

The most important mismatches are:

- no triplet-center loss baseline
- no SR data generation
- no style-transfer data generation
- no QMV
- no ensemble stage
- no Campus plus City cross-dataset setup
- no paper-style public-split validation

So this run should be described as:

**a partial BoT + SE-ResNet-50 adaptation inspired by the paper, not a faithful end-to-end reproduction of the paper.**

---

## 7. Results Of The Current Experiment

### 7.1 Internal training-domain results

Selected checkpoints from `Part-Aware-Transformer/models/bot_se_resnet50/train_log.txt`:

| Epoch | Internal mAP | Rank-1 |
|---|---:|---:|
| 1 | 13.4% | 27.4% |
| 5 | 30.4% | 49.2% |
| 10 | 39.7% | 60.0% |
| 20 | 45.0% | 65.4% |
| 30 | 48.6% | 68.7% |
| 40 | 50.0% | 69.8% |
| 50 | 55.5% | 75.1% |
| 60 | 58.7% | 77.4% |
| 70 | 62.2% | 79.4% |
| 75 | 63.4% | 80.6% |
| 80 | 64.1% | 81.1% |
| 85 | 64.0% | 80.9% |

The training run converged well and the best internal epoch selected by the trainer was **epoch 80**.

### 7.2 Saved checkpoint behavior

The training code saves checkpoints every 5 epochs, tracks the best internal mAP, reloads the best checkpoint, deletes intermediate `.pth` files, and writes the selected best model under the final epoch filename.

So the file:

- `Part-Aware-Transformer/models/bot_se_resnet50/se_resnet50_85.pth`

is effectively the promoted best model from the run, not necessarily the raw epoch-85 weights.

### 7.3 Kaggle result

The reported Kaggle score from submission was:

- `0.09831`

This is currently the only real leaderboard-facing number available for this experiment.

---

## 8. Why The Current Internal Score And The Paper's Score Are Not Comparable

At first glance:

- paper final score: `31.65 mAP`
- current internal score: `64.1% mAP`

This can look as if the current experiment is much better than the paper, but that would be the wrong interpretation.

These numbers measure different things.

### 8.1 The paper's score is a leaderboard-facing score

The paper's `31.65` is a result on the public challenge split, which reflects actual unseen evaluation data. On Kaggle's `0-1` display scale, that is about `0.3165`.

### 8.2 The current internal score is train-domain self-evaluation

The current training dataset loader uses `train.csv` and `image_train` for:

- training
- local query set
- local gallery set

That means internal validation is effectively happening on the same dataset domain used for training. This makes the internal mAP optimistic and not directly useful for estimating Kaggle performance.

### 8.3 The only fair direct comparison is Kaggle-facing score

So the real practical comparison is:

- paper public result: `31.65 mAP`
- current reported Kaggle result: `0.09831`

This makes it clear that the current experiment is still far from the paper's final performance.

---

## 9. Why The Current Kaggle Result Is Still Much Worse Than The Paper

After comparing the code and the paper, the likely reasons are:

### 9.1 The current experiment is missing the paper's strongest data pipeline

The paper's best results relied heavily on:

- super-resolution
- style transfer
- staged augmentation

The current experiment only implemented the augmentation side, not the SR and style-transfer pipeline that the paper says was crucial.

### 9.2 The current experiment does not use the paper's final reranking strategy

The paper's final best result used:

- tuning to 85 epochs
- QMV

The current experiment uses:

- re-ranking
- class-aware penalty

but does **not** implement QMV.

### 9.3 The current experiment does not reproduce the paper's exact BoT baseline

The paper baseline used triplet-center loss and the exact BoT recipe. The current experiment uses a BoT-like CNN path inside another repository, but it is not the exact same training formulation.

### 9.4 The local validation protocol is misleading

The model selection signal in the current training loop is based on train-domain validation, not a proper held-out challenge-like split.

This can easily cause:

- optimistic checkpoint selection
- false confidence in generalization
- wrong decisions about which inference settings are best

### 9.5 The competition setting itself is not identical to the paper

Urban2026 differs from the paper setting because:

- it includes `traffic signs`
- it already provides class labels
- the data distribution may differ

So not every paper component transfers one-to-one.

---

## 10. What The Paper Suggests We Should Do Next

The paper comparison gives a very clear priority list.

### 10.1 Highest-priority items

1. Build a proper held-out validation split from `train.csv`

Reason:

- this is required before trustworthy tuning is possible

2. Reproduce the paper's best-performing data-centric pipeline more faithfully

Reason:

- the paper's biggest gains came from data-level enhancements and backbone choice

3. Keep `SE-ResNet-50` as the primary backbone

Reason:

- the paper strongly supports this choice
- our current direction is correct here

### 10.2 Medium-priority items

4. Implement super-resolution

Recommendation:

- use it only as part of a broader pipeline, not as a standalone final fix

5. Implement style transfer after SR

Recommendation:

- this is one of the biggest missing paper components

6. Add QMV

Recommendation:

- useful, but lower impact than validation and data pipeline improvements

### 10.3 Lower-priority or de-prioritized items

The paper suggests these should **not** be early priorities:

- separate model per class
- class-specific resizing
- quadruplet loss
- naive ensembling
- deeper backbones just because they are larger

This is useful because it prevents wasted effort.

---

## 11. Recommended Next-Step Plan For This Project

Based on the paper comparison and the current codebase, the most practical roadmap is:

### Stage 1: Fix evaluation first

1. Create a held-out validation split by identity from `train.csv`
2. Build a local query/gallery evaluation protocol on that split
3. Compare checkpoints and inference settings on that split

### Stage 2: Improve the current submission pipeline

4. Run inference ablations for:
- re-ranking on or off
- class filter on or off
- class mismatch penalty strength

5. Submit the best inference variant to Kaggle

### Stage 3: Reproduce the strongest missing paper components

6. Implement SR preprocessing
7. Implement style-transfer preprocessing
8. Retrain `SE-ResNet-50` on the enriched dataset
9. Add QMV

### Stage 4: Only then consider advanced extras

10. Test carefully chosen ensembles
11. Compare against PAT and PHA only after validation is realistic

---

## 12. Final Comparison Summary

After comparing the paper with the current experiment, the fairest summary is:

- the current project correctly identified the **right backbone direction**
- the current project correctly added **paper-aligned first-stage augmentations**
- the current project correctly used **85 epochs**
- the current project partially implemented **class-aware reranking**
- the current project did **not yet reproduce the paper's strongest data pipeline**
- the current project did **not yet implement QMV**
- the current project used a **local validation protocol that is too optimistic**

Therefore:

**the current experiment should be treated as an intermediate BoT-inspired baseline, not as a full reproduction of the paper's 31.65 mAP pipeline.**

That is actually useful, because it tells us exactly what the next work should focus on.

---

## 13. Final Conclusion

The paper comparison changes the interpretation of the current results in an important way.

Before comparing against the paper, the current run could be described as a strong `SE-ResNet-50` experiment with good internal learning behavior but disappointing Kaggle performance. After comparing against the paper in detail, a more accurate conclusion is:

**the current run is only a partial implementation of the paper's method, and its poor Kaggle score is consistent with the fact that several of the paper's highest-impact components are still missing.**

The main lessons from the paper are:

- keep `SE-ResNet-50`
- prioritize data-centric improvements
- do not trust the current internal validation
- do not spend time first on per-class models, class-specific resizing, quadruplet loss, or naive ensembling

The best next move is:

**build a proper held-out validation split and then reproduce the paper's missing SR + style-transfer + QMV pipeline in a controlled way.**
