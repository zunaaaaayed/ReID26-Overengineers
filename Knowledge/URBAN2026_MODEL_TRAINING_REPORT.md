# Urban2026 Model Training Report

This report summarizes the actual training and testing setup used in the `Part-Aware-Transformer` folder for `Urban2026`, based on:

- `URBAN2026_FOLDER_OVERVIEW.md`
- `models/model/train_log.txt`
- `models/model/test_log.txt`
- the current PAT code and config files

The goal of this document is to answer:

- which model was used
- what parameters and settings were used
- what numbers were seen during training and testing
- what the results suggest
- what can be improved next

## 1. Short Summary

The run in this folder used:

- model: `part_attention_vit`
- transformer backbone: `vit_base_patch16_224_TransReID`
- training dataset adapter: `UrbanElementsReID`
- test dataset adapter: `UrbanElementsReID_test`
- input size: `256 x 128`
- optimizer: `SGD`
- batch size: `64`
- instances per identity: `4`
- epochs requested: `60`

The training run improved steadily from epoch 1 to about epoch 50, then collapsed into `nan` loss from epoch 51 onward.

The best validation result recorded before collapse was:

- epoch 50
- mAP: `52.6%`
- Rank-1: `70.3%`
- Rank-5: `88.6%`
- Rank-10: `93.5%`

After training finished, the code reloaded the best checkpoint and evaluated it again. That final reloaded best-checkpoint evaluation was:

- mAP: `52.5%`
- Rank-1: `70.4%`
- Rank-5: `88.4%`
- Rank-10: `93.4%`

## 2. Which Model Was Actually Used

The repository supports multiple model families, but the logs show that this run used only one real training model:

- `part_attention_vit`

From the merged config shown in `train_log.txt`, the core model settings were:

- `MODEL.NAME: part_attention_vit`
- `MODEL.TRANSFORMER_TYPE: vit_base_patch16_224_TransReID`
- `MODEL.NECK: bnneck`
- `MODEL.METRIC_LOSS_TYPE: triplet`
- `MODEL.NO_MARGIN: True`
- `MODEL.IF_LABELSMOOTH: on`
- `MODEL.PC_LOSS: True`
- `MODEL.SOFT_LABEL: True`
- `MODEL.FREEZE_PATCH_EMBED: True`

This means the run used the PAT architecture, not the plain `vit` baseline and not a ResNet backbone.

## 3. Model Size

Two parameter counts appear in the logs:

- training model: `87.36M`
- final evaluation/inference model: `86.52M`

Why they differ:

- during training, the model includes a classifier sized for `1088` train IDs
- during final evaluation, the repo rebuilds the model with `num_class=0`
- that removes the effective classification head size from the parameter count

So the backbone is the same, but the final inference model is slightly smaller.

## 4. Dataset Sizes Used in This Run

## 4.1 Training Run Dataset Stats

From `train_log.txt`:

- training set:
  - identities: `1088`
  - images: `11175`
  - cameras: `3`

The training config also used `UrbanElementsReID` as its validation-style dataset, which means:

- query set:
  - identities: `1088`
  - images: `11175`
  - cameras: `3`
- gallery set:
  - identities: `1088`
  - images: `11175`
  - cameras: `3`

Important note:

This is not the hidden challenge test split. This is a retrieval evaluation built from the training data adapter.

## 4.2 Test Run Dataset Stats

From `test_log.txt`:

- dataset: `UrbanElementsReID_test`
- query images: `928`
- gallery images: `2844`
- query cameras: `1`
- gallery cameras: `3`

The test dataset loader assigns dummy IDs for query and gallery, so the generic CMC/mAP printed in `test_log.txt` is not a reliable competition score.

That is why the `100.0%` test metric in `test_log.txt` should not be interpreted as real performance.

## 5. Exact Training Configuration Used

These values come from the merged config printed in `train_log.txt`.

## 5.1 Data Settings

- train dataset: `UrbanElementsReID`
- validation dataset during training: `UrbanElementsReID`
- root directory:
  - `d:/ALL Uni Documents/UAM/Courses/Deep Learning for Visual Signal Processing I/Competition/Urban2026/`
- train image size: `[256, 128]`
- test image size: `[256, 128]`
- random horizontal flip: `True`
- flip probability: `0.5`
- padding: `10`
- random crop after pad: enabled through `DO_PAD: True`
- Local Grayscale Transformation:
  - enabled: `True`
  - probability: `0.5`
- color jitter: `False`
- AugMix: `False`
- AutoAugment: `False`
- random erasing: `False`
- random patch: `False`

## 5.2 Dataloader Settings

- sampler: `softmax_triplet`
- batch size: `64`
- instances per identity: `4`
- workers: `0`
- naive identity sampling: `True`
- drop last: `False`

In practice, this means each batch is organized around roughly:

- `64 / 4 = 16` identities per batch
- `4` images per identity

From the log:

- iterations per epoch: `159`

## 5.3 Model Settings

- model name: `part_attention_vit`
- backbone: `vit_base_patch16_224_TransReID`
- pretrained choice: `imagenet`
- pretrained weights folder:
  - `d:/ALL Uni Documents/UAM/Courses/Deep Learning for Visual Signal Processing I/Competition/pretrained_models`
- stride size: `[16, 16]`
- drop path: `0.1`
- dropout: `0.0`
- attention dropout: `0.0`
- bnneck: enabled
- patch embedding frozen: `True`

## 5.4 Loss Settings

- ID loss type: `softmax`
- ID loss weight: `1.0`
- triplet loss weight: `1.0`
- metric loss type: `triplet`
- no margin: `True`
  - this means soft triplet loss was used
- label smoothing: `on`

PAT-specific settings:

- `PC_LOSS: True`
- `PC_LR: 1.0`
- `PC_SCALE: 0.02`
- `SOFT_LABEL: True`
- `SOFT_LAMBDA: 0.5`
- `SOFT_WEIGHT: 0.5`
- `CLUSTER_K: 10`

## 5.5 Optimizer and Scheduler Settings

- optimizer: `SGD`
- base learning rate: `0.001`
- momentum: `0.9`
- weight decay: `0.0001`
- bias LR factor: `2`
- warmup epochs: `5`
- warmup factor: `0.01`
- warmup method: `linear`
- max epochs in config: `60`
- checkpoint period: `5`
- log period: `60`
- eval period: `1`

Important implementation detail from the code:

- `solver/scheduler_factory.py` hardcodes `num_epochs = 120`

So the cosine scheduler was built for `120` epochs even though the run only trained for `60`.

That means the schedule and the configured run length were not fully aligned.

## 6. Test / Inference Configuration Used

From `test_log.txt`, the test-time config used:

- model: `part_attention_vit`
- backbone: `vit_base_patch16_224_TransReID`
- dataset: `UrbanElementsReID_test`
- test batch size: `128`
- feature normalization: `True`
- neck feature: `before`
- test checkpoint:
  - `./models/model/part_attention_vit_60.pth`

The same pretrained model directory and dataset root were used as in training.

## 7. Training Curve Summary

The training curve had three clear phases.

## 7.1 Fast Early Improvement

The model improved quickly in the first 10 epochs.

Selected milestones:

| Epoch | mAP | Rank-1 |
|---|---:|---:|
| 1 | 21.0% | 40.4% |
| 5 | 34.2% | 56.1% |
| 10 | 38.9% | 59.2% |

This shows the training setup was learning properly early on.

## 7.2 Strong Mid-Training Growth

The model kept improving strongly through the middle of training.

Selected milestones:

| Epoch | mAP | Rank-1 |
|---|---:|---:|
| 15 | 42.6% | 62.2% |
| 20 | 45.4% | 64.0% |
| 25 | 47.8% | 66.0% |
| 30 | 50.1% | 67.8% |
| 35 | 51.8% | 68.9% |

This was the most productive part of training.

## 7.3 Plateau and Final Best Range

From roughly epoch 35 onward, gains became much smaller.

Selected milestones:

| Epoch | mAP | Rank-1 |
|---|---:|---:|
| 40 | 52.2% | 69.7% |
| 45 | 52.5% | 70.3% |
| 50 | 52.6% | 70.3% |

This suggests the run had mostly saturated by epochs 45 to 50.

## 8. Collapse After Epoch 50

The most important issue in the logs is that training becomes unstable after epoch 50.

From epoch 51 onward:

- `total_loss: nan`
- `reid_loss: nan`
- `pc_loss` becomes tiny or `0.000`
- accuracy collapses
- validation mAP drops from `52.6%` to `0.2%`

The first clear failure point in the logs is:

- epoch 51
- total loss becomes `nan`
- validation immediately collapses

This means:

- the final epoch-60 model is not trustworthy
- the useful model is the best checkpoint saved before the collapse
- in this run, that best checkpoint was around epoch 50

## 9. Best Checkpoint Actually Used

The log shows the best epochs being updated at:

- epoch 5
- epoch 10
- epoch 15
- epoch 20
- epoch 25
- epoch 30
- epoch 35
- epoch 40
- epoch 45
- epoch 50

After training completed, the script rebuilt the model, loaded the best checkpoint, and evaluated it again.

Final reloaded best-checkpoint result:

- mAP: `52.5%`
- Rank-1: `70.4%`
- Rank-5: `88.4%`
- Rank-10: `93.4%`

This is the most meaningful result in the training log.

## 10. What the Test Log Means

The `test_log.txt` run shows:

- query images: `928`
- gallery images: `2844`
- checkpoint: `./models/model/part_attention_vit_60.pth`

It then prints:

- mAP: `100.0%`
- Rank-1: `100.0%`
- Rank-5: `100.0%`
- Rank-10: `100.0%`

This should not be treated as real performance.

Why:

- `UrbanElementsReID_test.py` gives dummy PID values for query and gallery
- the generic evaluator is not meaningful on that test adapter
- for real competition output, the important script is `update.py`
- for offline scoring of the generated ranked CSV, the important script is `evaluate_csv.py`

So:

- `train_log.txt` gives the useful training-side retrieval signal
- `test_log.txt` confirms the inference path ran
- `test_log.txt` does not give a trustworthy challenge metric

## 11. What Was Good About This Run

Several positive signals appear in the logs:

- the model learned steadily for 50 epochs
- the PAT setup reached over `52%` mAP on the train-based validation protocol
- Rank-1 reached about `70%`
- the model clearly benefited from longer training up to around epoch 45 to 50
- the best checkpoint reload still gave strong results after the later collapse

This means the basic training pipeline is working.

## 12. What Was Problematic in This Run

The main issues are:

### 12.1 NaN instability

The largest problem is the `nan` collapse after epoch 50.

### 12.2 Validation is built from the training split

The run evaluates on `UrbanElementsReID`, which uses the training data as train/query/gallery.

That means:

- it is useful for monitoring
- but it is not a real held-out validation score
- it may overestimate real generalization

### 12.3 Test log metrics are misleading

The `100%` test result is not a real retrieval score for the challenge.

### 12.4 Scheduler mismatch

The training config says `60` epochs, but the scheduler code uses `120`.

That mismatch may hurt optimization quality.

## 13. Most Likely Ways To Improve Results

These suggestions are based on the logs, the config, and the current code.

## 13.1 Stop Earlier or Use Early Stopping

The simplest improvement is:

- stop at epoch 45 to 50
- do not keep training past the point where the run becomes unstable

Because the run already reaches its best result at epoch 50, extending to 60 did not help.

This is the most directly supported improvement from the logs.

## 13.2 Fix the NaN Instability First

Since the run collapses after epoch 50, this is the first technical issue worth addressing.

Good experiments:

- lower `SOLVER.BASE_LR` from `0.001` to `0.0005`
- try `AdamW` instead of `SGD`
- add gradient clipping in the training loop
- temporarily disable AMP to see whether mixed precision is part of the instability
- reduce `MODEL.PC_LR` from `1.0` to something smaller like `0.5` or `0.2`
- reduce `MODEL.SOFT_WEIGHT` from `0.5`
- test with `MODEL.SOFT_LABEL = False`

These are good candidates because the failure appears suddenly, not gradually.

## 13.3 Align the Scheduler With the Training Length

Right now:

- config says `MAX_EPOCHS = 60`
- scheduler code is built for `120`

You should try one of these:

- train for `120` epochs
- or modify the scheduler to use `cfg.SOLVER.MAX_EPOCHS`

This is a strong improvement candidate because the current run uses only half of the scheduler horizon.

## 13.4 Build a Real Validation Split

Right now the validation protocol is train-based.

A stronger setup would be:

- split `train.csv` into train and validation IDs
- or split by camera
- or hold out a subset of IDs from training

This will help you:

- trust the score more
- compare experiments more honestly
- detect overfitting sooner

## 13.5 Turn On More Generalization Augmentations

Many augmentations are currently off:

- random erasing: off
- color jitter: off
- AugMix: off
- AutoAugment: off
- random patch: off

Possible experiments:

- enable random erasing
- enable color jitter
- test AugMix or AutoAugment
- test random patch

Since this is a re-identification task under distribution shift, better augmentation may improve generalization.

## 13.6 Save Checkpoints More Aggressively

The current checkpoint period is `5`.

That is acceptable, but if a run is unstable, it can be safer to:

- save every epoch
- or at least save every 1 to 2 epochs once the score plateaus

This helps preserve more candidate checkpoints around the best region.

## 13.7 Compare PAT Against the Plain ViT Baseline

This run only logged PAT.

The repository also supports:

- `vit`

It would be useful to run the plain ViT baseline with the same Urban2026 setup and compare:

- convergence speed
- best mAP
- stability
- sensitivity to NaNs

That comparison will tell you whether PAT is actually helping on your dataset.

## 13.8 Use `update.py` and `evaluate_csv.py` for Final Ranking Quality

For actual competition-style output:

- use `update.py` to generate ranked results
- use `evaluate_csv.py` when labels are available

This is better than relying on the generic `test.py` result.

## 14. Suggested Improvement Order

If you want a practical order of work, this is the most sensible sequence:

1. Keep the current PAT model but stop at epoch 50.
2. Fix instability:
   - lower LR
   - consider gradient clipping
   - test without AMP
3. Fix the scheduler mismatch so it matches the real epoch count.
4. Build a proper validation split.
5. Turn on one augmentation at a time and compare.
6. Compare PAT against plain `vit`.
7. Use `update.py` plus `evaluate_csv.py` to judge final ranking quality.

## 15. Recommended Baseline To Keep

Based on the logs, the safest baseline from this training run is:

- model: `part_attention_vit`
- backbone: `vit_base_patch16_224_TransReID`
- checkpoint region: epoch `45` to `50`
- best logged epoch: `50`

If you keep only one checkpoint from this run for further use, it should be the best checkpoint selected before the `nan` collapse, not the raw final-epoch training state after epoch 60.

## 16. Final Takeaway

This training run proves that the PAT pipeline works on `Urban2026`, and it reaches a strong train-side retrieval result around:

- mAP: `52.5%` to `52.6%`
- Rank-1: `70.3%` to `70.4%`

The main next step is not changing the whole architecture immediately.

The main next step is making the run stable:

- stop before epoch 51
- remove the `nan` collapse
- align the scheduler
- evaluate with a better validation protocol

That combination is the most likely path to a better and more trustworthy result.
