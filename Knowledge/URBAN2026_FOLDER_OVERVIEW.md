# Part-Aware-Transformer Folder Overview for Urban2026

This document explains how the `Part-Aware-Transformer` folder works in this workspace, with a focus on how it is used to train and infer on the `Urban2026` dataset.

The folder started as the official codebase for the ICCV 2023 paper "Part-Aware Transformer for Generalizable Person Re-identification", then it was adapted locally to the `Urban2026` competition format by adding dataset loaders, config files, and submission utilities.

## 1. What This Folder Is Doing

At a high level, this folder is a full training and inference pipeline for image re-identification:

1. Read the dataset metadata from CSV files inside `Urban2026/`.
2. Build PyTorch dataloaders for train, query, and gallery images.
3. Build the PAT model, which is a Vision Transformer with extra part tokens.
4. Train the model with ReID losses and PAT-specific patch-center learning.
5. Extract features for query and gallery images.
6. Rank gallery images for each query.
7. Optionally re-rank the distances and export a submission CSV.

In this workspace, the intended dataset root is:

```text
Competition/
  Urban2026/
    train.csv
    query.csv
    test.csv
    image_train/
    image_query/
    image_test/
```

## 2. Main Entry Points

These are the top-level files you will usually run:

- `train.py`: trains the model.
- `test.py`: loads a trained checkpoint and runs evaluation/inference.
- `update.py`: runs inference, extracts query/gallery features, applies re-ranking, and writes the final ranked output for submission.
- `evaluate_csv.py`: evaluates a prediction CSV against ground-truth CSV files.
- `run.sh`: a tiny wrapper that runs `train.py`.

Typical commands in this repo are:

```bash
python train.py --config_file config/UrbanElementsReID_train.yml
python test.py --config_file config/UrbanElementsReID_test.yml
python update.py --config_file config/UrbanElementsReID_test.yml --track track
python evaluate_csv.py --track track_submission.csv --path ../Urban2026/
```

## 3. Folder Structure

The most important subfolders are:

- `config/`: configuration files and default settings.
- `data/`: dataset registration, dataloaders, samplers, transforms, and image reading.
- `model/`: model factory and backbone implementations.
- `loss/`: loss functions, patch memory, and PAT-specific patch-center loss.
- `processor/`: training and inference loops.
- `solver/`: optimizer and learning-rate scheduler.
- `utils/`: logging, metrics, file utilities, distributed helpers, and re-ranking.
- `visualization/`: attention rollout tools for inspecting PAT attention behavior.
- `models/`: saved checkpoints and logs in this workspace.
- `tb_log/`: TensorBoard outputs.

## 4. Urban2026-Specific Files

The original PAT repo was adapted to `Urban2026` mainly through these files:

- `config/UrbanElementsReID_train.yml`
- `config/UrbanElementsReID_test.yml`
- `data/datasets/UrbanElementsReID.py`
- `data/datasets/UrbanElementsReID_test.py`
- `update.py`
- `evaluate_csv.py`

These are the files that connect the generic PAT code to your competition dataset.

## 5. Config Flow

All runs start from a YAML config file and then merge with defaults from `config/defaults.py`.

### 5.1 Training Config

`config/UrbanElementsReID_train.yml` is the training setup for this workspace.

Important values in that file:

- `MODEL.NAME: 'part_attention_vit'`
  This selects the PAT architecture instead of plain ViT or ResNet.
- `MODEL.PRETRAIN_PATH`
  This points to the local folder that stores ImageNet pretrained ViT weights.
- `DATASETS.TRAIN: ('UrbanElementsReID',)`
  This tells the loader to use the Urban2026 training adapter.
- `DATASETS.TEST: ('UrbanElementsReID',)`
  This uses the train split again as a validation-style retrieval benchmark.
- `DATASETS.ROOT_DIR`
  This points to the local `Urban2026/` folder.
- `LOG_ROOT: 'models/'`
  Checkpoints and logs are written under `Part-Aware-Transformer/models/`.

### 5.2 Test Config

`config/UrbanElementsReID_test.yml` is the challenge-time inference setup.

Important differences:

- `DATASETS.TEST: ('UrbanElementsReID_test',)`
  This switches to the query/test split adapter.
- `TEST.WEIGHT`
  This points to the trained checkpoint to load.

## 6. How Training Works End to End

## 6.1 `train.py`

`train.py` is the main training script.

It does the following:

1. Parse `--config_file`.
2. Merge the YAML with `config/defaults.py`.
3. Set random seeds.
4. Create the logging directory.
5. Build the training loader.
6. Build the validation loader.
7. Build the model.
8. Build the loss functions.
9. Build the optimizer and scheduler.
10. Start the correct training loop.

Important behavior:

- If `MODEL.NAME` is `part_attention_vit`, `train.py` uses `processor/part_attention_vit_processor.py`.
- Otherwise it falls back to `processor/ori_vit_processor_with_amp.py`.

## 6.2 Building the Train Loader

`data/build_DG_dataloader.py` contains `build_reid_train_loader`.

That function:

1. Creates training transforms.
2. Instantiates each dataset listed in `cfg.DATASETS.TRAIN`.
3. Converts the dataset samples into a unified format.
4. Adds domain metadata to each sample.
5. Wraps everything in `CommDataset`.
6. Applies an identity-based sampler.

The data sample returned by `CommDataset` has the form:

```python
{
    "images": image_tensor,
    "targets": pid,
    "camid": camid,
    "img_path": img_path,
    "others": others
}
```

This dictionary format is what the training and inference processors expect.

## 6.3 How `UrbanElementsReID.py` Reads Urban2026

`data/datasets/UrbanElementsReID.py` is the training dataset adapter.

It expects:

- `train.csv`
- `image_train/`

It reads `train.csv`, extracts:

- camera id
- image name
- object id

Then it creates tuples like:

```python
(image_path, pid, camid)
```

During training:

- the original object IDs are remapped to contiguous labels when `relabel=True`
- `image_train/` is used for train, query, and gallery inside this adapter

That means the "validation" used during training is not the hidden competition test split. It is a retrieval evaluation on the training data itself.

## 6.4 Sampler and Batch Construction

The training config uses:

- `DATALOADER.SAMPLER: 'softmax_triplet'`
- `DATALOADER.NUM_INSTANCE: 4`

This means each training batch is identity-balanced:

- choose multiple identities
- sample 4 images per identity

This is important because triplet loss works much better when each batch contains multiple examples of the same identity.

The samplers live in:

- `data/samplers/triplet_sampler.py`
- `data/samplers/data_sampler.py`

In this Urban2026 setup, `RandomIdentitySampler` is the main one actually used.

## 6.5 Image Transforms

`data/transforms/build.py` builds the training and test transforms.

Training transforms can include:

- resize
- random horizontal flip
- padding + random crop
- Local Grayscale Transformation (`LGT`)
- optional color jitter
- optional random erasing
- tensor conversion and normalization

In your training config:

- images are resized to `256 x 128`
- `LGT` is enabled
- normalization uses mean/std of `[0.5, 0.5, 0.5]`

## 6.6 Building the Model

`model/make_model.py` is the model factory.

It supports:

- ResNet backbones
- plain ViT (`vit`)
- Part-Aware Transformer (`part_attention_vit`)

For Urban2026 training, the selected model is:

```text
MODEL.NAME = part_attention_vit
MODEL.TRANSFORMER_TYPE = vit_base_patch16_224_TransReID
```

So the factory builds `build_part_attention_vit`.

## 6.7 What Makes PAT Different

The main PAT implementation lives in `model/backbones/vit_pytorch.py`.

Compared with a plain ViT, PAT adds:

- one `cls_token`
- three extra learnable part tokens
- a masked part-attention mechanism

These extra part tokens are intended to focus on different body/object regions rather than only using a single global token.

The core PAT backbone class is `part_Attention_ViT`.

Its important steps are:

1. Convert the image into overlapping patch embeddings.
2. Prepend:
   - one class token
   - three part tokens
3. Add positional embeddings.
4. Pass tokens through transformer blocks that use `part_Attention_Block`.
5. Return the tokens from every layer.

The model wrapper in `model/make_model.py` then:

- takes the last-layer class token as the global feature
- keeps the per-layer part tokens for the PAT-specific losses
- applies BN neck and a classifier head for identity prediction

During training, the PAT model returns:

```python
cls_score, layerwise_cls_tokens, layerwise_part_tokens
```

During inference, it returns only the final feature embedding.

## 6.8 Pretrained Weights

PAT is initialized from ImageNet-pretrained ViT weights.

The weight file names are mapped in `model/make_model.py`, for example:

- `jx_vit_base_p16_224-80ecf9dd.pth`

This pretrained file is expected under the folder pointed to by:

```text
MODEL.PRETRAIN_PATH
```

The PAT code also resizes positional embeddings when the pretrained model shape does not exactly match the current image geometry.

## 6.9 Losses Used in Training

`loss/build_loss.py` builds the standard ReID losses:

- classification loss
- triplet loss

For PAT training, `train.py` also creates:

- `PatchMemory`
- `Pedal`

These live in:

- `loss/smooth.py`
- `loss/myloss.py`

### Standard ReID Part

The standard part is:

- ID loss: predicts the identity class
- triplet loss: pushes same-ID features closer and different-ID features apart

### PAT-Specific Part

PAT also keeps a memory of patch-level features across images.

`PatchMemory` stores feature centers for local parts, keyed by image path.

`Pedal` computes a patch-center loss that encourages the model to learn local similarities between identities. This is the PAT-specific part that tries to reduce overfitting to global identity shortcuts.

## 6.10 Training Loop

The PAT training loop is implemented in `processor/part_attention_vit_processor.py`.

The main flow is:

1. Move model to GPU.
2. Initialize patch centers by forwarding the whole training set once.
3. For each epoch:
   - iterate over batches
   - compute model outputs
   - update patch memory
   - compute ReID loss
   - compute patch-center loss
   - sum them into total loss
   - backpropagate with AMP
4. Periodically run evaluation.
5. Periodically save checkpoints.
6. Reload the best checkpoint.
7. Run final inference.
8. Delete older checkpoints and keep the final one.

The optimizer is created by `solver/make_optimizer.py`, and the scheduler is created by `solver/scheduler_factory.py`.

## 7. How Validation Works During Training

The validation loader is built by `build_reid_test_loader`.

For `UrbanElementsReID`, the query and gallery sets both come from `image_train/`, because the dataset adapter uses the training split for both.

So the reported `mAP`, `Rank-1`, `Rank-5`, and `Rank-10` during training are a training-split retrieval signal, not the final competition score.

This is useful for checking whether the model learns anything, but it is not a direct measure of hidden-test performance.

## 8. How Test-Time Inference Works

## 8.1 `UrbanElementsReID_test.py`

`data/datasets/UrbanElementsReID_test.py` is the challenge-time dataset adapter.

It expects:

- `train.csv`
- `query.csv`
- `test.csv`
- `image_train/`
- `image_query/`
- `image_test/`

It builds:

- `train`: from `train.csv`
- `query`: from `query.csv`
- `gallery`: from `test.csv`

For query and gallery items, it assigns dummy labels of `-1`, because the real IDs are unknown during challenge inference.

## 8.2 `test.py`

`test.py`:

1. loads the config
2. builds the model
3. loads `TEST.WEIGHT`
4. builds the test loader
5. extracts embeddings
6. computes retrieval metrics through `utils/metrics.py`

Important caveat:

Because `UrbanElementsReID_test.py` assigns `-1` to query and gallery identities, the printed test-time CMC/mAP values are not meaningful competition metrics. They are just the result of running the generic evaluator on dummy labels.

## 8.3 `update.py`

`update.py` is the more important inference script for the competition.

It does this:

1. load the trained model
2. build the query+gallery dataloader
3. extract query features `qf`
4. extract gallery features `gf`
5. save them to `qf.npy` and `gf.npy`
6. compute query-gallery, query-query, and gallery-gallery similarities
7. apply re-ranking
8. sort gallery indices for every query
9. write a text track file
10. write a submission CSV with columns:
    - `imageName`
    - `Corresponding Indexes`

This is the script that actually turns model embeddings into a challenge submission.

## 9. Evaluation Utilities

`evaluate_csv.py` is an offline evaluator for a generated submission CSV.

It reads:

- `query.csv`
- `test.csv`
- the prediction CSV

Then it computes:

- mAP
- Rank-1
- Rank-5
- Rank-10
- Rank-20

This is the correct place to measure a generated ranking file against known labels, not the generic `test.py` metrics on the dummy-label test adapter.

## 10. Utilities and Support Code

## 10.1 Metrics

`utils/metrics.py` contains:

- Euclidean distance
- cosine similarity
- Market1501-style CMC/mAP evaluation
- the `R1_mAP_eval` accumulator

The inference processors use this module to collect features and compute retrieval scores.

## 10.2 Re-ranking

`utils/re_ranking.py` implements k-reciprocal re-ranking.

This is used by `update.py` after feature extraction to improve ranking quality before writing the final submission.

## 10.3 Logging

`utils/logger.py` writes:

- `train_log.txt`
- `test_log.txt`

under the folder defined by:

- `LOG_ROOT`
- `LOG_NAME`

In this workspace, that ends up under:

```text
Part-Aware-Transformer/models/model/
```

## 11. What the Saved Files Mean

During and after training, this folder may contain:

- `models/model/part_attention_vit_60.pth`
  Final PAT checkpoint.
- `models/model/train_log.txt`
  Training log.
- `models/model/test_log.txt`
  Test or inference log.
- `qf.npy`
  Query feature matrix.
- `gf.npy`
  Gallery feature matrix.
- `tb_log/...`
  TensorBoard summaries.

## 12. Real Urban2026 Workflow

If you want the practical flow for this workspace, it is:

### Step 1: Prepare data

Put the `Urban2026` folder in the location referenced by the config files.

### Step 2: Prepare pretrained ViT weights

Put the pretrained ViT checkpoint in:

```text
Competition/pretrained_models/
```

### Step 3: Train

Run:

```bash
python train.py --config_file config/UrbanElementsReID_train.yml
```

This writes logs and checkpoints under `models/model/`.

### Step 4: Load the trained checkpoint

Set `TEST.WEIGHT` in `config/UrbanElementsReID_test.yml` to the checkpoint you want to use.

### Step 5: Generate ranked results

Run:

```bash
python update.py --config_file config/UrbanElementsReID_test.yml --track track
```

This creates:

- a text ranking file named `track`
- a CSV file named `track_submission.csv`

### Step 6: Evaluate

Run:

```bash
python evaluate_csv.py --track track_submission.csv --path ../Urban2026/
```

## 13. Important Caveats in This Fork

These are worth knowing before trusting every metric printed by the folder:

### 13.1 Training validation uses the train split

`UrbanElementsReID.py` uses `image_train/` for train, query, and gallery. So the training-time validation is not the real challenge evaluation split.

### 13.2 Test-time generic metrics are misleading

`UrbanElementsReID_test.py` gives query and gallery images the dummy PID `-1`, so `test.py` prints metrics that do not reflect actual hidden-test correctness.

### 13.3 `update.py` is the real submission path

For the Urban2026 challenge, the important output is the ranked CSV from `update.py`, not the generic `test.py` evaluation numbers.

### 13.4 There are a couple of code rough edges

This fork has some signs of being adapted quickly:

- `loss/build_loss.py` still checks for `local_attention_vit` in one branch instead of `part_attention_vit`
- `solver/scheduler_factory.py` hardcodes `num_epochs = 120` instead of reading `cfg.SOLVER.MAX_EPOCHS`

These do not stop the folder from running, but they are worth remembering if you plan to clean up or extend the code.

## 14. Short Mental Model

If you want the simplest mental model of the whole folder:

- `config/` tells the code what dataset, model, and checkpoint to use.
- `data/` turns the `Urban2026` CSV files and image folders into PyTorch batches.
- `model/` builds PAT, a ViT with three extra part tokens.
- `loss/` combines standard ReID learning with patch-based local feature learning.
- `processor/` runs the actual train and inference loops.
- `update.py` converts embeddings into a ranked submission file.

So the full Urban2026 pipeline inside this folder is:

```text
Urban2026 CSV + images
-> dataset adapters
-> dataloader + transforms
-> PAT model
-> ReID loss + patch-center loss
-> trained checkpoint
-> query/gallery embeddings
-> re-ranking
-> submission CSV
```

## 15. Most Important Files to Read First

If you only want the shortest path through the codebase, read these in order:

1. `config/UrbanElementsReID_train.yml`
2. `train.py`
3. `data/build_DG_dataloader.py`
4. `data/datasets/UrbanElementsReID.py`
5. `model/make_model.py`
6. `model/backbones/vit_pytorch.py`
7. `processor/part_attention_vit_processor.py`
8. `update.py`
9. `evaluate_csv.py`

That sequence gives you the cleanest end-to-end picture of how this folder trains on `Urban2026` and then generates the final ranking file.
