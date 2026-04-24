# Execution Log — Urban2026 ReID Overengineers

Tracks what was done, when, and what results were observed.
Linked to: `strategy.md` — check that file for the full plan rationale.

---

## 2026-04-23 — Week 1: Fixes + Track A/B/C Setup

### What was done

#### Code fixes (all merged into main branch)

| File | Change | Why |
|---|---|---|
| `config/defaults.py` | Added `MODEL.CLS_LAYERS=1`, `SOLVER.GRAD_CLIP=1.0` | New config knobs needed by other changes |
| `processor/part_attention_vit_processor.py` | `clip_grad_norm_` now uses `cfg.SOLVER.GRAD_CLIP` (was hard-coded 10.0) | Tighter clipping at 1.0 stabilises training past ep 30 |
| `model/make_model.py` — `build_part_attention_vit` | Added `CLS_LAYERS` support: bottleneck + classifier sized to `768 × CLS_LAYERS`; inference returns concat of last N CLS tokens | Aymen's 0.13553 win uses this (N=2 → 1536-d descriptor) |
| `model/make_model.py` — `load_param` | Shape-mismatch tolerant loading; skips incompatible keys with warning | Allows re-loading old 768-d checkpoints into a 1536-d model without crash |
| `update.py` | Added `--cls_layers N` (bypass-bottleneck multi-layer extraction via `model.base()`), `--camera_aware` + `--same_cam_factor` flags | Free wins from existing checkpoint; Narmeen's camera-aware trick |
| `data/datasets/UrbanElementsReID.py` | Added `UrbanElementsReID_Val` class (90/10 ID split, seeded) | Proper local eval — existing train-split eval was 100% optimistic |
| `data/datasets/__init__.py` | Registered `UrbanElementsReID_Val`, `UrbanElementsReID_UAM` | Makes new datasets discoverable |

#### New files created

| File | Purpose |
|---|---|
| `config/UrbanElementsReID_v2_train.yml` | Track A: PAT-ViT-B + CLS_LAYERS=2 + LR=0.00035 + GRAD_CLIP=1.0 + no-LGT + REA + CJ + RPT |
| `config/UrbanElementsReID_v2_test.yml` | Track A inference config |
| `config/UrbanElementsReID_vitl_train.yml` | Track B: PAT-ViT-L + CLS_LAYERS=2, batch=32 |
| `config/UrbanElementsReID_vitl_test.yml` | Track B inference config |
| `config/UrbanElementsReID_uam_train.yml` | Track C: merged UAM+URVAM+Urban2026 |
| `data/datasets/UrbanElementsReID_UAM.py` | Track C data adapter (gracefully skips missing UAM image dirs) |

#### Sanity checks passed
- Config parses: `CLS_LAYERS=2`, `GRAD_CLIP=1.0`, `BASE_LR=0.00035`, `LGT=False`, `REA=True`, `CJ=True`, `RPT=True`
- `UrbanElementsReID_Val` split: 980 train IDs / 10,176 imgs · 108 val IDs / 999 imgs
- `build_part_attention_vit(CLS_LAYERS=2)`: `in_planes=1536`, bottleneck `[1536]`, classifier `[980, 1536]`
- All 4 Urban dataset classes register correctly

---

### Next actions (in order)

#### Immediate — no GPU needed
- [x] **Re-infer existing checkpoint** with `--cls_layers 2 --camera_aware`:
  ```bash
  cd Part-Aware-Transformer
  conda activate ReID26
  python update.py \
    --config_file config/UrbanElementsReID_test.yml \
    --track ../track/track_cls2_camaware \
    --cls_layers 2 \
    --camera_aware \
    --k1 20 --k2 6 --lam 0.3
  ```
  Then evaluate locally:
  ```bash
  python evaluate_csv.py \
    --track ../track/track_cls2_camaware_submission.csv \
    --path /media/DiscoLocal/IPCV/OI/ReID26-Overengineers/Urban2026/
  ```
  → Submit to Kaggle if better than 0.11.

**Note on local evaluation:** `evaluate_csv.py` needs an ID-annotated `test.csv`. The competition split has no IDs (hidden). Use the `UrbanElementsReID_Val` training val split (reported during `train.py`) as the local signal. Kaggle submission is the only ground-truth score.

#### GPU training — Track A
- [x] **Train PAT-ViT-B v2** — RUNNING (started 2026-04-23 01:24, ~10 h on RTX 2080 Ti):
  ```bash
  cd Part-Aware-Transformer
  conda activate ReID26
  python train.py --config_file config/UrbanElementsReID_v2_train.yml
  ```
  Then infer:
  ```bash
  python update.py \
    --config_file config/UrbanElementsReID_v2_test.yml \
    --track ../track/track_v2 \
    --camera_aware --k1 20 --k2 6 --lam 0.3
  ```

#### GPU training — Track B
- [ ] **Download ViT-L pretrained weights** (required before training):
  ```bash
  cd /media/DiscoLocal/IPCV/OI/ReID26-Overengineers/pretrained_models
  wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth
  ```
- [ ] **Train PAT-ViT-L** (expected ~20 h, batch=32):
  ```bash
  python train.py --config_file config/UrbanElementsReID_vitl_train.yml
  ```

#### Data preparation — Track C
**Key finding (2026-04-23):** UAM/URVAM images are NOT separate downloads — they live inside Urban2026/image_train/.
- URVAM = exact duplicate of Urban2026 train (zero new info — skipped in adapter)
- UAM = 6387 images, 4443 with different (imageName, cameraID) pairs → adds 4443 new training pairs, 479 identities
- All images already on disk. `UrbanElementsReID_UAM` adapter verified: 15,618 imgs / 1,567 IDs
- Set env var and train:
  ```bash
  export UAM_ROOT=/media/DiscoLocal/IPCV/OI/ReID26-Overengineers
  python train.py --config_file config/UrbanElementsReID_uam_train.yml
  ```

---

## Results Tracker

| Run | Config | Kaggle mAP | Notes |
|---|---|---:|---|
| baseline | UrbanElementsReID_test.yml (existing ckpt) | 0.110 | PAT-ViT-B ep50, k=20/6/0.3, class filter |
| cls2+cam | UrbanElementsReID_test.yml --cls_layers 2 --camera_aware | **SUBMIT** | re-inference done → `submissions/track_cls2_camaware_submission.csv` |
| v2 | UrbanElementsReID_v2_test.yml | **SUBMIT** | DONE — ep50, train-mAP 96.7%, no NaN. CSV: submissions/track_v2_submission.csv |
| vitl | UrbanElementsReID_vitl_test.yml | TBD | ViT-L, batch=32, CLS_LAYERS=2 |
| uam | UrbanElementsReID_uam_train.yml | TBD | UAM+URVAM merged (requires image download) |
