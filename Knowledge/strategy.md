# Urban2026 ReID Strategy — Overengineers

**Target:** Kaggle mAP > 0.17 (current: 0.11)
**Gap:** +0.06 absolute mAP, i.e. ~55% relative improvement
**Date:** 2026-04-23

---

## 1. Situation Assessment

### 1.1 Competition
- **Urban2026** — 4-class urban-element ReID on Kaggle.
- **Classes:** trafficsignal (~67% train imgs), Crosswalk (14%), Container (11%), RubbishBins (8%).
- **Cameras:** train/gallery from c001/c002/c003, query from c004 (reverse-direction traversal → strong viewpoint domain shift).
- **Splits:** 11,175 train imgs / 1,088 IDs · 928 query imgs · 2,844 gallery imgs.
- **Metric:** mAP (rank-100 submission CSV).

### 1.2 Current Best Submission (us)
- Model: `part_attention_vit` (PAT-ViT-B/16) · input 256×128 · SGD · 60 ep.
- Train-side mAP ≈ 52.6% (unrealistic — evaluates on train split).
- Kaggle mAP ≈ 0.11.
- Known bugs in the PAT fork: `nan`-collapse after ep 50, scheduler hard-coded to 120 ep, `loss/build_loss.py` branch check wrong (`local_attention_vit` vs `part_attention_vit`).
- Re-ranking (k1=20, k2=6, λ=0.3) and class filter (container+rubbishbin merged as "bin_group") are already wired in `update.py`.

### 1.3 Competitor Intel (scores.txt)

| Rank | Person | Score | Winning Techniques |
|---|---|---:|---|
| 1 | Aymen | 0.13553 | ViT-Base, merge UrbAM, 30 ep, PAT loss, LR barely decayed, **disabled LGT**, color jitter + random patch, **concat CLS from last 2 ViT-L blocks (2×768 feat)** |
| 2 | Mohsin | 0.13361 | **ViT-Large** 256×128, 60 ep, k1=20/k2=6/λ=0.3, class-group filter (container ∪ rubbishbin) |
| 3 | Peyman | 0.12039 | **DINOv2**, per-class reciprocal rerank, inverse super-category CE weighting, category-based PK sampling, multi-aspect-ratio training |
| 4 | Narmeen | 0.11947 | Query expansion (weak), **camera-aware Jaccard re-rank** (helped) |
| — | **Us** | **0.11000** | PAT-ViT-B, default recipe |
| 5 | Emre | 0.10949 | Offline ESRGAN (didn't move needle) |

### 1.4 Common Failures (already known)
- Naive SR/ESRGAN alone doesn't help (Emre, Urban2025 paper).
- Quadruplet loss, separate-per-class models, class-specific resizing all hurt (Urban2025 paper).
- Naive ensemble hurts; training-domain validation is optimistic.

### 1.5 Key Insights
- The top scorers all beat us with **bigger backbones** (ViT-L / DINOv2) and **last-layer feature fusion**.
- Class filter (bin_group) consistently helps and we already have it.
- k-reciprocal @ (20,6,0.3) is the community-validated setting — already matches Mohsin.
- **Camera-aware Jaccard rerank** (Narmeen) is a free lunch not yet in our pipeline.
- c004 only → c001-003 domain gap is the core difficulty. Augmentation and backbone strength are our main levers.

---

## 2. Strategy: Three Parallel Tracks

We ship submissions from each track independently; the final result is an ensemble/best-of.

### Track A — Fix & Squeeze the PAT Baseline (fastest ROI, 1–2 days)
Goal: lift current PAT from 0.11 → 0.13+ by fixing known bugs and aligning inference with community wins.

1. **Fix scheduler mismatch** — `solver/scheduler_factory.py` hard-codes `num_epochs=120`; replace with `cfg.SOLVER.MAX_EPOCHS`.
2. **Fix NaN collapse at ep 51** — lower `BASE_LR` from 1e-3 to 5e-4, add gradient clipping (max_norm=1.0), and/or switch to AdamW (matches Aymen's working recipe).
3. **Disable LGT (Local Grayscale Transformation)** — Aymen explicitly flagged it as destructive; color is class-discriminative on urban elements.
4. **Enable the right augs:** ColorJitter(b=0.3, c=0.3, s=0.2, h=0.1, p=0.8), RandomErasing(p=0.5), random patch. (Already present in train YAML — verify they actually activate.)
5. **Train-time input:** keep 256×128; **try 384×128** as a variant (larger vertical resolution helps tall elements like traffic signs).
6. **Last-layer CLS fusion** (Aymen's sweet spot): concatenate CLS from the last 2 transformer blocks → 1536-d descriptor. Implement in `model/make_model.py` during feature extraction; patch `update.py` to use `feat_dim=1536`.
7. **Camera-aware Jaccard re-ranking** (Narmeen): weight gallery-gallery reciprocals by same-cam flag so cross-cam pairs (query c004 ↔ gallery c001-3) are not penalized for having few reciprocal neighbors. Add `--camera_aware` flag to `update.py`.
8. **Train-val split on IDs** — hold out ~10% train IDs for a realistic local eval loop, so future A/B tests are trustworthy (currently impossible).

**Exit criterion:** Kaggle ≥ 0.13 before moving on.

### Track B — Backbone Upgrade (core lift, 2–4 days)
Goal: match/beat Mohsin (ViT-L) and Peyman (DINOv2). This is where the top scorers got their edge.

1. **ViT-L/16** via timm (`vit_large_patch16_224`) — swap into PAT factory (`model/make_model.py`). Drops to batch size ~24–32 on an 11 GB 2080 Ti with 256×128 — verify VRAM; use gradient accumulation if needed.
2. **DINOv2-B/14** (alternative backbone) — `dinov2_vitb14` (86 M params, stronger self-supervised features). Crop sizes must be multiples of 14 (use 224×112 or 252×126). Freeze patch-embed, fine-tune transformer.
3. **Same training recipe as Track A** (AdamW, cosine schedule over actual epochs, grad clip, color jitter, no LGT, random erasing).
4. **Last-2-layer CLS fusion** on both backbones.
5. **Feature dim matters:** ViT-L → 1024×2 = 2048-d; DINOv2-B → 768×2 = 1536-d. Verify `update.py` detects `in_planes` correctly.
6. **Train length:** 30–50 ep with cosine to zero. Save every epoch past ep 25 so we can pick checkpoints if NaN recurs.

**Exit criterion:** at least one backbone run submits ≥ 0.15 on Kaggle.

### Track C — Data Pipeline & Domain Bridge (highest ceiling, 3–7 days)
Goal: close the c001-3 ↔ c004 domain gap. The Urban2025 paper puts data-centric gains at +12 mAP alone.

1. **Merge external data (UAM + URVAM-ReID2026)** — Aymen's working move. The UAM folder has per-class CSVs (container/crosswalk/rubbishbins/trafficsign); URVAM is the older variant. Build a combined train adapter that:
   - Maps UAM/URVAM IDs into a disjoint numeric space (offset by max train ID).
   - Preserves class labels for class-aware sampling.
   - Filters missing image files (some rows may not have images on disk — verify).
2. **Category-balanced PK sampler** (Peyman): trafficsignal is 67% of the data → sample identities inverse-proportional to class frequency. Implement in `data/samplers/triplet_sampler.py`. Inverse-freq weighting for CE head (`SOFT_WEIGHT` per class).
3. **Aug-for-c004 simulation:**
   - Horizontal flip is always-on; add **small perspective warps** (`torchvision.transforms.RandomPerspective(0.3, p=0.5)`) to simulate reverse-direction viewpoint.
   - Small rotation ±10°, scale jitter 0.9–1.1.
   - **Random patch** (per Aymen) — already in transforms builder; confirm it activates.
4. **Multi-scale / multi-aspect training** (Peyman): randomly train on {256×128, 320×128, 256×160} per batch — implement via a batch-level collate that picks one size. Eval at 256×128.
5. **Super-resolution + style transfer** (Urban2025 paper, +4 mAP): only if Tracks A & B plateau. Use Real-ESRGAN × 2 on small rubbishbin crops, then a U-Net w/ VGG-perceptual loss to de-smooth. This is the costliest step and skips to optional.
6. **Class-specific preprocessing** is **not** a priority — the paper showed it hurts.

**Exit criterion:** merged-data run submits ≥ 0.16.

---

## 3. Submission Combination (end-game)

Once we have 2+ strong models (ViT-L-PAT, DINOv2, or merged-data PAT):

1. **Feature-level fusion** (L2-normalize each model's embedding, concatenate, L2-renormalize) — usually > rank-level fusion for ReID.
2. **Query Majority Voting (QMV)** — for each query, use top-K retrievals from model A to re-weight model B's rank. Urban2025 reports this as the final +0.5 mAP at the top of the recipe.
3. **Single-model ensemble is hurtful** (Urban2025 paper's naive ensemble dropped 1+ mAP) — only combine models with genuinely different inductive biases (e.g. ViT-L vs DINOv2 vs SE-ResNet50 if revived).

---

## 4. Prioritized Execution Plan

Order is by **expected mAP-per-hour**:

### Week 1 — Fix + squeeze (target: 0.13)
- [ ] A1: Fix scheduler num_epochs. (1 h)
- [ ] A2: Add gradient clipping + lower LR + switch to AdamW. Retrain PAT. (8 h GPU)
- [ ] A3: Disable LGT, enable color jitter + random erasing. Retrain. (8 h GPU)
- [ ] A6: Implement last-2-layer CLS fusion in PAT extractor. Re-infer with existing checkpoint. (2 h)
- [ ] A7: Camera-aware Jaccard re-rank in `update.py`. (2 h)
- [ ] A8: ID-based train/val split; wire into `UrbanElementsReID.py`. (3 h)
- [ ] Submit best A-track run to Kaggle.

### Week 2 — Backbone (target: 0.15–0.16)
- [ ] B1: ViT-L/16 swap. Verify VRAM, reduce batch if needed. Train 30 ep. (24 h GPU)
- [ ] B2 (parallel if time): DINOv2-B/14 swap. Train 30 ep. (24 h GPU)
- [ ] Run CLS-fusion + camera-aware re-rank + class filter on both.
- [ ] Submit best B-track run.

### Week 3 — Data (target: 0.17+)
- [ ] C1: External-data merger (UAM + URVAM). Validate IDs non-overlapping. Retrain best backbone. (24 h GPU)
- [ ] C2: Inverse-frequency PK sampler + class-weighted CE. Retrain.
- [ ] C3: Perspective/scale augs. Retrain.
- [ ] Submit best C-track run.

### Week 4 — Ensemble (target: stretch)
- [ ] Feature-fusion of top-2 models. Submit.
- [ ] QMV on top-3. Submit.

---

## 5. Concrete File-Level Changes Already Needed

These are surgical fixes — apply in Week 1:

1. `Part-Aware-Transformer/solver/scheduler_factory.py` — replace hard-coded `num_epochs=120` with `cfg.SOLVER.MAX_EPOCHS`.
2. `Part-Aware-Transformer/loss/build_loss.py` — branch check: `local_attention_vit` → `part_attention_vit`.
3. `Part-Aware-Transformer/config/UrbanElementsReID_train.yml`:
   - `SOLVER.OPTIMIZER_NAME: AdamW` (it's already AdamW in the current file — good, keep)
   - `INPUT.LGT.DO_LGT: False`
   - `INPUT.REA.ENABLED: True, PROB: 0.5`
   - `SOLVER.BASE_LR: 0.00035` (BoT/AdamW canonical)
   - Add grad-clip param (may need processor patch).
4. `Part-Aware-Transformer/processor/part_attention_vit_processor.py` — add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` before `optimizer.step()`.
5. `Part-Aware-Transformer/update.py` — add `--camera_aware` flag; load camera info from `query.csv`/`test.csv` and compute cam-aware k-reciprocal.
6. `Part-Aware-Transformer/data/datasets/UrbanElementsReID.py` — add train/val ID split.

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| ViT-L OOM on 2080 Ti (11 GB) | Drop batch to 16, enable AMP, gradient accumulation ×2 to keep effective B=64; or use input 224×112. |
| External data introduces noise (annotation inconsistency between Urban2026 and UAM) | Train two versions: with UAM only, with UAM+URVAM. Compare on held-out val split (not Kaggle budget). |
| NaN recurrence under new recipe | Grad clip + AMP disabled for first run; checkpoint every epoch past ep 20. |
| Kaggle daily submission cap | Use held-out val split as primary signal; only submit ≤ 1 per decisive change. |
| Overfitting to train-val split | Use both held-out val AND Kaggle score; disagreement = redesign val. |

---

## 7. Non-Goals

Explicitly skip (evidence says they don't help):
- Quadruplet loss.
- Separate-per-class models.
- Class-specific resizing.
- Naive ensemble of same-arch same-seed runs.
- Standalone offline SR (Emre proved it).

---

## 8. Success Criteria

- **Minimum win:** Kaggle mAP ≥ 0.17 (beats target).
- **Stretch:** ≥ 0.20 (clear first place given current leaderboard).
- **Non-negotiable:** every A/B experiment must be compared on the held-out val split before a Kaggle submission, not on the current train-split "mAP".
