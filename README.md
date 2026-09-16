# 3D OCT Convolutional Autoencoder + Glaucoma Classifier

Self-supervised pretraining on 3D optical coherence tomography volumes, followed by
glaucoma classification from the learned latent representation combined with patient
demographics.

Trained end to end on a single RTX 4060 Ti (8GB).

## Dataset

[Harvard-GF](https://huggingface.co/datasets/harvardairobotics/Harvard-GF) — 3,300 OCT
volumes of the optic nerve head, one `.npz` per eye containing a 200×200×200 uint8
scan and a binary glaucoma label. Demographics come from `data_summary.csv`.

| split | volumes | glaucoma |
|---|---|---|
| training | 2,100 | 51.6% |
| validation | 300 | 58.7% |
| test | 900 | 54.3% |

Splits are pre-defined by the dataset's `use` column. Race is balanced 1100/1100/1100 —
Harvard-GF is a fairness benchmark.

### Preprocessing

Centre crop to 128×128×128, isolating the optic nerve head and cutting the compression
ratio by a factor of four. Volumes are handed to the GPU as raw uint8 and normalized
to [-1, 1] there, which keeps the CPU out of the critical path.

## Stage 1 — Autoencoder

```
encoder   1×128³ → 256×4³ → flatten 16,384 → latent 8,192
decoder   8,192 → 16,384 → 256×4³ → 1×128³
```

Five 3D convolutional layers each way, kernel 3 / stride 2 / padding 1, ReLU
throughout, Tanh on the output. 256:1 compression.

**Loss:** `0.6 · MSE + 0.4 · (1 − SSIM)`. MSE alone produces blurred layer boundaries
because it is minimised by the conditional mean; the structural term rewards matching
local contrast and pattern. SSIM is computed in fp32 — its variance terms underflow
under mixed precision.

| | |
|---|---|
| epochs | 15 |
| optimizer | Adam, lr 1e-4 |
| batch size | 8 |
| precision | mixed (AMP), loss in fp32 |

The encoder is verified to carry per-subject anatomy rather than a population average:
each reconstruction scores 4-5× lower error against its own volume than against another
subject's.

## Stage 2 — Classifier

The encoder is **frozen**. Fine-tuning 138M parameters on 2,100 samples overfits from
the first epoch — validation loss rises while training loss falls.

```
LayerNorm(8192)                  latent only
concat 15 demographic features → 8,207
Linear(8207, 512) → ReLU → Dropout(0.5)
Linear(512, 2)
```

The demographics (age, gender, race, ethnicity, marital status; one-hot encoded) join
*after* the normalization — passing them through it would rescale 0/1 indicators by the
latent's statistics and destroy their meaning.

The LayerNorm is load-bearing. The encoder ends in a bare Linear, so latent dimensions
have arbitrary and wildly differing scales; without it, training stalls around 60% and
never converges.

| | |
|---|---|
| epochs | 15 |
| loss | Cross Entropy |
| optimizer | Adam, lr 1e-3 |
| batch size | 2 |
| weight decay | 1e-4 |

## Results

Test set, 900 volumes:

| metric | value |
|---|---|
| accuracy | 71.00% |
| majority baseline | 54.33% |
| **balanced accuracy** | **71.39%** |
| specificity | 75.91% |
| sensitivity | 66.87% |

```
              pred no   pred yes
actual no      75.91%     24.09%
actual yes     33.13%     66.87%
```

Balanced accuracy is the headline figure — any constant predictor scores exactly 50%
on it, so unlike raw accuracy it cannot be inflated by class imbalance.

## Limitations

- **Sensitivity.** A third of glaucoma cases are missed. The decision threshold is left
  at 0.5 rather than tuned for screening, where false negatives cost more than false
  alarms.
- **Sample size.** 2,100 training volumes against a 138M-parameter encoder is why
  fine-tuning had to be abandoned.
- **Reconstruction ceiling.** Five stride-2 layers mean each latent cell summarizes 32
  voxels, and OCT speckle is unpredictable by construction. Six different loss
  configurations produced visually identical reconstructions.
- **Demographics added no measurable gain**, despite age alone reaching AUC 0.617. The
  latent appears to already encode age through visible retinal ageing.
- **No AUC or per-group analysis reported**, which the fairness benchmark is designed for.

## Files

```
AutoEncoder.py                    Encoder / Decoder / AutoEncoder
OCT3D_DataLoader.py               Dataset, crop, demographics join
OCT3D_Pretrained_Classifier.py    classification head
OCT3D_AutoEncoder.ipynb           stage 1 training and evaluation
OCT3D_Classifier.ipynb            stage 2 training and evaluation
```

Set `DATA` in `OCT3D_DataLoader.py` to your dataset path, then run the autoencoder
notebook followed by the classifier notebook. Model weights are gitignored.
