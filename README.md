# ProjectExhibition

# Requirements 

# ── Core ML ────────────────────────────────────────────────────
tensorflow>=2.13.0
keras>=2.13.0

# ── Data & Utils ───────────────────────────────────────────────
numpy>=1.24.0
Pillow>=9.5.0
scikit-learn>=1.3.0

# ── Visualization ──────────────────────────────────────────────
matplotlib>=3.7.0
seaborn>=0.12.0

# ── Kaggle dataset download (optional) ────────────────────────
kaggle>=1.5.16

# ── Notebook support ───────────────────────────────────────────
jupyter>=1.0.0
ipywidgets>=8.0.0


# 🌿 Plant Disease Detection — ML Module

EfficientNetB3 fine-tuned on PlantVillage dataset.  
**38 classes · 9 crop types · 54K+ training images**

---

## Architecture

```
EfficientNetB3 (ImageNet pretrained)
    └── GlobalAveragePooling2D
    └── BatchNormalization
    └── Dropout(0.4)
    └── Dense(512, relu)
    └── BatchNormalization
    └── Dropout(0.3)
    └── Dense(38, softmax)   ← output layer
```

## Two-Phase Training Strategy

| Phase | What's trained | LR | Epochs |
|-------|---------------|-----|--------|
| 1 — Head | Classification head only (base frozen) | 1e-3 | 15 |
| 2 — Fine-tune | Top 30 EfficientNet layers + head | 1e-5 | 25 |

This avoids catastrophic forgetting while still adapting the feature extractor to plant imagery.

---

## Supported Plants & Diseases

| Plant | Classes |
|-------|---------|
| Tomato | 10 (9 diseases + healthy) |
| Apple | 4 (3 diseases + healthy) |
| Grape | 4 (3 diseases + healthy) |
| Corn | 4 (3 diseases + healthy) |
| Potato | 3 (2 diseases + healthy) |
| Pepper | 2 (1 disease + healthy) |
| Strawberry | 2 (1 disease + healthy) |
| Cherry | 2 (1 disease + healthy) |
| Peach | 2 (1 disease + healthy) |
| + more | ... |

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset from Kaggle
chmod +x download_dataset.sh
./download_dataset.sh ./data

# 3. Train
python train.py \
  --data_dir ./data/plantvillage_raw/plantvillage\ dataset/color \
  --model_dir ./models/output \
  --img_size 224 \
  --batch_size 32 \
  --head_epochs 15 \
  --finetune_epochs 25
```

## File Structure

```
ml/
├── train.py                  # Main training script
├── inference.py              # Inference engine (used by backend)
├── requirements.txt
├── download_dataset.sh
├── notebooks/
│   └── training_notebook.ipynb
└── utils/
    ├── config.py             # Hyperparameters & paths
    ├── data_loader.py        # Train/val/test split + augmentation
    ├── class_weights.py      # Balanced class weighting
    └── metrics.py            # Eval, confusion matrix, plots
```

## Output Files (after training)

```
models/output/
├── plant_disease_model_final.h5    # Final model
├── saved_model/                    # TF SavedModel format
├── class_indices.json              # Index → class label mapping
├── training_history.png            # Loss/acc curves
├── confusion_matrix.png            # Per-class confusion matrix
├── classification_report.txt       # Precision/recall/F1
├── checkpoints/
│   ├── best_phase1.h5
│   └── best_phase2.h5
└── logs/
    ├── phase1/                     # TensorBoard logs
    └── phase2/
```

## Expected Performance (PlantVillage)

| Metric | Expected |
|--------|----------|
| Top-1 Accuracy | ~96–98% |
| Top-3 Accuracy | ~99%+ |

> Note: Real-world field images will be lower. Consider adding field photos to your training set.

## Run Inference Standalone

```bash
python inference.py \
  ./models/output/plant_disease_model_final.h5 \
  ./models/output/class_indices.json \
  path/to/leaf_image.jpg
```

## TensorBoard

```bash
tensorboard --logdir ./models/output/logs
```

---

## Next Steps

Once training is complete, the `inference.py` module is imported directly by the **FastAPI backend**.  
Point it to your saved model path — no changes needed.
