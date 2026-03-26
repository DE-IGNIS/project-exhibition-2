# ProjectExhibition

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

# 2. Downloading dataset from Kaggle
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
├── train.py                  
├── inference.py              
├── requirements.txt
├── download_dataset.sh
├── notebooks/
│   └── training_notebook.ipynb
└── utils/
    ├── config.py             
    ├── data_loader.py        
    ├── class_weights.py      
    └── metrics.py            
```

## Output Files (after training)

```
models/output/
├── plant_disease_model_final.h5    
├── saved_model/                    
├── class_indices.json              
├── training_history.png           
├── confusion_matrix.png            
├── classification_report.txt       
├── checkpoints/
│   ├── best_phase1.h5
│   └── best_phase2.h5
└── logs/
    ├── phase1/                     
    └── phase2/
```

## Expected Performance (PlantVillage)

| Metric | Expected |
|--------|----------|
| Top-1 Accuracy | ~94–96% |
| Top-3 Accuracy | ~98%+ |

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

# Requirements 

# Core ML
tensorflow>=2.13.0
keras>=2.13.0

# Data & Utils
numpy>=1.24.0
Pillow>=9.5.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

#  Notebook support
jupyter>=1.0.0
ipywidgets>=8.0.0
