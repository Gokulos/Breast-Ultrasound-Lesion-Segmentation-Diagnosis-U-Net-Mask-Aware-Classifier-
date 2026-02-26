## Breast Ultrasound Lesion Segmentation + Diagnosis Classification (Two-Stage Deep Learning)

This project implements a two-stage deep learning pipeline on the Breast Ultrasound Images Dataset (BUSI):

1️⃣ U-Net segmentation to localize breast lesions
2️⃣ Mask-aware classifier to predict normal / benign / malignant

The model first identifies the lesion region and then performs diagnosis using both the image and predicted mask, improving interpretability compared to direct classification.

⚠️ Disclaimer: This project is for research and educational purposes only. It is not a medical diagnostic tool.

🚀 Project Motivation

Most student projects perform either segmentation or classification.
This project combines both to mimic a more realistic clinical AI workflow:

Detect lesion region (Where?)

Predict diagnosis (What?)

This improves:

interpretability

robustness

alignment with real medical AI systems

## Architecture Overview
Stage 1 — Lesion Segmentation

Model: U-Net

Input: ultrasound image

Output: binary lesion mask

Stage 2 — Mask-Aware Diagnosis Classification

Model: CNN classifier

Input: 2-channel tensor

channel 1 → ultrasound image

channel 2 → predicted mask

Output: normal / benign / malignant

## Dataset

Breast Ultrasound Images Dataset (BUSI)
https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

Dataset contains:

Ultrasound images

Corresponding lesion masks (*_mask.png)

Class labels: normal / benign / malignant

Important preprocessing details:

Images resized and normalized

Masks binarized

Normal samples assigned empty masks

Segmentation trained only on images with real masks

## Installation
git clone <your_repo_url>
cd breast-ultrasound-ai

python -m venv .venv
source .venv/bin/activate      # Linux / Mac
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
🧪 Training
1️⃣ Train U-Net (Segmentation)
python src/train_unet.py

Outputs:

models/unet.h5

Metrics:

Dice coefficient

BCE + Dice loss

2️⃣ Train Classifier (Diagnosis)
python src/train_classifier.py

Outputs:

models/classifier.h5

Metrics:

Accuracy

F1-score per class (recommended)

🔎 Inference
python src/infer.py --image path/to/image.png

Output:

predicted mask

predicted class

class probabilities

🖥️ GUI Demo (optional)
python src/gui_app.py

Displays:

input image

predicted lesion mask

overlay visualization

predicted diagnosis

📊 Results (add your numbers here)
Task	Metric	Score
Segmentation	Dice	TBD
Classification	Accuracy	TBD
Malignant Detection	Recall	TBD
⭐ Key Learning Points

Multi-stage medical AI pipelines

Segmentation-guided classification

Handling missing masks (normal class)

Avoiding data leakage when generating predicted masks

Interpretable deep learning design

## Future Improvements

Dice + Focal hybrid loss

Attention U-Net

Uncertainty estimation

Grad-CAM explainability

Clinical risk calibration

Multi-view ultrasound fusion

## References

Segmentation

Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI 2015
https://arxiv.org/abs/1505.04597

Isensee et al., nnU-Net: Self-configuring method for biomedical segmentation
https://arxiv.org/abs/1809.10486

Medical Ultrasound AI

Yap et al., Automated Breast Ultrasound Lesion Detection Using Deep Learning
https://ieeexplore.ieee.org/document/8373050

BUSI dataset paper
Al-Dhabyani et al., Dataset of breast ultrasound images
https://doi.org/10.1016/j.dib.2019.104863

Explainability & Multi-Stage Medical AI

Litjens et al., A Survey on Deep Learning in Medical Image Analysis
