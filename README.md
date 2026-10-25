# Oral Disease Classifier

Deep learning project that classifies oral diseases from photos into 6 classes. It compares a custom CNN against three ImageNet-pretrained models, fine-tunes the best one (ResNet50), and ships it in a Gradio web app with Grad-CAM heatmaps, a PDF report and batch mode.

> **Medical disclaimer:** this is a student research project and a screening aid, not a diagnostic tool. It must not replace a dentist or physician.

## Results

All numbers come from a held-out **test set of 506 images**, opened once after the model was chosen on validation. Confidence intervals are 95% bootstrap (2,000 resamples).

| Model | Test accuracy | Test macro-F1 | Params (M) | File (MB) |
| --- | --- | --- | --- | --- |
| **ResNet50 (tuned)** | **0.947** | **0.884** | 23.6 | 95.1 |
| ResNet50 | 0.919 | 0.852 | 23.6 | 210.6 |
| MobileNetV2 | 0.887 | 0.822 | 2.27 | 21.8 |
| EfficientNetB0 | 0.877 | 0.804 | 4.06 | 29.0 |
| Custom CNN | 0.800 | 0.699 | 1.21 | 14.6 |

Key findings:

- All four pretrained models clearly beat the custom CNN (paired-bootstrap CI of the macro-F1 difference excludes zero).
- Tuned vs. untuned ResNet50 is **not statistically distinguishable** (+0.032 macro-F1, CI [-0.022, +0.088]), so the tuning gain is not claimed as established.
- **Caries** (recall 0.607) and **Discoloration** (0.739) are the weak classes; they have only 28 and 23 test images, so their estimates are noisy.
- MobileNetV2 is 4.4x smaller and 2.5x faster than the tuned ResNet50, at 0.062 lower macro-F1, and is the fallback if size matters most.

Per-class results of the deployed model (tuned ResNet50) on the test set:

| Class | Test images | Recall | Precision |
| --- | --- | --- | --- |
| Calculus | 60 | 0.950 | 0.919 |
| Caries | 28 | 0.607 | 0.810 |
| Discoloration | 23 | 0.739 | 0.773 |
| Gingivitis | 183 | 0.973 | 0.957 |
| Hypodontia | 180 | 0.994 | 0.978 |
| Mouth_Ulcer | 32 | 0.969 | 0.969 |

The full analysis (CIs, paired tests, error analysis, speed) is in [`reports/model_comparison_report.docx`](reports/model_comparison_report.docx).

## Dataset

[Oral Diseases dataset on Kaggle]([PASTE_KAGGLE_LINK_HERE](https://www.kaggle.com/datasets/salmansajid05/oral-diseases
)): 6 classes (Calculus, Caries, Discoloration, Gingivitis, Hypodontia, Mouth_Ulcer). Full credit to the dataset authors.

Only the 5,563 original images are used (the augmented copies and the YOLO folder are excluded). A leak-free cleaning and split pipeline was applied:

- Exact duplicates (MD5) are removed; a file that appears under several classes is dropped.
- Near-duplicates are detected with a perceptual hash (pHash, Hamming distance <= 4). Near-duplicates with different labels are dropped; those with the same label are forced into the same split.
- Stratified 70/15/15 split grouped by near-duplicate group: **2,497 train / 498 validation / 506 test**.
- Class imbalance is handled with class weights.

## Method

Five models, same data, augmentation (flip, rotation, zoom, contrast), class weights and 224x224 input:

1. **Custom CNN**: 4 conv blocks, global average pooling, dense head, trained from scratch.
2. **ResNet50, EfficientNetB0, MobileNetV2**: ImageNet-pretrained, two-phase training (head only, then partial fine-tuning with BatchNorm frozen).
3. **ResNet50 (tuned)**: random search over 8 trials (learning rate, unfrozen layers, dropout, batch size). Best: lr 1e-4, 80 unfrozen layers, dropout 0.5, batch 16.

## Repository structure

```
notebooks/   step1 data cleaning + split -> step5 final test evaluation
app/         Gradio web app (app.py, requirements.txt)
reports/     model comparison report
```

| Notebook | Purpose |
| --- | --- |
| `step1_data_cleaning_split.ipynb` | Deduplication and leak-free split |
| `step2_custom_cnn.ipynb` | Custom CNN baseline |
| `step3_pretrained_models.ipynb` | ResNet50, EfficientNetB0, MobileNetV2 |
| `step4_tune_resnet50.ipynb` | Hyperparameter search for ResNet50 |
| `step5_final_test_evaluation.ipynb` | Test metrics, CIs, error analysis |

## Run the app

The model file (95 MB) is hosted on Hugging Face because it is too large for the GitHub web uploader.

**Model:** [mazen-ayad/Oral_Project on Hugging Face](https://huggingface.co/mazen-ayad/Oral_Project)

```bash
git clone https://github.com/USERNAME/oral-disease-classifier.git
cd oral-disease-classifier/app

# download the model next to app.py
curl -L -o ResNet50_tuned_best.keras \
  https://huggingface.co/mazen-ayad/Oral_Project/resolve/main/ResNet50_tuned_best.keras

pip install -r requirements.txt
python app.py
```

Then open the local URL printed in the terminal. You can also point to the model elsewhere with the `MODEL_PATH` environment variable.

App features: single-image analysis with confidence, Grad-CAM heatmap, image-quality warnings, downloadable PDF report, batch mode with CSV export, and a session history tab.

## Limitations

- One split and one training seed per model; close rankings could change with another seed.
- Validation was used for both hyperparameter and best-epoch selection, so validation scores are optimistic; only test numbers are reliable.
- Near-duplicate test images are correlated, so the confidence intervals are slightly too narrow.
- Single public dataset; performance on photos from other cameras or clinics is untested.

## Author

**Mazen Ayad**
