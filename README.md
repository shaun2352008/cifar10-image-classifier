# CIFAR-10 Image Classifier

A small convolutional neural network (CNN) built with **PyTorch** that looks at tiny colour images and predicts which of 10 categories they belong to (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck).

I built this as a hands-on project to learn the full machine learning workflow: loading data, defining a model, training it, evaluating it honestly on unseen images, and understanding where it goes wrong.

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [How It Works](#how-it-works)
5. [Getting Started](#getting-started)
6. [Results](#results)
7. [Project Structure](#project-structure)
8. [Key Concepts](#key-concepts)
9. [Limitations](#limitations)
10. [Future Improvements](#future-improvements)
11. [Tech Stack](#tech-stack)

---

## Overview

Image classification is one of the core problems in computer vision. This project trains a compact CNN from scratch (no pretrained weights) on the CIFAR-10 dataset and reports:

- Test accuracy after every training epoch
- Final test accuracy on 10,000 images the model never trained on
- A confusion matrix showing which classes the model mixes up

The goal is clarity and reproducibility, not a record-breaking score.

## Dataset

**CIFAR-10** is a standard benchmark dataset:

- 60,000 colour images, each 32 x 32 pixels
- 10 classes, 6,000 images per class
- 50,000 training images and 10,000 test images

The dataset downloads automatically the first time you run the script (about 170 MB). It is not stored in this repository.

## Model Architecture

The model (`SmallCNN` in [train.py](train.py)) has two parts.

**Feature extractor**
| Layer | Output shape |
|---|---|
| Input image | 3 x 32 x 32 |
| Conv2d (3 to 32 filters, 3x3) + ReLU + MaxPool | 32 x 16 x 16 |
| Conv2d (32 to 64 filters, 3x3) + ReLU + MaxPool | 64 x 8 x 8 |

**Classifier head**
| Layer | Output shape |
|---|---|
| Flatten | 4096 |
| Linear + ReLU | 128 |
| Linear | 10 (one score per class) |

**Training setup**
- Loss: Cross-entropy
- Optimizer: Adam, learning rate 0.001
- Batch size: 64
- Epochs: 5
- Input normalisation: each colour channel scaled to the range -1 to 1

## How It Works

1. **Load data:** CIFAR-10 is downloaded and converted into tensors, then normalised.
2. **Build model:** the CNN is created and moved to the GPU if one is available.
3. **Train:** for each batch of images, the model makes predictions, the loss measures how wrong they are, and the optimizer adjusts the weights to reduce it.
4. **Evaluate:** after each epoch, the model is tested on the 10,000 held-out test images and the accuracy is printed.
5. **Visualise:** at the end, a confusion matrix is generated and saved as `confusion_matrix.png`.

## Getting Started

### Option A: Run on Google Colab (no setup needed)
1. Open [colab.research.google.com](https://colab.research.google.com) and create a new notebook.
2. Go to **Runtime > Change runtime type** and select **T4 GPU**.
3. Paste the contents of `train.py` into a cell and run it.

### Option B: Run locally
```bash
git clone https://github.com/[YOUR-USERNAME]/cifar10-image-classifier.git
cd cifar10-image-classifier
pip install -r requirements.txt
python train.py
```

Training takes a few minutes on a GPU and longer on a CPU.

## Results

| Metric | Value |
|---|---|
| Epochs | 5 |
| Final test accuracy | **[72.1%]%** |

Training log:
```
Epoch 1/5 | loss 1.350 | test accuracy 60.8%
Epoch 2/5 | loss 0.983 | test accuracy 66.3%
Epoch 3/5 | loss 0.833 | test accuracy 68.8%
Epoch 4/5 | loss 0.722 | test accuracy 69.5%
Epoch 5/5 | loss 0.622 | test accuracy 72.1%

Final test accuracy: 72.1%
```

Confusion matrix:

<img width="1200" height="1200" alt="confusion_matrix (2)" src="https://github.com/user-attachments/assets/7c426979-8c68-434e-90cf-0cfa6f5f43f9" />


**What the results show:** [Write 2 to 3 sentences in your own words. For example: which classes the model gets right most often, and which pairs it confuses, such as cats and dogs. Look at your matrix and describe what you actually see.]

## Project Structure

```
cifar10-image-classifier/
├── train.py              # data loading, model, training loop, evaluation
├── requirements.txt      # Python dependencies
├── confusion_matrix.png  # generated after training
└── README.md
```

## Key Concepts

- **Epoch:** one full pass through all the training images.
- **Convolutional layer:** scans the image with small filters to detect patterns like edges, textures, and shapes.
- **Max pooling:** shrinks the image while keeping the strongest signals, which makes the model faster and less sensitive to exact positions.
- **Loss:** a number measuring how wrong the predictions are. Training tries to push it down.
- **Test accuracy:** the percentage of correct predictions on images the model has never seen. This is the honest measure of performance.
- **Confusion matrix:** a grid where rows are the true classes and columns are the predictions. Bright cells off the diagonal show common mistakes.

## Limitations

- The model is small and trained for only 5 epochs, so accuracy is modest compared with state-of-the-art models.
- No data augmentation is used, so the model can overfit to the training images.
- CIFAR-10 images are only 32 x 32 pixels, so fine details are lost. Similar classes such as cats and dogs are hard to separate.
- Results vary slightly between runs because of random initialisation and shuffling.

## Future Improvements

- Train for more epochs and track training vs test accuracy to check for overfitting
- Add data augmentation (random flips, crops)
- Add batch normalisation and dropout
- Try a deeper network or transfer learning with a pretrained model such as ResNet
- Save the trained model and build a small demo that classifies an uploaded image

## Tech Stack

- Python
- PyTorch and torchvision
- scikit-learn (confusion matrix)
- Matplotlib (plotting)
- Google Colab (training)

## Acknowledgements

- CIFAR-10 dataset by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- PyTorch documentation and tutorials
