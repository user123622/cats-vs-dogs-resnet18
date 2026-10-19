# Cats vs. Dogs Classification with ResNet18

An image classifier built with **PyTorch** using **transfer learning** on a pretrained **ResNet18**, trained on the Microsoft PetImages dataset (~25,000 images).

The project covers a full deep learning pipeline: data cleaning, augmentation, a reproducible train/dev/test split, training, and a final evaluation on unseen data.

## Results

| Set | Images | Accuracy |
|---|---|---|
| Train (with augmentation) | 24,498 | 90.60% |
| Dev (validation) | 249 | 96.8% – 98.0% across epochs |
| **Test (unseen data)** | **251** | **98.80% (248 / 251)** |

**Why is training accuracy lower than test accuracy?** Training accuracy is measured on heavily augmented images (blur, erased patches, color and geometry changes) with dropout active, which makes them much harder than the clean test images. This gap is the intended effect of the augmentation: the model no longer memorizes the training set.

*Note: the Dev and Test sets are small (~250 images each), so a single image changes accuracy by about 0.4%. Small differences between epochs are within that noise.*

## What I did

**1. Cleaned the raw data**
The dataset contains corrupted and zero-byte images that crashed training with `UnidentifiedImageError`. I wrote a script that fully loads every image, removes files that cannot be decoded, and removes non-image files.

**2. Transfer learning**
I used a ResNet18 pretrained on ImageNet, froze the backbone, and replaced the final layer with a custom head:
`Linear(512 → 256) → ReLU → Dropout(0.5) → Linear(256 → 2)`

**3. Diagnosed and addressed overfitting**
In the first version, the model memorized the clean training images within 2–3 epochs while validation accuracy plateaued. To force it to learn more robust features, I added aggressive augmentation to the training set:
- **TrivialAugmentWide** – random rotation, shear, translation, brightness, contrast, saturation and hue
- **Random horizontal flip**
- **Gaussian blur** – removes fine detail
- **Random erasing (Cutout)** – hides random patches so the model can't rely on a single part of the animal

**4. Evaluated correctly**
A fixed-seed 98% / 1% / 1% split keeps results reproducible, and the test set is used only once, after training is complete.

## Tech stack

Python · PyTorch · torchvision · ResNet18 · AdamW · Cross-Entropy Loss · Google Colab (GPU)

## How to run

1. Open `cats_vs_dogs_resnet18.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Set the runtime to GPU: *Runtime → Change runtime type → GPU*.
3. Run all cells. The notebook downloads the dataset automatically.

## Dataset

[Microsoft Kaggle Cats and Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765). The dataset is not included in this repository; the notebook downloads it.
