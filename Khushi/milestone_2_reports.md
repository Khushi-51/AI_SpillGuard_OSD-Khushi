# 🛢️ Oil Spill Detection – Milestone 2 Report

## ✅ Overview

In this milestone, we focused on building the **dataset pipeline, model architecture, and training setup** for oil spill detection using deep learning. By the end of this stage, we have a complete workflow that takes raw satellite images → preprocesses them → feeds them into a UNet segmentation model → and begins the training process.

---

## 📂 Dataset Pipeline

* **Preprocessing:**

  * Images and segmentation masks resized to 512×512.
  * Normalization applied (`mean=0.485`, `std=0.229`).
  * Augmentations added (flip, rotation, brightness/contrast, noise).
* **Dataloader:**

  * Separate loaders for training, validation, and test sets.
  * Batched data loading for efficient GPU/CPU usage.
* **Verification Outputs:**

  * Visualized a few image-mask pairs to confirm correct alignment.
  * Printed dataset stats (number of samples, image shape, channels).

---

## 🏗️ Model Development

* Implemented **U-Net** for semantic segmentation.
* Architecture verified with `torchsummary`:

  * Input: (1 × 512 × 512) grayscale image
  * Output: (1 × 512 × 512) binary segmentation mask
* Model parameters and layer structure printed for validation.

---

## ⚙️ Training Setup

* **Loss function:** Binary Cross Entropy with Logits (`BCEWithLogitsLoss`)
* **Optimizer:** Adam (learning rate = 0.001, weight decay = 0.0001)
* **Scheduler:** ReduceLROnPlateau (patience = 3)
* **Metrics tracked:** Training loss, Validation loss, IoU/Accuracy
* **Checkpoints:** Model saving every few epochs for recovery and monitoring.

---

## 📊 Outputs & Observations

* Sample preprocessed image and its segmentation mask displayed.
* Model summary confirmed correct number of parameters.
* Initial training logs show the loop executing successfully (loss decreasing across epochs).
* Validation pipeline in place to prevent overfitting.

---

## 🚀 Milestone 2 Achievements

* End-to-end dataset → model → training loop **implemented successfully**.
* Verified augmentations, normalization, and dataloaders.
* Defined UNet architecture for segmentation task.
* Set up training with loss, optimizer, scheduler, and metrics.
* Run **full training** with more epochs.
* Save best model checkpoints.

---

## 🔜 Next Steps (Milestone 3)

* Evaluate on test set (IoU, Dice score, Precision/Recall).
* Visualize predictions (overlay segmentation mask on satellite images).

---

📌 *With Milestone 2 completed, the project has a working deep learning pipeline and trained model ready for visulisation and evaluation.*
