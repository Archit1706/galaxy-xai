# Galaxy Morphology Classification and Explainable AI Benchmarking

## Socially Responsible AI Final Project

### Evaluating CNN Classifiers and Post-hoc XAI Methods for Trustworthy Galaxy Classification Across Datasets

---

# Overview

Modern astronomical surveys generate massive volumes of galaxy imagery that exceed human classification capacity. Convolutional Neural Networks (CNNs) are now widely used to automate galaxy morphology classification tasks such as distinguishing **Smooth** galaxies from **Featured** galaxies with spiral arms or structural details.

While these models achieve high predictive accuracy, they often function as **black boxes**, offering little transparency into *why* a specific prediction was made. In scientific domains, this lack of interpretability poses serious challenges:

* Difficult debugging of incorrect predictions
* Poor trust under domain shift across surveys
* Risk of spurious correlations
* Reduced scientific reliability

This project investigates how effectively major **post-hoc Explainable AI (XAI)** methods explain CNN-based galaxy classifiers, and whether those explanations remain reliable across:

* Different classifier architectures
* Different datasets
* Different evaluation metrics

---

# Project Objectives

## Main Research Question

**Which XAI methods produce the most faithful, stable, and scientifically trustworthy explanations for galaxy morphology classifiers?**

### Sub-questions

* Which XAI method best reflects true model reasoning?
* How stable are explanations under small perturbations?
* Do XAI rankings generalize across architectures?
* Do findings remain consistent across datasets?
* Is there an accuracy–interpretability tradeoff?

---

# Datasets

## Dataset 1: Galaxy10 DECaLS

* Source: Hugging Face (`matthieulel/galaxy10_decals`)
* Original size: 17,736 images
* Processed size: 9,060 balanced images
* Resolution: 224×224
* Task: Binary classification

  * Smooth
  * Featured
* Split:

  * Train: 70%
  * Validation: 15%
  * Test: 15%

### Why chosen?

* Real astronomical labels from Galaxy Zoo
* DECam survey similarity to Rubin Observatory
* Large enough for robust CNN training
* Easy reproducibility

---

## Dataset 2: Galaxy Zoo Evo (tiny)

* Source: Hugging Face (`mwalmsley/gz_evo`)
* Original size: 8,053 images
* Usable after thresholding: 1,698 images
* Labeling via vote-fraction threshold ≥ 0.6
* Practical task reduced to binary classification

### Why chosen?

* Cross-survey dataset
* Different telescope sources
* Different curation standards
* Enables robustness testing

---

# Preprocessing Pipeline

* Resize to 224×224
* ImageNet normalization
* Data augmentation:

  * Horizontal flips
  * Vertical flips
  * 180° rotations
  * Color jitter
* Stratified splitting
* Class balancing

---

# Classifier Architectures

## 1. ResNet-18

* Pretrained on ImageNet
* Residual connections
* 11.2M parameters
* Fine-tuned upper layers

### Purpose:

Baseline modern architecture with strong literature precedent.

---

## 2. VGG-16

* Pretrained on ImageNet
* Sequential deep convolutional stack
* 138M parameters

### Purpose:

Older non-residual paradigm for architecture diversity.

---

## 3. EfficientNet-B0

* Pretrained on ImageNet
* Compound scaling
* Squeeze-and-Excitation blocks
* 4.0M parameters

### Purpose:

Modern efficiency-focused architecture.

---

## 4. Custom CNN

* Scratch-trained
* 4 convolutional blocks
* 3.27M parameters

### Purpose:

Control model to isolate pretraining effects.

---

# Explainable AI Methods

## Grad-CAM

* Gradient-based
* Model-specific
* Coarse localization

### Strength:

Highly stable and computationally efficient

---

## LIME

* Perturbation-based
* Model-agnostic
* Superpixel explanations

### Strength:

Architecture-independent

### Weakness:

Very slow

---

## Integrated Gradients

* Path-integral gradients
* Theoretically grounded
* Completeness guarantees

---

## GradientSHAP

* Shapley-value inspired
* Game-theoretic
* High pixel precision

### Strength:

Best fine-grained insertion performance

### Weakness:

Low stability

---

# Evaluation Metrics

## Faithfulness Metrics

### Deletion AUC ↓

Measures confidence drop when important pixels are removed.

### Insertion AUC ↑

Measures confidence recovery when important pixels are added.

---

## Stability Metric

### Consistency ↑

Cosine similarity under Gaussian noise perturbation.

---

## Usability Metric

### Sparsity / Gini ↑

Measures attribution concentration.

---

# Statistical Validation

## Wilcoxon Signed-Rank Test

Used for pairwise XAI comparison.

---

## McNemar Test

Used for classifier prediction comparison.

---

# Key Results

# Dataset 1 Findings

## Classifier Performance

| Model           | Accuracy |
| --------------- | -------- |
| ResNet-18       | 96.10%   |
| VGG-16          | 93.52%   |
| EfficientNet-B0 | 92.20%   |
| Custom CNN      | 93.23%   |

### Key takeaway:

**ResNet-18 was the strongest overall classifier.**

---

## XAI Performance

### Grad-CAM:

* Best Deletion AUC
* Most stable
* Strongest general-purpose method

### GradientSHAP:

* Best Insertion AUC
* Most precise
* Less stable

### LIME:

* Competitive sparsity
* Computationally expensive

### Integrated Gradients:

* Balanced but rarely dominant

---

# Dataset 2 Findings

## Major discovery:

**ResNet-18 collapsed to majority-class prediction.**

### Cause:

* Small dataset
* Class imbalance
* Frozen-layer limitations

### Practical lesson:

**Architecture performance can fail dramatically under scale changes.**

---

## Best Classifier:

* VGG-16 (95.29%)

---

# Cross-Dataset Insights

## Most consistent finding:

**Featured galaxies are significantly easier to explain than Smooth galaxies.**

### Deletion AUC:

* Featured: 0.4306
* Smooth: 0.6470

### Interpretation:

Spiral arms and visible structure provide localized decision features.

---

# Final Conclusions

## Core Finding:

**No single XAI method universally dominates.**

### Recommended practical strategy:

### Use:

* Grad-CAM for stable broad explanations
* GradientSHAP for precise attribution

### Avoid:

* Relying on one XAI method alone
* Assuming findings generalize across datasets

---

# Socially Responsible AI Contributions

This project directly supports responsible AI by emphasizing:

* Transparency
* Interpretability
* Cross-domain robustness
* Honest failure reporting
* Scientific trustworthiness

---

# Practical Recommendations for Researchers

* Benchmark multiple architectures
* Compare multiple XAI methods
* Use multiple faithfulness metrics
* Validate statistically
* Test across datasets
* Report failures honestly

---

# Future Work

* Multi-class morphology
* Transformer-based vision models
* Self-supervised astronomical embeddings
* More survey datasets
* Human astronomer evaluation
* Calibration-aware XAI

---

# Authors

Socially Responsible AI Project
Gargi Sathe
Archit Rathod
University of Illinois Chicago

---

# Final Message

As AI becomes central to scientific discovery, accuracy alone is not enough.

**We must build systems that are not only powerful, but understandable, transparent, and worthy of trust.**
