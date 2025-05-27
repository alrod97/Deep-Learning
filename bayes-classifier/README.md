# Bayes Classifier with DINO Features and PCA

A Python implementation of a Bayes classifier for image classification using DINO (self-DIstillation with NO labels) features and Principal Component Analysis (PCA) for dimensionality reduction.

## 🎯 Overview

This project implements a Bayes classifier that:
- Extracts high-quality image features using Facebook's DINOv2-giant model
- Applies PCA for dimensionality reduction while preserving information
- Classifies images into K classes using Gaussian distributions
- Achieves >96% test accuracy on car and vehicle classification tasks

📚 **Complete handwritten notes** detailing the mathematical derivations and implementation decisions are included in the repository.

## 📐 Theory

📚 **[Full Handwritten Notes Available (PDF)](./Bayes-Classification-Notes.pdf)** - Detailed mathematical derivations and implementation notes

### Bayes Classification

The classifier is based on Bayes' theorem. For a data point **x**, we assign it to class k that maximizes the posterior probability:

```
k* = argmax P(y=k|x) = argmax P(x|y=k)P(y=k)
      k∈C                k∈C
```

Where:
- `P(y=k|x)` is the posterior probability
- `P(x|y=k)` is the class-conditional probability (likelihood)
- `P(y=k)` is the prior probability of class k

### Gaussian Assumption

We model each class with a multivariate Gaussian distribution:

```
P(x|y=k) = 1/((2π)^(d/2)|Σₖ|^(1/2)) exp(-1/2(x-μₖ)ᵀΣₖ⁻¹(x-μₖ))
```

Where:
- `μₖ` is the mean vector for class k
- `Σₖ` is the covariance matrix for class k
- `d` is the feature dimension

### Decision Rule

To avoid numerical instability, we work with log probabilities:

```
k* = argmax [-1/2 log|Σₖ| - 1/2(x-μₖ)ᵀΣₖ⁻¹(x-μₖ) + log P(y=k)]
      k∈C
```

## 📝 Detailed Mathematical Notes

For a complete understanding of the theory and implementation, see the **[handwritten notes (PDF)](./Bayes-Classification-Notes.pdf)** which cover:

- **Pages 1-2**: One-hot encoding and matrix formulations for K classes
- **Pages 3-4**: Posterior probability derivation and Gaussian conditional distributions
- **Pages 5-6**: Multi-variate Gaussian formula and log-probability transformations
- **Page 7**: Feature extraction with DINO and dimensionality challenges
- **Page 8**: PCA for compression and numerical stability

These notes bridge the gap between the mathematical theory and the actual Python implementation.

## 🚀 Features

- **DINO Feature Extraction**: Uses DINOv2-giant model for powerful visual representations
- **PCA Dimensionality Reduction**: Reduces feature dimensions from 1536 to 50 while preserving variance
- **Numerical Stability**: Adds regularization (ε=1e-6) to covariance matrices
- **One-Hot Encoding**: Efficient matrix operations for multi-class classification
- **Modular Design**: Separate modules for feature extraction and classification

## 📁 Project Structure

```
.
├── bayes.py                        # Core Bayes classifier implementation
├── extract_features.py             # DINO feature extraction script
├── BayesClassifier.ipynb          # Example usage notebook
├── Bayes-Classification-Notes.pdf  # Handwritten theory & implementation notes
├── dataset/                       # Dataset directory
│   ├── train/                    # Training images
│   │   ├── class1/
│   │   ├── class2/
│   │   └── ...
│   └── test/                     # Test images
│       ├── class1/
│       ├── class2/
│       └── ...
└── README.md
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bayes-classifier-dino.git
cd bayes-classifier-dino
```

2. Install dependencies:
```bash
pip install numpy torch torchvision transformers pillow scikit-learn pathlib
```

## 📊 Usage

### 1. Extract DINO Features

First, extract features from your image dataset:

```bash
python extract_features.py path/to/dataset
```

This will:
- Process all images in each class folder
- Extract 1536-dimensional DINO features
- Save features as `.npy` files in `features_dinov2-giant/` subdirectories

### 2. Train and Evaluate the Classifier

```python
from bayes import BayesClassifier, load_features_and_labels, one_hot_encode
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score

# Load data
X_train, y_train, label_map = load_features_and_labels("dataset/train")
X_test, y_test, _ = load_features_and_labels("dataset/test")

# Apply PCA
pca = PCA(n_components=50, random_state=0)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Convert labels to one-hot encoding
k = len(label_map)
y_train_oh = one_hot_encode(y_train, k)

# Train classifier
clf = BayesClassifier(k=k, X=X_train_pca, y=y_train_oh)
clf.fit()

# Make predictions
y_pred_train = clf.predict(X_train_pca)
y_pred_test = clf.predict(X_test_pca)

# Evaluate
print("📊 Train Accuracy:", accuracy_score(y_train, y_pred_train))
print("📊 Test Accuracy:", accuracy_score(y_test, y_pred_test))
```

### 3. Check Results

See `BayesClassifier.ipynb` for a complete example with performance metrics.

## 🎯 Performance

On the example car/vehicle dataset with 8 classes:
- **Train Accuracy**: 98.87%
- **Test Accuracy**: 96.82%

### Per-class Test Recall:
```
🔍 Class 'Audi' recall: 1.000
🔍 Class 'Hyundai Creta' recall: 0.970
🔍 Class 'Mahindra Scorpio' recall: 0.987
🔍 Class 'Rolls Royce' recall: 0.959
🔍 Class 'Swift' recall: 0.961
🔍 Class 'Tata Safari' recall: 0.972
🔍 Class 'Toyota Innova' recall: 0.979
🔍 Class 'airplane' recall: 0.922
```

## 🔧 Implementation Details

### Feature Extraction (`extract_features.py`)
- Uses DINOv2-giant model from Facebook
- Extracts CLS token features (1536-dimensional)
- L2 normalizes features for better performance
- Supports various image formats (jpg, png, etc.)

### Bayes Classifier (`bayes.py`)
- **Prior Estimation**: Computed as class frequencies
- **Mean Estimation**: Sample mean per class
- **Covariance Estimation**: Sample covariance with regularization
- **Prediction**: Vectorized implementation for efficiency

### Key Parameters
- PCA Components: 50 (reduces from 1536 dimensions)
- Regularization: ε = 1e-6 (prevents singular covariance matrices)
- Model: facebook/dinov2-giant

## 📈 Mathematical Formulation

### Prior Probability
```
P(y=k) = Nₖ/N
```
where Nₖ is the number of samples in class k.

### Mean Estimation
```
μₖ = (1/Nₖ) Σᵢ xᵢ  for all i where yᵢ=k
```

### Covariance Estimation
```
Σₖ = (1/(Nₖ-1)) Σᵢ (xᵢ-μₖ)(xᵢ-μₖ)ᵀ + εI
```
where ε is the regularization parameter and I is the identity matrix.

## 🔍 Notes

- The classifier assumes Gaussian distributions for each class
- PCA helps with numerical stability and reduces computation
- DINO features provide robust visual representations without requiring labeled pretraining data
- Regularization prevents singular covariance matrices
- The feature directory structure follows: `class_name/features_dinov2-giant/class_name_image_name.npy`

## 🚀 Future Work

- Experiment with different numbers of PCA components
- Try other pooling strategies (mean pooling vs CLS token)
- Implement LDA (Linear Discriminant Analysis) as alternative to PCA
- Add cross-validation for hyperparameter tuning
- Support for incremental learning with new classes

## 📝 References

- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- [Pattern Recognition and Machine Learning - Christopher Bishop](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
