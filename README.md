

# Image Classification with Deep Learning 🖼️

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-yellow)](https://www.tensorflow.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)  
[![Stars](https://img.shields.io/github/stars/yourusername/image-classification-cnn)](https://github.com/yourusername/image-classification-cnn)

## 📋 Project Overview
This repository demonstrates **Image Classification using Convolutional Neural Networks (CNNs)** built with **TensorFlow** and **Keras**. We train and evaluate models on three popular datasets:

- **MNIST**: Handwritten digits (10 classes, grayscale images) – A classic starter for CNNs.
- **CIFAR-10**: Small color images of everyday objects (10 classes like airplanes, cats, and cars).
- **Cats vs Dogs**: Binary classification of cat vs. dog photos (from Kaggle) – Real-world application with data augmentation.

**Goals**:
- Build, train, and evaluate basic CNN models.
- Visualize training history and predictions.
- Achieve high accuracy: ~99% on MNIST, ~75% on CIFAR-10, ~88% on Cats vs Dogs (basic models).

Perfect for beginners in Machine Learning/Deep Learning. Run locally in VS Code or Google Colab for GPU acceleration.

## 🛠️ Tech Stack
- **Language**: Python 3.8+
- **Core Libraries**: TensorFlow/Keras (CNNs), NumPy (data handling), Matplotlib (visualization)
- **IDE/Environment**: VS Code with Python extension; Virtual environment (venv)
- **Hardware**: CPU/GPU supported (GPU recommended for faster training)

## 📁 Project Structure
```
image-classification-cnn/
├── README.md                 # This file (or upload README.docx)
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── main.py                   # Unified script: Run all datasets with menu
├── data/                     # Datasets folder
│   └── cats_dogs/            # Kaggle dataset (optional for binary classification)
│       ├── train/            # ~20k images (cats/ and dogs/ subfolders)
│       └── test/             # ~10k images (cats/ and dogs/ subfolders)
└── models/                   # Saved models (auto-generated)
    ├── mnist_model.h5
    ├── cifar_model.h5
    └── catsdogs_model.h5
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ installed.
- GitHub account (to clone/fork this repo).

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/image-classification-cnn.git
cd image-classification-cnn
```

### 2. Setup Virtual Environment
```bash
# Create venv
python -m venv venv

# Activate (Windows: venv\Scripts\activate | macOS/Linux: source venv/bin/activate)

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Datasets
- **MNIST & CIFAR-10**: Automatically loaded via Keras (no manual download needed).
- **Cats vs Dogs** (optional, for binary task):
  - Download from [Kaggle: Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) (~800MB ZIP).
  - Extract `train.zip` to `data/cats_dogs/train/` (create subfolders `cats/` and `dogs/` with ~10k images each).
  - Extract `test1.zip` to `data/cats_dogs/test/` (similar structure).

### 4. Run the Project
```bash
# Run the unified script (interactive menu for dataset choice)
python main.py
```
- Select 1 (MNIST), 2 (CIFAR-10), or 3 (Cats vs Dogs).
- **Training Time**: MNIST (~5-10 min), CIFAR-10 (~20-30 min), Cats vs Dogs (~1-2 hours on CPU).
- **Output**: Console logs for progress, accuracy; Pop-up plots for history and predictions.
- Models auto-save to `models/` folder.

### 5. Alternative: Jupyter Notebook
For interactive exploration:
- Install: `pip install jupyter`
- Run: `jupyter notebook`
- Create a new notebook and copy code from `main.py`.

## 📊 Results & Visualization
Each run generates:
- **Training Plots**: Accuracy/Loss curves over epochs.
- **Prediction Samples**: Grid of 9 test images with labels (green = correct, red = wrong).

Example Output (Console):  
**MNIST Test Accuracy: 0.9923**

*(In Word: Insert placeholders for images here – e.g., right-click > Insert Picture for screenshots of plots.)*

**Accuracy Plot Example:**  
[Placeholder: Insert image of accuracy vs. loss curves]

**Predictions Example:**  
[Placeholder: Insert image of sample predictions grid]

## 🔧 Customization & Improvements
- **Hyperparameters**: Edit `epochs`, `batch_size`, or add layers in `build_cnn()` function.
- **Data Augmentation**: Already included for Cats vs Dogs; extend to others using `ImageDataGenerator`.
- **Advanced Models**: Integrate transfer learning (e.g., ResNet50):  
  ```python
  from tensorflow.keras.applications import ResNet50
  base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
  # Freeze base and add custom head
  ```
- **Extensions**:
  - Deploy as a web app with Streamlit/Flask.
  - Convert to TensorFlow Lite for mobile.
  - Add confusion matrix: Use `sklearn.metrics.confusion_matrix`.

## 📝 Dependencies
Create `requirements.txt`:  
```
tensorflow==2.15.0
matplotlib==3.8.0
numpy==1.24.3
```
Install: `pip install -r requirements.txt`

## 🐛 Troubleshooting
| Issue                  | Solution                                                                 |
|------------------------|--------------------------------------------------------------------------|
| **ImportError**        | Activate venv: `source venv/bin/activate` (or equivalent).                |
| **Dataset Not Found**  | Re-download Cats vs Dogs; ensure folder structure matches.               |
| **Out of Memory**      | Reduce `batch_size` to 16 or use Google Colab (free GPU).                |
| **Slow Training**      | Switch to GPU: Check with `nvidia-smi`; or use Colab.                    |
| **Plots Not Visible**  | Run in Jupyter or add `plt.show(block=True)`.                            |
| **Kaggle Download**    | Install Kaggle CLI: `pip install kaggle`; then `kaggle datasets download -d ...`. |

