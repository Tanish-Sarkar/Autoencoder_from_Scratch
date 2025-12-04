# **Autoencoder-Based Anomaly Detection (Credit Card Fraud Dataset)**

## 🚀 **Project Overview**

This project implements an **Autoencoder-based Anomaly Detection system** using the **Credit Card Fraud Detection** dataset.
The model is trained **only on normal transactions** and learns to reconstruct them with low error. Fraud samples (anomalies) produce a **high reconstruction error**, allowing the system to detect anomalies effectively.

---

## 🎯 **Objectives**

* Build a fully functional **encoder–decoder (Autoencoder)** from scratch.
* Learn **representation learning** and anomaly detection fundamentals.
* Use **reconstruction error** as the anomaly score.
* Evaluate with:

  * ROC-AUC
  * Precision, Recall, F1
  * Confusion matrix
* Provide an **inference pipeline** to score any transaction.

---

## 📂 **Project Structure**

```
project-autoencoder/
  ├── data/
  │     └── creditcard.csv
  ├── models/
  │     ├── autoencoder_best.pth
  │     ├── scaler.pkl
  │     ├── X_test_scaled.npy
  │     └── y_test.npy
  ├── notebooks/
  │     ├── 01_eda_and_preprocessing.ipynb
  │     ├── 02_training_logs_and_visuals.ipynb
  │     └── 03_anomaly_demo.ipynb
  ├── src/
  │     ├── model.py
  │     ├── train.py
  │     ├── eval.py
  │     └── inference.py
  ├── requirements.txt
  └── README.md
```

---

## 🧠 **What is an Autoencoder?**

An Autoencoder is a neural network with two parts:

1. **Encoder** → Compress input to a small latent vector
2. **Decoder** → Reconstruct input from latent

When trained only on **normal** data:

* Normal samples → **low reconstruction error**
* Fraud samples → **high reconstruction error**

This makes it perfect for anomaly detection.

---

## 📦 **Installation**

### 1. Clone the repository

```
git clone https://github.com/Tanish-Sarkar/Autoencoder_from_Scratch
cd Autoencoder_from_Scratch
```

### 2. Create and activate a virtual environment

**Windows:**

```
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**

```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install requirements

```
pip install -r requirements.txt
```

---

## 📊 **Training the Autoencoder**

### Option 1: Train via notebook

Open:

```
notebooks/02_training_logs_and_visuals.ipynb
```

and run all cells.

### Option 2: Train via script

Run:

```
python -m src.train
```

This will save:

* Best model → `models/autoencoder_best.pth`
* Scaler → `models/scaler.pkl`
* Test data → `models/X_test_scaled.npy`, `models/y_test.npy`

---

## 🧪 **Evaluation**

Run:

```
python -m src.eval
```

This prints:

* ROC AUC
* Best threshold (based on F1)
* Confusion matrix
* Precision, recall, F1
* Number of predicted anomalies

### 📈 Example Output

```
ROC AUC: 0.9470
Best F1 threshold: 0.588957
Confusion Matrix:
[[56768    95]
 [  129   363]]
F1 Score (fraud class): 0.7642
```

---

## ⚡ **Inference (Score New Samples)**

Run:

```
python -m src.inference
```

This demo:

* Loads trained model
* Scores a few test samples
* Outputs reconstruction error and predicted anomaly flag

### Use inside other Python code:

```python
from src.inference import score_samples
import numpy as np

sample = np.array([...])  # shape: (30,)
errors, flags = score_samples(sample)
print("Error:", errors[0])
print("Anomaly:", flags[0])
```

---

## 📑 **Key Results**

* **ROC AUC:** ~0.94
* **Fraud F1 Score:** ~0.76
* **High precision + strong recall** for anomaly class
* **Very low false positive rate**
* Strong evidence Autoencoder learned normal patterns well

These results align with literature on credit card anomaly detection.

---

## 🧱 **Model Architecture**

### Encoder

* Linear → 128 → ReLU
* Linear → 64 → ReLU
* Linear → latent_dim

### Decoder

* Linear → 64 → ReLU
* Linear → 128 → ReLU
* Linear → input_dim

---

## 📘 **Tech Stack**

* Python
* PyTorch
* NumPy / Pandas
* scikit-learn
* Matplotlib / Seaborn
* Jupyter Notebooks

---

## 🏁 **Next Steps / Future Work**

* Add **Variational Autoencoder (VAE)**
* Add **latent space visualization (t-SNE, UMAP)**
* Wrap inference in **FastAPI**
* Deploy endpoint for real-time anomaly scoring
* Train deeper AE for better fraud recall

---

## 👤 **Author**

**Tanish Sarkar**
(Data Science & Machine Learning Enthusiast)

