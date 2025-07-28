# 📦 Amazon Product Length Prediction Using LSTM

**Predicting box sizes from product descriptions using Deep Learning**

This project explores the use of **LSTM (Long Short-Term Memory)** networks to estimate the **physical length of Amazon product packages** based on textual information such as **titles, bullet points, and descriptions**.

> ⚠️ *Note: Current results are not optimal. This project is a work in progress aimed at improving accuracy through better data, preprocessing, and model tuning.*

---

## 🧭 Table of Contents

* [📊 Dataset](#-dataset)
* [🧼 Preprocessing](#-preprocessing)
* [🧠 Model Architecture](#-model-architecture)
* [📈 Results](#-results)
* [🚀 Future Improvements](#-future-improvements)
* [⚙️ Setup & Installation](#️-setup--installation)
* [▶️ Usage](#️-usage)

---

## 📊 Dataset

The dataset contains **2.2 million** training samples and **734k** test entries with the following features:

| Column            | Description                                           |
| ----------------- | ----------------------------------------------------- |
| `PRODUCT_ID`      | Unique identifier                                     |
| `TITLE`           | Product name                                          |
| `BULLET_POINTS`   | Feature highlights                                    |
| `DESCRIPTION`     | Detailed product description                          |
| `PRODUCT_TYPE_ID` | Categorical product ID                                |
| `PRODUCT_LENGTH`  | (Target) Length of product box (only in training set) |

**Missing Data (Train):**

* `TITLE`: 13
* `BULLET_POINTS`: 837,366
* `DESCRIPTION`: 1,157,382

> All rows with missing values are dropped before training.

---

## 🧼 Preprocessing

To clean and standardize the text data, we apply:

1. **Drop NaNs** from critical text columns
2. **Lowercasing** all text
3. **Remove HTML tags**, emojis, URLs, and punctuation
4. **Expand contractions** (e.g., "don't" → "do not")
5. **Concatenate** `BULLET_POINTS` + `DESCRIPTION` → `text`
6. **Truncate** dataset to 458 rows (for quick experimentation)

> ⚠️ Limited samples were used due to hardware constraints; full dataset training is planned.

---

## 🧠 Model Architecture

We use a **Sequential LSTM** model built with TensorFlow:

| Layer       | Details                                          |
| ----------- | ------------------------------------------------ |
| `Embedding` | (88,636 vocab, 200 dims, input\_len=507)         |
| `LSTM x2`   | 256 units each, one with `return_sequences=True` |
| `Dense`     | 1 unit, `linear` activation for regression       |

**Compilation:**

* `Loss`: Mean Squared Error (MSE)
* `Optimizer`: Adam
* `Metrics`: MAE, RMSE

---

## 📈 Results

After training for 100 epochs on the mini dataset:

| Metric       | Value          |
| ------------ | -------------- |
| **MSE**      | `6,901,454.45` |
| **RMSE**     | `2,627.06`     |
| **MAE**      | `6,901,454.45` |
| **R² Score** | `-30.0152` ❌   |

> Model output was constant (`~283.73`) indicating **no meaningful learning** on this sample size.

---

### 📉 Visualizations

* 📍 **Actual vs Predicted**: Scatter plot with dashed perfect-fit line
* 📊 **Error Distribution**: Histogram of prediction errors
* 📦 **Predicted Lengths**: Histogram of output distribution

---

## 🚀 Future Improvements

Here’s how we plan to improve the model:

### 🔁 Data

* Use **full dataset** (2.2M rows)
* Handle **missing values** more smartly (e.g., imputation)

### 🧪 Preprocessing

* Add **stop word removal**, **lemmatization**, **POS tagging**
* Use **TF-IDF** or **custom tokenizer**

### 🔧 Features

* Include `PRODUCT_TYPE_ID` (categorical encoding)
* Add **text length** and **readability** metrics

### 🤖 Modeling

* Use **Bidirectional LSTM**, **GRU**, or **Transformer-based models** (like BERT)
* Add **Attention layers**
* Try **pretrained embeddings** (Word2Vec, GloVe, FastText)

### 📉 Regularization

* Add **Dropout**, **L2 regularization**

### 🧮 Optimization

* Tune **hyperparameters** (LR, batch size, embedding dims)

---

## ⚙️ Setup & Installation

### 1. Clone the repo

```bash
git clone <repo-url>
cd <repo-dir>
```

### 2. Create and activate environment

```bash
python -m venv LSTMenv
source LSTMenv/bin/activate  # or LSTMenv\Scripts\activate on Windows
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

### 4. Prepare dataset

Place your `train.csv` and `test.csv` under:

```
project_root/
├── LSTMmodel.ipynb
└── dataset/
    ├── train.csv
    └── test.csv
```

Update file paths in the notebook if necessary.

---

## ▶️ Usage

1. Launch Jupyter:

   ```bash
   jupyter notebook LSTMmodel.ipynb
   ```
2. Run all cells to:

   * Load and preprocess the dataset
   * Train the LSTM model
   * Evaluate performance and visualize predictions

---

## 📌 Project Status

🔧 **Under active development** — working on full dataset training, feature engineering, and integration of transformer models for better performance.

---

## ✨ Contributions

Pull requests, ideas, and suggestions are welcome!
