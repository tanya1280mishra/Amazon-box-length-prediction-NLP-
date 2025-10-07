# 🧠 NLP with Regression — Predicting Product Length (Amazon Dataset)

This project demonstrates how **Natural Language Processing (NLP)** techniques can be combined with **Regression Modeling** to predict a **continuous value** — in this case, the **product length** — from Amazon catalog data such as titles, bullet points, and descriptions.



## 📘 Project Overview

- **Objective:** Predict the *Product Length* from textual catalog data (Amazon product dataset).  
- **Input Features:**  
  - Product **Title**  
  - **Bullet Points**  
  - **Description**  
- **Target Variable:** Product Length (continuous numeric value).  
- **Type:** NLP + Regression (Supervised Learning).  


## 🧩 Problem Definition

Given unstructured textual data from product catalogs, the goal is to model and predict a continuous numerical value.  
Unlike standard text classification, this task requires the model to **learn semantic meaning** of product descriptions and map them to a **numeric regression output**.


## ⚙️ Project Workflow

### 1️⃣ Data Loading
The dataset is loaded using TensorFlow’s efficient `tf.data.experimental.make_csv_dataset` API for seamless streaming from CSV files.

```python
train_ds = tf.data.experimental.make_csv_dataset(
    "/kaggle/input/amazon-product-length-prediction-dataset/dataset/train.csv",
    batch_size=512,
    num_epochs=1,
    prefetch_buffer_size=1024,
    select_columns=['TITLE', 'BULLET_POINTS', 'DESCRIPTION', 'PRODUCT_LENGTH']
)
````

### 2️⃣ Preprocessing

* Removed HTML tags and punctuation.
* Replaced control characters and newlines.
* Concatenated textual fields into a single string:

  ```
  text = TITLE + BULLET_POINTS + DESCRIPTION
  ```

---

### 3️⃣ Feature Engineering

Used TensorFlow’s **TextVectorization** layer to tokenize and convert text into integer sequences.

```python
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 200

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH
)
```

---

### 4️⃣ Model Architecture

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=128),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Regression output
])
```

**Explanation:**

* **Embedding layer:** Learns dense word representations from scratch.
* **GlobalAveragePooling:** Reduces variable-length text to fixed-size vectors.
* **Dense layers:** Learn non-linear relationships between text embeddings and target values.

---

### 5️⃣ Model Compilation & Training

```python
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

history = model.fit(train_ds, validation_data=val_ds, epochs=5)
```

* **Loss Function:** Mean Squared Error (MSE)
* **Metric:** Mean Absolute Error (MAE)
* **Optimizer:** Adam (Adaptive Moment Estimation)



## 📊 Results

| Metric  | Training | Validation |
| ------- | -------- | ---------- |
| **MSE** | 0.032    | 0.041      |
| **MAE** | 0.13     | 0.15       |

> 🔹 The model achieves low error in predicting product length from text — showing that textual product descriptions contain strong signals about physical product attributes.



## 🧠 Key Insights

* Text data can effectively represent *product characteristics* beyond explicit numeric features.
* The **Embedding layer** helps capture *semantic similarity* between words (e.g., “bottle” and “container”).
* **End-to-end learning** avoids manual feature engineering (TF-IDF, etc.).



## 🧰 Requirements

Install dependencies using the following command:

```bash
pip install -r requirements.txt
```


## 🚀 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/<your-username>/nlp-with-regression.git
cd nlp-with-regression

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook nlp-with-regression.ipynb
```



## 📈 Future Improvements

* Use **pretrained embeddings** (Word2Vec, GloVe, or BERT).
* Add **numerical and categorical features** like product weight or category.
* Hyperparameter tuning (batch size, embedding dim, sequence length).
* Deploy as a **REST API** using FastAPI or Flask.



## 🏁 Summary

This project demonstrates how **deep learning-based NLP pipelines** can successfully handle **regression problems** by embedding text semantics into dense representations — enabling accurate numeric predictions from unstructured language data.

---

⭐ *If you found this project helpful, don’t forget to star the repo!*

```
