# 🏨 Hotel Review Sentiment Analysis
### Bidirectional GRU with Attention Mechanism

> Classifying TripAdvisor hotel reviews as **Positive** or **Negative** using a
> custom deep learning architecture combining Bidirectional GRU and a Global
> Attention Mechanism with Word2Vec-style embeddings.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.x-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-9cf?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-93.12%25-brightgreen?style=for-the-badge)

---

## 📌 Overview

Online hotel reviews are a goldmine of customer sentiment — but manually reading
thousands of them is impossible. This project builds an end-to-end deep learning
NLP pipeline that **automatically classifies hotel reviews** as positive or
negative with **93%+ accuracy**.

The architecture goes beyond standard LSTM/GRU models by incorporating a
**Global Attention Mechanism**, which forces the model to focus on the most
emotionally significant words in each review — words like *"terrible"*,
*"amazing"*, *"dirty"*, or *"wonderful"* — rather than treating all tokens
equally.

📥 **Dataset:** [TripAdvisor Hotel Reviews — Kaggle](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews) — 20,000 reviews

---

## 🏆 Results

| Metric | Score |
|--------|-------|
| ✅ Accuracy | **93.12%** |
| 📈 ROC-AUC | **93.32%** |
| 🎯 Precision-Recall AUC | **95.28%** |

---

## ✨ Key Features

- 🔤 **Full NLP Preprocessing Pipeline** — HTML stripping, contraction expansion,
  stopword removal, tokenization
- ⚖️ **Class Imbalance Handling** — upsampling minority class via `resample`
- 🧠 **Bidirectional GRU** — captures context from both directions in text
- 🎯 **Custom Attention Layer** — learns to weight important words dynamically
- 📊 **Full Evaluation Suite** — Confusion Matrix, ROC-AUC, Precision-Recall curve

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Deep Learning** | TensorFlow 2.x, Keras 3.x |
| **NLP** | NLTK, BeautifulSoup, contractions |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook |

---

## 🏗️ Model Architecture
---

## 🔄 Pipeline Walkthrough

### 1️⃣ Data Loading & Labelling
- Load 20,000 TripAdvisor reviews with star ratings (1–5)
- Map ratings → binary sentiment:
  - ⭐ 1–2 → `negative`
  - ⭐ 4–5 → `positive`
  - ⭐ 3 → dropped (neutral — ambiguous signal)

### 2️⃣ Text Preprocessing
### 3️⃣ Handling Class Imbalance
- Upsample the minority class (`negative`) to match majority (`positive`)
- Ensures the model doesn't develop bias toward the dominant class

### 4️⃣ Tokenization & Padding
- Fit `Tokenizer` on training data with `<UNK>` token for unknown words
- Pad/truncate all sequences to `MAX_SEQUENCE_LENGTH = 100`

### 5️⃣ Model Training
- **Optimizer:** Adam (`lr=0.001`)
- **Loss:** Binary Cross-Entropy
- **EarlyStopping** on `val_loss` (patience=10)
- **ModelCheckpoint** saves best weights automatically

### 6️⃣ Evaluation
- Classification Report (Precision, Recall, F1)
- Confusion Matrix heatmap
- ROC-AUC curve
- Precision-Recall curve

---

## 📊 Model Performance Curves

The model tracks both **training and validation** accuracy/loss across epochs,
with early stopping preventing overfitting.

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/Anindya-Arnob/hotel-sentiment-bigru-attention
cd hotel-sentiment-bigru-attention

# 2. Install dependencies
pip install tensorflow keras nltk pandas numpy matplotlib seaborn scikit-learn
pip install beautifulsoup4 contractions tqdm

# 3. Download NLTK data (run once)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 4. Launch Jupyter
jupyter notebook bi-gru-sentiment-analysis-FIXED.ipynb
```

> 💡 Place `tripadvisor_hotel_reviews.csv` in the same folder as the notebook.

---

## 📁 Project Structure
---

## 🔑 Key Deep Learning Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| **Tokenization** | Keras Tokenizer with `<UNK>` and `<PAD>` tokens |
| **Word Embeddings** | Learned Embedding layer (dim=300) |
| **Bidirectional RNN** | BiGRU reads sequence forward + backward |
| **Attention Mechanism** | Custom Global Attention Layer with learned weights |
| **Transfer of Context** | GRU hidden states passed into attention |
| **Regularization** | Dropout layers to prevent overfitting |
| **Imbalance Handling** | Oversampling with `sklearn.utils.resample` |

---

## 💡 Why BiGRU + Attention?

Standard GRU reads text in **one direction only** — it may miss context from
later words that clarify earlier ones. Bidirectional GRU solves this by reading
**both forward and backward**.

But even BiGRU treats all hidden states equally. The **Attention Layer** adds
a learned weighting — the model assigns **higher importance to emotionally
charged words** and lower importance to filler words, producing a richer
context vector for classification.

---

## 🔮 Future Improvements

- [ ] Use pre-trained GloVe or FastText embeddings
- [ ] Experiment with Multi-Head Self-Attention (Transformer-style)
- [ ] Add cross-validation to reduce overfitting
- [ ] Deploy as a REST API with FastAPI or Flask
- [ ] Extend to multi-class rating prediction (1–5 stars)

---

## 📜 License

This project is open-source. For inquiries or contributions,
please open an issue in the repository.

---

⭐ **If this project helped you, please give it a star!**
