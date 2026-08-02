# 🛡️ Comment Toxicity Detection using Deep Learning

A complete end-to-end deep learning project that detects and classifies toxic comments using NLP models (RNN, LSTM, BERT), deployed as an interactive Streamlit web application.

---

## 📌 Problem Statement

Online communities and social media platforms have become integral parts of modern communication. However, the prevalence of toxic comments — including harassment, hate speech, and offensive language — poses significant challenges to maintaining healthy online discourse.

This project builds an automated deep learning system capable of detecting and classifying toxic comments in real-time across 6 toxicity categories. The system assists platform moderators in taking appropriate actions to mitigate toxic behavior.

---

## 🎯 Objective

- Build and compare multiple deep learning models (RNN, LSTM, BERT) for multi-label text classification
- Handle real-world challenges like class imbalance in NLP datasets
- Deploy the best performing model as an interactive Streamlit web application
- Support both single comment analysis and bulk CSV predictions

---

## 💼 Business Use Cases

1. **Social Media Platforms** — Automatically detect and filter toxic comments in real-time
2. **Online Forums** — Moderate user-generated content efficiently
3. **Content Moderation Services** — Enhance moderation capabilities
4. **Brand Safety** — Ensure advertisements appear in safe environments
5. **E-learning Platforms** — Create safer online learning environments
6. **News Websites** — Moderate reader comments on articles

---

## 📊 Dataset

- **Source:** Jigsaw Toxic Comment Classification Challenge (Kaggle)
- **Size:** 159,571 Wikipedia comments
- **Labels:** 6 toxicity categories (multi-label classification)
- **Key Challenge:** Heavy class imbalance (~90% non-toxic)

### Label Distribution
| Label | Count | Percentage |
|-------|-------|------------|
| toxic | 15,294 | 9.58% |
| severe_toxic | 1,595 | 1.00% |
| obscene | 8,449 | 5.29% |
| threat | 478 | 0.30% |
| insult | 7,877 | 4.94% |
| identity_hate | 1,405 | 0.88% |

---

## 🗂️ Project Structure

CommentToxicity/
├── app.py # Streamlit web application (BERT inference)
├── CommentToxicity.ipynb # Complete training notebook (Google Colab)
├── models/ # Saved model files
│ └── bert_model.pt # Fine-tuned BERT model weights
├── sample_comments.csv # Sample CSV for bulk prediction testing
└── README.md # Project documentation


---

## 🔧 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| Deep Learning | PyTorch, TensorFlow/Keras |
| NLP/Transformers | HuggingFace Transformers (BERT) |
| Web App | Streamlit |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Training Environment | Google Colab (T4 GPU) |
| Development | VS Code |
| Version Control | GitHub |

---

## 📈 Complete Project Workflow

### Phase 1 — Environment Setup (Google Colab)

The project was trained on Google Colab using a free T4 GPU for faster training. Google Drive was mounted to save models permanently across sessions.

```python
from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

### Phase 2 — Data Loading & Exploratory Data Analysis

The Jigsaw dataset was loaded and explored to understand its structure and identify key challenges.

```python
df = pd.read_csv('/content/drive/MyDrive/CommentToxicity/train.csv')
print("Shape:", df.shape)
# Output: (159571, 8)
```

**Key findings from EDA:**
- No missing values across all 159,571 rows
- Labels already encoded as integers (0 or 1)
- Severe class imbalance — ~90% of comments are non-toxic
- `threat` and `identity_hate` are extremely rare (0.30% and 0.88%)
- Accuracy is a misleading metric — F1 score must be used instead

### Phase 3 — Text Preprocessing

Raw comments were cleaned and converted into numerical sequences for model input.

**Steps:**
1. **Text Cleaning** — lowercase, remove HTML tags, URLs, punctuation, numbers
2. **Tokenization** — convert words to unique integer IDs (top 20,000 words)
3. **Padding** — standardize all sequences to length 200
4. **Train/Test Split** — 80% training (127,656), 20% testing (31,915)

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)       # Remove HTML
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)         # Remove numbers
    return text.strip()

# Tokenize and pad
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])
X = pad_sequences(sequences, maxlen=200, padding='post')
y = df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values
```

### Phase 4 — Model Development & Training

Three architectures were trained and compared:

---

#### Model 1 — Simple RNN

The simplest recurrent architecture. Reads text sequentially, word by word, maintaining a hidden state.

```python
model = Sequential([
    Embedding(20000, 32),
    SimpleRNN(64, return_sequences=False),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(6, activation='sigmoid')
])
```

**Architecture explanation:**
- `Embedding` — converts word IDs to 32-dimensional vectors
- `SimpleRNN` — reads words sequentially, updates memory at each step
- `Dropout(0.5)` — prevents overfitting by randomly disabling 50% of neurons
- `Dense(6, sigmoid)` — outputs 6 probabilities (one per toxicity label)

**Result:** Macro F1 = 0.09 (poor)
**Why it failed:** Vanishing gradient problem — RNN forgets early words in long comments, missing toxic patterns

---

#### Model 2 — LSTM with Class Weights

LSTM addresses RNN's forgetting problem using 3 gates (forget, input, output) to selectively remember important information.

**Key improvement — Class Weights:**
Since 90% of data is non-toxic, we computed class weights to force the model to pay more attention to rare toxic examples.

```python
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=y_train[:, 0]
)
# Result: {0: 0.55, 1: 5.21}
# Toxic comments penalized 5x more when missed
```

```python
model = Sequential([
    Embedding(20000, 32),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(6, activation='sigmoid')
])
```

**Training with EarlyStopping:**
```python
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=10, batch_size=64,
          validation_split=0.1, callbacks=[early_stop],
          class_weight=class_weight_dict)
```

**Result:** Macro F1 = 0.39 (significant improvement)
**Why it improved:** LSTM remembered toxic patterns better + class weights forced attention on minority classes

---

#### Model 3 — BERT (bert-base-uncased) ✅ Best Model

BERT (Bidirectional Encoder Representations from Transformers) is a pretrained transformer model that reads text in both directions simultaneously.

**Why BERT is different:**
- **Pretrained** on 3.3 billion words (Wikipedia + BookCorpus) — already understands English
- **Bidirectional** — understands context from both left and right simultaneously
- **Attention mechanism** — focuses on the most important words

```python
class BertToxicClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 6)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output  # [CLS] token summary
        return self.classifier(self.dropout(cls_output))
```

**Training details:**
- Optimizer: AdamW (lr=2e-5)
- Epochs: 3
- Batch size: 16 (smaller due to BERT's memory requirements)
- Training time: ~45-60 minutes on T4 GPU

**Result:** Macro F1 = 0.68 (best performer)
**Why it won:** Pretrained knowledge + bidirectional attention solved rare label detection

---

### Phase 5 — Model Comparison

| Label | RNN F1 | LSTM F1 | BERT F1 |
|-------|--------|---------|---------|
| toxic | 0.13 | 0.78 | **0.83** |
| severe_toxic | 0.00 | 0.13 | **0.51** |
| obscene | 0.21 | 0.75 | **0.84** |
| threat | 0.00 | 0.00 | **0.54** |
| insult | 0.20 | 0.67 | **0.78** |
| identity_hate | 0.00 | 0.00 | **0.57** |
| **Macro F1** | **0.09** | **0.39** | **0.68** |

BERT was the clear winner — especially for rare labels like `threat` and `identity_hate` that LSTM completely missed.

---

### Phase 6 — Streamlit Web Application

The best model (BERT) was deployed as an interactive web application using Streamlit.

**App Features:**
- 💬 Single comment analysis with per-label probability scores
- 📂 Bulk CSV upload for batch predictions
- ⬇️ Download results as CSV
- 📊 Model performance metrics in sidebar

**How the app works:**
1. User enters a comment
2. BERT tokenizer converts it to token IDs + attention mask
3. Fine-tuned BERT model predicts 6 probability scores
4. Scores above 0.5 threshold flagged as toxic
5. Results displayed with color-coded indicators

---

## ⚙️ How to Run Locally

### Prerequisites
- Python 3.10
- Git

### Steps

1. **Clone the repository:**
```bash
git clone https://github.com/CharanrajSridharan/CommentToxicity.git
cd CommentToxicity
```

2. **Create virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install streamlit
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers pandas numpy
```

4. **Download model file:**
- Download `bert_model.pt` separately
- Place it inside a `models/` folder

5. **Run the app:**
```bash
venv\Scripts\python.exe -m streamlit run app.py
```

6. **Open browser:**

http://localhost:8501


---

## 🌟 Key Learnings

1. **Class imbalance is critical** — Accuracy is misleading on imbalanced datasets. F1 score is the right metric. Class weights dramatically improved LSTM recall from near-zero to 0.78 for toxic comments.

2. **Architecture matters** — RNN → LSTM → BERT showed clear progression. Each architecture solved the previous one's weakness.

3. **Pretraining is powerful** — BERT's pretrained knowledge on billions of words gave it ability to detect rare toxicity types (threat, identity_hate) that LSTM completely missed.

4. **Environment consistency** — Model format and library versions must match between training (Colab) and inference (local). Keras version mismatches caused deployment issues.

5. **Real-world NLP is messy** — Extremely rare classes (threat: 0.30%) are very hard to detect even with the best models.

---

## 📋 Project Evaluation Criteria

- ✅ Code written in modular/functional style
- ✅ Public GitHub repository
- ✅ Detailed README with workflow
- ✅ PEP 8 coding standards followed
- ✅ Streamlit web application deployed
- ✅ Demo video posted on LinkedIn

---

## 👨‍💻 Author

**Charanraj Sridharan**
- GitHub: [@CharanrajSridharan](https://github.com/CharanrajSridharan)
- LinkedIn: [Your LinkedIn URL]

---

## 📄 References

- [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)
- [BERT Paper — Devlin et al. 2018](https://arxiv.org/abs/1810.04805)
- [HuggingFace Transformers](https://huggingface.co/bert-base-uncased)
- [Streamlit Documentation](https://docs.streamlit.io)
