import streamlit as st
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import re
import string

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Comment Toxicity Detector",
    page_icon="🛡️",
    layout="wide"
)

# ── Constants ─────────────────────────────────────────────────
LABELS    = ['toxic', 'severe_toxic', 'obscene', 
             'threat', 'insult', 'identity_hate']
BERT_PATH = 'models/bert_model.pt'

# ── Text Cleaning ─────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

# ── BERT Model Class ──────────────────────────────────────────
class BertToxicClassifier(nn.Module):
    def __init__(self):
        super(BertToxicClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 6)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.pooler_output
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)

# ── Load BERT ─────────────────────────────────────────────────
@st.cache_resource
def load_bert():
    with st.spinner("Loading BERT model... this may take a minute"):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertToxicClassifier()
        model.load_state_dict(torch.load(BERT_PATH, map_location='cpu'))
        model.eval()
    return model, tokenizer

# ── BERT Prediction ───────────────────────────────────────────
def predict_bert(text, model, tokenizer):
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(
            encoding['input_ids'],
            encoding['attention_mask']
        )
    preds = torch.sigmoid(outputs).numpy()[0]
    return preds

# ── Load Model ────────────────────────────────────────────────
model, tokenizer = load_bert()

# ── UI ────────────────────────────────────────────────────────
st.title("🛡️ Comment Toxicity Detector")
st.markdown("Detect toxic content in comments using **BERT** — our best performing model (Macro F1: 0.68)")

# Sidebar
st.sidebar.title("📊 Model Info")
st.sidebar.info("**Model:** BERT\n\n**Architecture:** Transformer\n\n**Parameters:** 110M")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Performance")

perf = {
    'Toxic': 0.83,
    'Severe Toxic': 0.51,
    'Obscene': 0.84,
    'Threat': 0.54,
    'Insult': 0.78,
    'Identity Hate': 0.57
}

for label, score in perf.items():
    st.sidebar.markdown(f"**{label}** — `{score}`")
    st.sidebar.progress(score)

st.sidebar.markdown("---")
st.sidebar.success("**Macro F1: 0.68**")

# ── Tab Layout ────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Single Comment", "📂 Bulk CSV"])

# ── Tab 1: Single Comment ─────────────────────────────────────
with tab1:
    st.subheader("Enter a comment to analyze:")
    user_input = st.text_area("Comment", height=150,
                               placeholder="Type your comment here...")

    if st.button("🔍 Analyze", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter a comment first!")
        else:
            with st.spinner("Analyzing..."):
                preds = predict_bert(user_input, model, tokenizer)

            st.markdown("### 🎯 Results")
            cols = st.columns(6)
            for i, (label, score) in enumerate(zip(LABELS, preds)):
                with cols[i]:
                    emoji = "🔴" if score > 0.5 else "🟢"
                    st.markdown(f"**{label.replace('_', ' ').title()}**")
                    st.markdown(f"### {score:.2%}")
                    st.markdown(emoji)

            overall = "⚠️ TOXIC" if any(preds > 0.5) else "✅ NOT TOXIC"
            if any(preds > 0.5):
                st.error(f"Overall: {overall}")
            else:
                st.success(f"Overall: {overall}")

# ── Tab 2: Bulk CSV ───────────────────────────────────────────
with tab2:
    st.subheader("Upload a CSV file for bulk predictions:")
    st.markdown("CSV must have a column named `comment_text`")

    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if 'comment_text' not in df.columns:
            st.error("CSV must have a 'comment_text' column!")
        else:
            st.write(f"Found {len(df)} comments. Analyzing...")
            progress = st.progress(0)
            results = []

            for i, row in df.iterrows():
                preds = predict_bert(row['comment_text'], model, tokenizer)
                result = {'comment': row['comment_text'][:50]}
                for label, score in zip(LABELS, preds):
                    result[label] = f"{score:.2%}"
                results.append(result)
                progress.progress((i + 1) / len(df))

            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

            csv = results_df.to_csv(index=False)


            st.download_button(
                "⬇️ Download Results",
                csv,
                "toxicity_results.csv",
                "text/csv"
            )