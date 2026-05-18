# 🗞️ NewsBrief — Abstractive News Summarizer

> An Inshorts-style AI-powered news summarizer that fetches today's news and generates concise summaries using the BART transformer model. Built as part of a Natural Language Processing (NLP) university assignment.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://news-summariser-qtejpjjtuogknz3maeao7w.streamlit.app)

---

## 📌 Project Overview

**Problem:** Information overload — people don't have time to read full news articles.

**Solution:** An NLP-powered web app that automatically fetches today's top news and summarizes each article into 60 words or less — just like Inshorts.

**Model:** `facebook/bart-large-cnn` — a sequence-to-sequence transformer pre-trained on CNN/DailyMail news articles for abstractive summarization.

---

## 🧱 Tech Stack

| Layer | Tool |
|---|---|
| NLP Model | facebook/bart-large-cnn (HuggingFace) |
| Dataset | [Kaggle Inshorts News Summary](https://www.kaggle.com/datasets/sunnysai12345/news-summary) |
| Training Environment | Kaggle (Tesla T4 GPU) |
| Web App | Streamlit |
| News Feed | NewsAPI.org |
| Model Inference | HuggingFace Inference API |
| Evaluation | ROUGE-1, ROUGE-2, ROUGE-L |

---

## 📊 Model Performance (ROUGE Scores)

Evaluated on 1,000 sampled articles from the Inshorts dataset:

| Metric | F1 Score | Percentage | Best Score | Worst Score |
|---|---|---|---|---|
| ROUGE-1 | 0.4244 | 42.44% | 0.8855 | 0.0440 |
| ROUGE-2 | 0.2056 | 20.56% | 0.7600 | 0.0000 |
| ROUGE-L | 0.3052 | 30.52% | 0.8454 | 0.0345 |

---

## 📁 Repository Structure

```
📦 news-summarizer/
├── 📓 notebook/
│   └── nlp_a3_summarizer.ipynb   ← Kaggle training notebook
├── 📊 outputs/
│   ├── results.csv               ← Full inference results
│   ├── rouge_summary.csv         ← ROUGE scores summary
│   ├── sample_outputs.txt        ← Example summaries
│   ├── experiment_config.json    ← Experiment configuration
│   ├── eda_length_distributions.png
│   ├── rouge_distributions.png
│   ├── rouge_bar_chart.png
│   ├── length_comparison.png
│   └── rouge_correlation.png
├── app.py                        ← Streamlit web application
├── requirements.txt              ← Python dependencies
├── .gitignore                    ← Git ignore rules
└── README.md                     ← This file
```

---

## 🚀 Running Locally

### Prerequisites
- Python 3.8+
- HuggingFace account (free) — for inference API token
- NewsAPI account (free) — for live news

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/news-summarizer.git
cd news-summarizer
```

### 2. Create virtual environment
```bash
python -m venv .venv

# Mac/Linux:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create `.env` file
```
HF_TOKEN=your_huggingface_token_here
NEWS_API_KEY=your_newsapi_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```

### 6. Open in browser
```
http://localhost:8501
```

---

## 🌐 Deployment on Streamlit Cloud

1. Push code to GitHub (public repo)
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Click **New app** → select this repo → `app.py`
5. Add secrets in **Advanced Settings:**
```
HF_TOKEN = "your_huggingface_token"
NEWS_API_KEY = "your_newsapi_key"
```
6. Click **Deploy** → get permanent URL ✅

---

## 🔑 Getting API Keys

**HuggingFace Token (free):**
1. Go to [huggingface.co](https://huggingface.co)
2. Settings → Access Tokens → New Token (Read)
3. Copy token → paste in `.env`

**NewsAPI Key (free):**
1. Go to [newsapi.org](https://newsapi.org)
2. Get API Key → Sign up
3. Copy key → paste in `.env`

---

## 📓 Kaggle Notebook

The full NLP pipeline is available on Kaggle:

🔗 **[View Kaggle Notebook](https://www.kaggle.com/code/domamanaswini/nlp-a3-25517970)**

### Notebook Sections:
1. Setting Up Environment
2. Configuration + Hyperparameters
3. Loading the Raw Data
4. Exploratory Data Analysis (EDA)
5. Data Quality Check
6. Text Preprocessing
7. Load Tokeniser and Inspect Tokenisation
8. Load BART Model and Run Inference
9. ROUGE Evaluation
10. Results Visualisation
11. Save Results and Model Output

---

## 🏗️ System Architecture

```
[NewsAPI] ─────────────────────────────────────┐
                                                ↓
[User] → [Streamlit App] → [HuggingFace API] → [BART Model]
                ↓
         [News Cards UI]
         Headline + Summary + Source
```

---

## 📈 EDA Findings

| Statistic | Articles | Summaries |
|---|---|---|
| Count | 4,396 | 4,396 |
| Mean length | 343 words | 58 words |
| Min length | 1 word | 44 words |
| Max length | 12,202 words | 62 words |
| Compression ratio | — | ~17% |

---

## ⚙️ Model Configuration

```python
MODEL_NAME       = 'facebook/bart-large-cnn'
MAX_INPUT_TOKENS = 1024
MIN_SUMMARY_LEN  = 30
MAX_SUMMARY_LEN  = 150
BEAM_SIZE        = 4
LENGTH_PENALTY   = 2.0
NO_REPEAT_NGRAM  = 3
SAMPLE_SIZE      = 1000
DEVICE           = 'cuda'  # Tesla T4 GPU on Kaggle
```

---

## 👥 Group Members
Manaswini Doma
Brunda Varshini Katapalli



