import streamlit as st
import requests
import os
import re
from dotenv import load_dotenv
from datetime import datetime

# ── Load environment variables from .env file ──
load_dotenv()
HF_TOKEN   = os.getenv("HF_TOKEN")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ── HuggingFace API settings ──
# Using same BART model as Kaggle notebook
HF_API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title  = "NewsBrief — AI News Summarizer",
    page_icon   = "🗞️",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ============================================================
# CUSTOM CSS — Inshorts style cards
# ============================================================
st.markdown("""
    <style>
    /* Main background */
    .main { background-color: #f5f5f5; }
    
    /* News card styling */
    .news-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #ff4b4b;
    }
    
    /* Headline styling */
    .headline {
        font-size: 18px;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 10px;
    }
    
    /* Summary styling */
    .summary {
        font-size: 14px;
        color: #444444;
        line-height: 1.6;
        margin-bottom: 10px;
    }
    
    /* Category badge */
    .badge {
        background-color: #ff4b4b;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }

    /* Header styling */
    .app-header {
        text-align: center;
        padding: 20px 0;
        margin-bottom: 30px;
    }
    
    /* Source text */
    .source-text {
        font-size: 12px;
        color: #888888;
        margin-top: 8px;
    }
    
    /* Loading text */
    .loading {
        text-align: center;
        color: #ff4b4b;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def summarize_text(text: str) -> str:
    words = text.split()
    if len(words) > 1024:
        text = ' '.join(words[:1024])
    
    try:
        response = requests.post(
            HF_API_URL,
            headers=HF_HEADERS,
            json={
                "inputs": text,
                "parameters": {
                    "max_length": 150,
                    "min_length": 50,
                    "num_beams": 4,
                    "length_penalty": 3.0,
                    "no_repeat_ngram_size": 3
                }
            },
            timeout=60
        )
        
        # Print status for debugging
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text[:200]}")
        
        # Model still loading
        if response.status_code == 503:
            return "⏳ Model is loading on HuggingFace servers, please wait 20 seconds and try again..."
        
        # Check if response is empty
        if not response.text:
            return "⏳ Empty response — model may be loading, please try again..."
        
        result = response.json()
        
        if isinstance(result, dict) and 'error' in result:
            return f"⏳ {result['error']} — please try again in 20 seconds"
        
        if isinstance(result, list) and len(result) > 0:
            return result[0]['summary_text']
        
        return "⏳ Model loading — please try again in 20 seconds"
    
    except requests.exceptions.Timeout:
        return "⏳ Request timed out — please try again"
    except Exception as e:
        return f"Error: {str(e)}"

def clean_news_content(content: str, description: str = "") -> str:
    if not content:
        content = ""
    
    # Remove [+XXXX chars] truncation marker
    content = re.sub(r'\[\+\d+ chars\]', '', content)
    content = content.strip()
    
    # Only use description if content is very short
    # Don't combine both to avoid repetition
    if len(content.split()) < 20 and description:
        content = description
    
    return content.strip()

def fetch_news(category: str = "general",
               country: str = "us") -> list:
    if not NEWS_API_KEY:
        return get_sample_articles()
    
    try:
        # US works with top-headlines
        # Other countries use everything endpoint
        if country == "us":
            url = (
                f"https://newsapi.org/v2/top-headlines?"
                f"country=us&"
                f"category={category}&"
                f"pageSize=10&"
                f"apiKey={NEWS_API_KEY}"
            )
        else:
            # Map country to search terms
            country_query = {
                "gb": "UK Britain",
                "au": "Australia",
                "in": "India",
                "ca": "Canada",
                "ae": "UAE Dubai",
                "za": "South Africa",
                "ng": "Nigeria",
                "nz": "New Zealand",
                "ie": "Ireland"
            }.get(country, "world")
            
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q={country_query}&"
                f"language=en&"
                f"sortBy=publishedAt&"
                f"pageSize=10&"
                f"apiKey={NEWS_API_KEY}"
            )
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data['status'] == 'ok':
            articles = []
            for a in data['articles']:
                content = a.get('content', '')
                description = a.get('description', '')
                cleaned = clean_news_content(content, description)
                if len(cleaned.split()) >= 20:
                    a['content'] = cleaned
                    articles.append(a)
            return articles if articles else get_sample_articles()
        return get_sample_articles()
    
    except Exception:
        return get_sample_articles()


def get_sample_articles() -> list:
    """
    Returns sample articles when NewsAPI key is not available.
    Used for demo purposes.
    """
    return [
        {
            "title": "Scientists develop new battery technology that charges in seconds",
            "content": """Researchers at MIT have developed a groundbreaking new battery 
            technology that can fully charge in under 60 seconds. The new lithium-ceramic 
            battery uses a novel electrode structure that allows ions to move much faster 
            than conventional batteries. The team says the battery can withstand over 
            10,000 charge cycles without degradation, far exceeding current lithium-ion 
            batteries that typically last 500-1000 cycles. The technology could 
            revolutionize electric vehicles and consumer electronics. Commercial 
            applications are expected within 3-5 years pending further testing and 
            regulatory approval. The research was published in the journal Nature Energy.""",
            "source": {"name": "Sample News"},
            "url": "#",
            "publishedAt": datetime.now().isoformat()
        },
        {
            "title": "Global temperatures hit record high for third consecutive year",
            "content": """World meteorological organizations have confirmed that global 
            average temperatures have reached a new record high for the third consecutive 
            year. The data shows average global temperatures were 1.5 degrees Celsius 
            above pre-industrial levels, crossing the critical threshold set by the Paris 
            Agreement. Scientists warn that extreme weather events including hurricanes, 
            droughts and floods will become more frequent and severe. World leaders are 
            being urged to accelerate the transition to renewable energy sources and 
            reduce carbon emissions more aggressively. Several countries have announced 
            new climate commitments in response to the alarming data.""",
            "source": {"name": "Sample News"},
            "url": "#",
            "publishedAt": datetime.now().isoformat()
        },
        {
            "title": "Major tech company announces breakthrough in quantum computing",
            "content": """A leading technology company has announced a significant 
            breakthrough in quantum computing, achieving a new milestone in qubit 
            stability. The company's quantum processor maintained coherence for over 
            10 minutes, far exceeding previous records of just a few seconds. This 
            development could accelerate the timeline for practical quantum computers 
            that can solve complex problems in fields like drug discovery, financial 
            modeling and cryptography. The breakthrough was achieved using a new error 
            correction technique that dramatically reduces quantum noise. Experts say 
            this brings us significantly closer to quantum advantage over classical 
            computers for real-world applications.""",
            "source": {"name": "Sample News"},
            "url": "#",
            "publishedAt": datetime.now().isoformat()
        }
    ]


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    
    st.markdown("### 🗞️ NewsBrief")
    # News category selector
    category = st.selectbox(
        "📂 News Category",
        ["general", "technology", "business", 
         "science", "health", "sports", "entertainment"],
        index=0
    )
    
    # Country selector
    country = st.selectbox(
        "🌍 Country",
        ["us", "gb", "au", "in", "ca"],
        format_func=lambda x: {
            "us": "🇺🇸 United States",
            "gb": "🇬🇧 United Kingdom", 
            "au": "🇦🇺 Australia",
            "in": "🇮🇳 India",
            "ca": "🇨🇦 Canada"
        }[x],
        index=0
    )
    
    st.markdown("---")
    
    # Custom text summarizer
    st.markdown("### ✏️ Summarize Custom Text")
    custom_text = st.text_area(
        "Paste any article text:",
        height=150,
        placeholder="Paste a news article here to summarize..."
    )
    
    summarize_btn = st.button(
        "🤖 Summarize", 
        use_container_width=True,
        type="primary"
    )
    
    if summarize_btn and custom_text:
        with st.spinner("Generating summary..."):
            custom_summary = summarize_text(custom_text)
        st.success("✅ Summary Generated!")
        st.markdown(f"**Summary:**\n\n{custom_summary}")
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown("""
    **Model:** facebook/bart-large-cnn  
    **Dataset:** Inshorts News Summary  
    **Evaluation:** ROUGE-1: 42.44%
    **Built with:** Streamlit + HuggingFace  
    """)

# ============================================================
# MAIN APP
# ============================================================

# Header
st.markdown("""
    <div class='app-header'>
        <h1>🗞️ NewsBrief</h1>
        <p style='color: #888; font-size: 16px;'>
            AI-powered news summarizer using BART — 
            Get today's news in 60 words or less
        </p>
    </div>
""", unsafe_allow_html=True)

# Fetch news button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    fetch_btn = st.button(
        "🔄 Fetch Today's News", 
        use_container_width=True,
        type="primary"
    )

st.markdown("---")

# ── Fetch and display news ──
if fetch_btn:
    with st.spinner("📡 Fetching latest news..."):
        articles = fetch_news(category=category, country=country)
    
    if not articles:
        st.warning("No articles found. Try a different category or country.")
    else:
        st.success(f"✅ Found {len(articles)} articles — generating summaries...")
        
        # Display each article as a card
        for i, article in enumerate(articles):
            title   = article.get('title', 'No title')
            content = article.get('content', '')
            source  = article.get('source', {}).get('name', 'Unknown')
            url     = article.get('url', '#')
            
            # Skip if still too short
            if not content or len(content.split()) < 20:
                continue
            
            # Generate summary
            with st.spinner(f"🤖 Summarizing article {i+1}/{len(articles)}..."):
                summary = summarize_text(content)
            
            # Display card
            st.markdown(f"""
                <div class='news-card'>
                    <div class='headline'>{title}</div>
                    <div class='summary'>{summary}</div>
                    <div class='source-text'>
                        📰 {source} • 
                        <a href='{url}' target='_blank'>Read Full Article →</a>
                    </div>
                </div>
            """, unsafe_allow_html=True)

else:
    # Default state — show instructions
    st.markdown("""
        <div style='text-align: center; padding: 50px; color: #888;'>
            <h3>👆 Click "Fetch Today's News" to get started</h3>
            <p>Or paste any article text in the sidebar to summarize it instantly</p>
            <br>
            <p style='font-size: 13px;'>
                Powered by <b>facebook/bart-large-cnn</b> — 
                the same model evaluated with ROUGE-1: 42.44% on the Inshorts dataset
            </p>
        </div>
    """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; font-size: 12px;'>
        NLP Assignment — Abstractive News Summarization using BART |
        Built with Streamlit + HuggingFace Transformers
    </div>
""", unsafe_allow_html=True)
