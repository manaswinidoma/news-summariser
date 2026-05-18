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
    page_icon   = "News Icon.png",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ============================================================
# CUSTOM CSS — Inshorts style cards
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

/* ── OVERALL PAGE background ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #050505;
    color: #08027a;
}

/* ── MAIN CONTENT AREA ── */
.main .block-container {
    background: #f2e6ff;
    padding: 2rem 3rem 4rem;
    max-width: 1100px;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #c293ae; }
::-webkit-scrollbar-thumb { background: #c293ae; border-radius: 5px; }

/* ── MASTHEAD ── */
.masthead {
    border-top: 3px solid #451ce8;
    border-bottom: 3px solid #451ce8;
    padding: 1.5rem 0 1.2rem;
    margin-bottom: 2.5rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
}

.masthead-title {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    font-weight: 900;
    line-height: 1;
    letter-spacing: -2px;
    color: #0f013b;
    margin: 0;
}

.masthead-title span { color: #7175bf; }

.masthead-meta {
    text-align: right;
    font-size: 0.8rem;
    font-weight: 300;
    color: #000000;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    line-height: 1.8;
}

.masthead-date {
    font-size: 0.90rem;
    color: #999;
    font-weight: 400;
    letter-spacing: 0.06em;
}
    
            
            /* ── NEWS GRID ── */
.news-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5px;
    background: #451ce8;
    margin-bottom: 1.5px;
}

.news-grid-full {
    display: grid;
    grid-template-columns: 1fr;
    gap: 1.5px;
    background: #010729;
    margin-bottom: 1.5px;
}
            
            /* News card styling */
    .news-card {
        background-color: #ecebfa;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 2 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #0c0461;
    }
    


            /* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: #c5c5f0 !important;
    border-right: 2px solid #451ce8;
}

section[data-testid="stSidebar"] > div {
    padding-top: 0 !important;
    padding-left: 1.2rem;
    padding-right: 1.2rem;
    margin-top: -4rem !important;
}

.sidebar-brand {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 900;
    letter-spacing: -0.5px;
    color: #0f013b;
    margin-bottom: 0.2rem;
}

.sidebar-brand span { color: #7175bf; }

.sidebar-tagline {
    font-size: 0.65rem;
    font-weight: 400;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #000000;
    margin-bottom: 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #451ce8;
}

.sidebar-section {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #000000;
    margin: 0.2rem 0 0.2rem;
}

/* ── MODEL INFO ── */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid #7175bf;
}

.stat-label { font-size: 0.72rem; color: #000000; font-weight: 500; }
.stat-value { font-size: 0.72rem; color: #000000; font-weight: 300; font-family: 'Playfair Display', serif; }

/* ── ALL BUTTONS ── */
.stButton > button {
    background: #590355 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 900 !important;
    letter-spacing: 0.19em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1.5rem !important;
    transition: background 0.2s ease !important;
    width: 100% !important;
}

.stButton > button:hover {
    background: #d4aecf !important;
    color: #000000 !important;
}

.stButton > button p,
.stButton > button span {
    color: inherit !important;
    font-weight: 900 !important;
}

/* ── DROPDOWNS ── */
.stSelectbox label {
    font-size: 0.62rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #000000 !important;
}
            


/* ── Summary result box ── */
.summary-result {
    background: #141414;
    border: 1px solid #451ce8;
    border-left: 3px solid #7175bf;
    padding: 1rem;
    border-radius: 0 2px 2px 0;
    margin-top: 0.8rem;
}

.summary-result p {
    font-size: 0.82rem !important;
    color: #bbb !important;
    line-height: 1.7 !important;
    margin: 0 !important;
}

hr { border: none; border-top: 1px solid #451ce8; margin: 0.5rem 0; }
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
            return " Model is loading on HuggingFace servers, please wait 20 seconds and try again..."
        
        # Check if response is empty
        if not response.text:
            return " Empty response - model may be loading, please try again..."
        
        result = response.json()
        
        if isinstance(result, dict) and 'error' in result:
            return f" {result['error']} - please try again in 20 seconds"
        
        if isinstance(result, list) and len(result) > 0:
            if 'summary_text' in result[0]:
                return result[0]['summary_text']
            return " Unexpected response format — please try again"
        
        return "Model loading — please try again in 20 seconds"
    
    except requests.exceptions.Timeout:
        return "Request timed out - please try again"
    except Exception as e:
        error_msg = str(e)
        if "index out of range" in error_msg:
            return " Model is warming up — please try again in 20 seconds"
        return f" Request timed out — please try again in 20 seconds"

def clean_news_content(content: str, description: str = "") -> str:
    if not content:
        content = ""
    
    content = re.sub(r'\[\+\d+ chars\]', '', content)
    content = content.strip()
    
    # Only use description if content is very short
    # Don't combine both to avoid repetition
    if len(content.split()) < 50 and description:
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
                if len(cleaned.split()) >= 50:
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

    # ── Sidebar logo ──
    st.markdown("""
        <div class='sidebar-brand'>News<span>Brief</span></div>
        <div class='sidebar-tagline'>AI-powered summaries</div>
    """, unsafe_allow_html=True)

    # ── "EDITION" section label ──
    st.markdown("<div class='sidebar-section'>Edition</div>", unsafe_allow_html=True)

    # ── Category dropdown ──
    category = st.selectbox(
        "Category",
        ["General", "Technology", "Business", "Science", "Health", "Sports", "Entertainment"],
        index=0,
        label_visibility="collapsed"
    )

    # ── Country dropdown ──
    country = st.selectbox(
        "Country",
        ["us", "uk", "au", "in", "ca"],
        format_func=lambda x: {
            "us": "US  United States",
            "uk": "UK  United Kingdom",
            "au": "AU  Australia",
            "in": "IN  India",
            "ca": "CA  Canada"
        }[x],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Custom text summarizer
    st.markdown("<div class='sidebar-section'>Summarize any text</div>", unsafe_allow_html=True)
    custom_text = st.text_area(
        "Paste article",
        height=130,
        placeholder="Paste any article here...",
        label_visibility="collapsed"
    )
    
    summarize_btn = st.button(
        " Summarize", 
        use_container_width=True,
        type="primary"
    )
    
    
            
    if summarize_btn:
        if not custom_text:
            st.warning(" Please paste some text first.")
        elif len(custom_text.split()) < 50:
            st.error(f" Please enter at least 50 words. You entered {len(custom_text.split())} word(s).")
        else:
            with st.spinner("Generating summary..."):
                custom_summary = summarize_text(custom_text)
            
            # Check if summary is an error message
            if any(phrase in custom_summary for phrase in [
                "Error", "timed out", "loading", "please try again"
            ]):
                st.error(f" {custom_summary}")
            else:
                st.success(" Summary Generated!")
                st.markdown(f"**Summary:**\n\n{custom_summary}")
    
    # ── MODEL INFO stats panel at bottom of sidebar ──
    st.markdown("<div class='sidebar-section'>Model info</div>", unsafe_allow_html=True)
    for label, val in [
        ("Model", "BART-large-CNN"),
        ("Dataset", "Inshorts"),
        ("ROUGE-1", "42.44 %"),
        ("Max tokens", "1024"),
        
    ]:
        st.markdown(f"""
            <div class='stat-row'>
                <span class='stat-label'>{label}</span>
                <span class='stat-value'>{val}</span>
            </div>
        """, unsafe_allow_html=True) 
    

# ============================================================
# MAIN APP
# ============================================================

# Header

today = datetime.now().strftime("%A, %B %d %Y").upper()
edition_label = category.upper()

# ── MASTHEAD: big "NewsBrief" heading + date + top-right meta ──
st.markdown(f"""
    <div class='masthead'>
        <div>
            <div class='masthead-title'>News<span>Brief</span></div>
            <div class='masthead-date'>{today}</div>
        </div>
        <div class='masthead-meta'>
            AI-Powered News Intelligence<br>
            Edition: {edition_label}<br>
            Model: BART-large-CNN
        </div>
    </div>
""", unsafe_allow_html=True)


# Fetch news button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    fetch_btn = st.button(
        "Fetch Today's News", 
        use_container_width=True,
        type="primary"
    )

st.markdown("---")

# ── Fetch and display news ──
if fetch_btn:
    with st.spinner(" Fetching latest news..."):
        articles = fetch_news(category=category, country=country)
    
    if not articles:
        st.warning("No articles found. Try a different category or country.")
    else:
        st.success(f" Found {len(articles)} articles - Generating summaries....")
        
        # Display each article as a card
        for i, article in enumerate(articles):
            title   = article.get('title', 'No title')
            content = article.get('content', '')
            source  = article.get('source', {}).get('name', 'Unknown')
            url     = article.get('url', '#')
            
            # Skip if still too short
            if not content or len(content.split()) < 50:
                continue
            
            # Generate summary
            with st.spinner(f" Summarizing Article {i+1}/{len(articles)}..."):
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
            <h3> Click "Fetch Today's News" to get started</h3>
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