"""
Bank Indonesia – Perception Engineering Tool (POC)
Single-page Streamlit application with LLM-powered topic extraction and recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from collections import Counter
import re
import os
import json
import csv

# OpenAI direct import (no LangChain to avoid torch/CUDA issues)
from openai import OpenAI
from dotenv import load_dotenv

# Load .env (lokal). Di server, set OPENAI_API_KEY sebagai env var sistem.
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Bank Indonesia – Perception Engineering Tool (POC)",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CUSTOM CSS - Adaptive for Dark/Light Mode
# =============================================================================
st.markdown("""
<style>
    /* Auto-adaptive header colors based on theme */
    .main-header { 
        font-size: 2rem; 
        font-weight: 700; 
        color: inherit; 
        margin-bottom: 0.5rem; 
    }
    .sub-header { 
        font-size: 1rem; 
        color: #6c757d; 
        margin-bottom: 2rem; 
    }
    .section-header { 
        font-size: 1.3rem; 
        font-weight: 600; 
        color: inherit; 
        margin: 2rem 0 1rem 0; 
        padding-bottom: 0.5rem; 
        border-bottom: 2px solid #0066cc; 
    }
    .metric-card { 
        background-color: rgba(255, 255, 255, 0.05); 
        border-radius: 10px; 
        padding: 1.5rem; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.1); 
        border: 1px solid rgba(128, 128, 128, 0.2); 
    }
    .metric-value { 
        font-size: 1.8rem; 
        font-weight: 700; 
        color: inherit; 
    }
    .metric-label { 
        font-size: 0.8rem; 
        color: #6c757d; 
        text-transform: uppercase; 
        letter-spacing: 0.5px; 
    }
    .spike-alert { 
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%); 
        color: white; 
        padding: 0.5rem 1rem; 
        border-radius: 20px; 
        font-weight: 600; 
    }
    .spike-normal { 
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
        color: white; 
        padding: 0.5rem 1rem; 
        border-radius: 20px; 
        font-weight: 600; 
    }
    .risk-badge { 
        padding: 0.5rem 1rem; 
        border-radius: 20px; 
        font-weight: 600; 
        text-align: center; 
    }
    .risk-high { background-color: #dc3545; color: white; }
    .risk-moderate { background-color: #ffc107; color: #1a1a2e; }
    .risk-low { background-color: #28a745; color: white; }
    .decision-gate { 
        padding: 1rem 2rem; 
        border-radius: 10px; 
        font-weight: 700; 
        text-align: center; 
        font-size: 1.2rem; 
    }
    .action-required { background-color: #dc3545; color: white; }
    .monitor { background-color: #17a2b8; color: white; }
    .chart-container { 
        background-color: rgba(255, 255, 255, 0.05); 
        border-radius: 10px; 
        padding: 1.5rem; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.1); 
        border: 1px solid rgba(128, 128, 128, 0.2); 
    }
    .info-box { 
        background-color: rgba(0, 102, 204, 0.1); 
        border-left: 4px solid #0066cc; 
        padding: 1rem; 
        border-radius: 0 5px 5px 0; 
        margin: 1rem 0; 
    }
    .recommendation-box { 
        background: linear-gradient(135deg, rgba(248, 249, 250, 0.1) 0%, rgba(233, 236, 239, 0.1) 100%); 
        padding: 1.5rem; 
        border-radius: 10px; 
        border-left: 4px solid #0066cc; 
    }
    .topic-tag { 
        background-color: #0066cc; 
        color: white; 
        padding: 0.3rem 0.8rem; 
        border-radius: 15px; 
        font-size: 0.85rem; 
        margin: 0.2rem; 
        display: inline-block; 
    }
    .filter-container { 
        background-color: rgba(128, 128, 128, 0.1); 
        padding: 1rem; 
        border-radius: 10px; 
        margin-bottom: 2rem; 
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
def detect_separator(filepath, sample_lines=5):
    """Auto-detect CSV separator (, or ;) by reading a small sample of the file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            sample = ''.join(f.readline() for _ in range(sample_lines))
        dialect = csv.Sniffer().sniff(sample, delimiters=',;|\t')
        return dialect.delimiter
    except Exception:
        # Fallback: count occurrences in first line
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                first_line = f.readline()
            return ';' if first_line.count(';') > first_line.count(',') else ','
        except Exception:
            return ','  # default fallback

@st.cache_data(ttl=3600)
def load_data():
    """Load and combine social media and online news media datasets."""
    df_sosmed = None
    df_onm = None
    
    # Try to load social media data (parquet-first, CSV fallback + auto-regenerate)
    try:
        _csv_s = 'data_sosmed.csv'
        _pq_s  = 'data_sosmed.parquet'
        _pq_s_fresh = (
            os.path.exists(_pq_s) and
            (not os.path.exists(_csv_s) or
             os.path.getmtime(_pq_s) >= os.path.getmtime(_csv_s))
        )
        if _pq_s_fresh:
            df_sosmed = pd.read_parquet(_pq_s)
        else:
            sep_sosmed = detect_separator(_csv_s)
            df_sosmed = pd.read_csv(_csv_s, sep=sep_sosmed, low_memory=False)
            df_sosmed.to_parquet(_pq_s, index=False)
        df_sosmed['source'] = 'Social Media'
        
        # Clean object_name column - remove leading '
        if 'object_name' in df_sosmed.columns:
            df_sosmed['object_name'] = df_sosmed['object_name'].astype(str).str.lstrip("'")
        
        # Map date_created to date - try multiple formats
        if 'date_created' in df_sosmed.columns:
            # Try Indonesian format first: DD/MM/YYYY HH.MM.SS
            df_sosmed['date'] = pd.to_datetime(df_sosmed['date_created'], format='%d/%m/%Y %H.%M.%S', errors='coerce')
            
            # If all NaT, try standard format
            if df_sosmed['date'].isna().all():
                df_sosmed['date'] = pd.to_datetime(df_sosmed['date_created'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            
            # If still NaT, try auto-detect
            if df_sosmed['date'].isna().all():
                df_sosmed['date'] = pd.to_datetime(df_sosmed['date_created'], errors='coerce')
        
        # Ensure final_topic exists - use 'topic' column (from Colab) if exists
        if 'final_topic' not in df_sosmed.columns:
            if 'topic' in df_sosmed.columns:
                df_sosmed['final_topic'] = df_sosmed['topic']
            else:
                df_sosmed['final_topic'] = 'Unknown'
        
        # Map engagement metrics
        if 'like' in df_sosmed.columns:
            df_sosmed['likes'] = df_sosmed['like']
        if 'share' in df_sosmed.columns:
            df_sosmed['shares'] = df_sosmed['share']
        
        # Calculate influencer score based on followers and engagement
        # Priority: 1) Followers count, 2) Engagement (likes + shares + comments), 3) Reach
        if 'followers' in df_sosmed.columns:
            # Normalize followers (log scale to handle large differences)
            df_sosmed['followers_clean'] = pd.to_numeric(df_sosmed['followers'], errors='coerce').fillna(0)
            df_sosmed['influencer_score'] = np.log1p(df_sosmed['followers_clean'])
            
            # Add engagement boost if available
            engagement_cols = []
            if 'engagement' in df_sosmed.columns:
                engagement_cols.append('engagement')
            else:
                if 'like' in df_sosmed.columns:
                    engagement_cols.append('like')
                if 'share' in df_sosmed.columns:
                    engagement_cols.append('share')
                if 'comment' in df_sosmed.columns:
                    engagement_cols.append('comment')
            
            if engagement_cols:
                total_engagement = df_sosmed[engagement_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1).fillna(0)
                # Add engagement weight (log scale)
                df_sosmed['influencer_score'] += np.log1p(total_engagement) * 0.5
            
            # Normalize influencer score to 0-1 range
            max_score = df_sosmed['influencer_score'].max()
            if max_score > 0:
                df_sosmed['influencer_score'] = df_sosmed['influencer_score'] / max_score
        else:
            # Fallback: use engagement only if followers not available
            df_sosmed['influencer_score'] = 0.5  # Default neutral score
        
        # Drop rows with invalid dates
        # Ensure final_sentiment column exists (check 'sentiment' from CSV)
        if 'sentiment' in df_sosmed.columns and 'final_sentiment' not in df_sosmed.columns:
            df_sosmed['final_sentiment'] = df_sosmed['sentiment']
        elif 'final_sentiment' not in df_sosmed.columns:
            df_sosmed['final_sentiment'] = 'neutral'
        
        # Normalize sentiment values
        def normalize_sentiment(series):
            return (
                series.astype(str)
                .str.lower()
                .str.strip()
                .str.replace(r"[^a-z]", "", regex=True)
            )
        df_sosmed['final_sentiment'] = normalize_sentiment(df_sosmed['final_sentiment'])
        
        df_sosmed = df_sosmed.dropna(subset=['date'])
    except FileNotFoundError:
        st.sidebar.warning("⚠️ data_sosmed.csv not found")
    except Exception as e:
        st.sidebar.error(f"Error loading sosmed: {str(e)}")
    
    # Try to load online media data (parquet-first, CSV fallback + auto-regenerate)
    try:
        _csv_o = 'data_onm.csv'
        _pq_o  = 'data_onm.parquet'
        _pq_o_fresh = (
            os.path.exists(_pq_o) and
            (not os.path.exists(_csv_o) or
             os.path.getmtime(_pq_o) >= os.path.getmtime(_csv_o))
        )
        if _pq_o_fresh:
            df_onm = pd.read_parquet(_pq_o)
        else:
            sep_onm = detect_separator(_csv_o)
            df_onm = pd.read_csv(_csv_o, sep=sep_onm, low_memory=False)
            df_onm.to_parquet(_pq_o, index=False)
        df_onm['source'] = 'Online Media'
        # Detect date column (date_published or date_created)
        date_col_onm = 'date_published' if 'date_published' in df_onm.columns else 'date_created' if 'date_created' in df_onm.columns else None
        if date_col_onm:
            df_onm['date'] = pd.to_datetime(df_onm[date_col_onm], format='%d/%m/%Y %H.%M.%S', errors='coerce')
            if df_onm['date'].isna().all():
                df_onm['date'] = pd.to_datetime(df_onm[date_col_onm], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            if df_onm['date'].isna().all():
                df_onm['date'] = pd.to_datetime(df_onm[date_col_onm], errors='coerce')
        # Map sentiment
        if 'final_sentiment' not in df_onm.columns:
            if 'sentiment' in df_onm.columns:
                df_onm['final_sentiment'] = df_onm['sentiment'].astype(str).str.lstrip("'").str.lower().str.strip()
            else:
                df_onm['final_sentiment'] = 'neutral'
        # Map content column
        if 'content' not in df_onm.columns and 'body' in df_onm.columns:
            df_onm['content'] = df_onm['body']
        if 'final_topic' not in df_onm.columns:
            df_onm['final_topic'] = 'Unknown'
        df_onm = df_onm.dropna(subset=['date'])
    except FileNotFoundError:
        pass
    except Exception as e:
        st.sidebar.error(f"Error loading onm: {str(e)}")
    
    # Generate dummy data if needed
    if df_sosmed is None or df_onm is None:
        df_dummy_sosmed, df_dummy_onm = generate_dummy_data()
        if df_sosmed is None:
            df_sosmed = df_dummy_sosmed
        if df_onm is None:
            df_onm = df_dummy_onm

    # Memory optimization: convert low-cardinality string columns to category
    # Saves ~60-80% memory for sentiment/topic/emotion columns (100k rows)
    for _df in [df_sosmed, df_onm]:
        if _df is not None:
            for _col in ['final_sentiment', 'final_topic', 'final_emotion', 'source']:
                if _col in _df.columns:
                    _df[_col] = _df[_col].astype('category')

    return df_sosmed, df_onm

def generate_dummy_data():
    """Generate dummy data for POC."""
    np.random.seed(42)
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(30)]
    
    sosmed_contents = [
        "Suku bunga BI naik lagi, rakyat makin susah",
        "Inflasi tahun ini sangat mengkhawatirkan",
        "Digital Rupiah akan meluncur tahun depan",
        "Ekonomi Indonesia stabil di tengah ketidakpastian global",
        "Kebijakan moneter BI efektif menahan inflasi",
        "Rupiah melemah terhadap dolar, ini analisisnya",
        "Sektor perbankan tumbuh positif di Q4",
        "BI rate naik 25 bps, ini dampaknya",
        "Ekonom memprediksi pertumbuhan ekonomi 5%",
        "Pariwisata Indonesia pulih pasca pandemi"
    ]
    
    onm_contents = [
        "Bank Indonesia mempertahankan suku bunga acuan",
        "Analisis: Kebijakan moneter ketat diperlukan",
        "Digitalisasi sistem pembayaran percepat inklusi keuangan",
        "Stabilitas sistem keuangan terjaga dengan baik",
        "Gubernur BI paparkan prospek ekonomi 2024",
        "Cadangan devisa Indonesia mencapai rekor",
        "Nilai tukar rupiah menguat tipis hari ini",
        "Sektor UMKM didorong untuk digitalisasi",
        "BI luncurkan program edukasi keuangan nasional",
        "Kredit perbankan tumbuh double digit"
    ]
    
    sosmed_data = []
    for i, date in enumerate(dates):
        is_spike = 11 <= i <= 14
        n_posts = np.random.randint(50, 150) if not is_spike else np.random.randint(200, 400)
        for _ in range(n_posts):
            sentiment = np.random.choice(['negative', 'positive', 'neutral'], 
                                         p=[0.5, 0.3, 0.2] if is_spike else [0.2, 0.5, 0.3])
            emotion = np.random.choice(['fear', 'anger', 'trust', 'joy']) if sentiment != 'neutral' else 'neutral'
            sosmed_data.append({
                'date': date, 'content': np.random.choice(sosmed_contents),
                'final_sentiment': sentiment, 'final_emotion': emotion,
                'likes': np.random.randint(10, 500), 'shares': np.random.randint(1, 100)
            })
    
    onm_data = []
    for i, date in enumerate(dates):
        is_spike = 11 <= i <= 14
        n_articles = np.random.randint(10, 30) if not is_spike else np.random.randint(50, 100)
        for _ in range(n_articles):
            sentiment = np.random.choice(['negative', 'positive', 'neutral'], 
                                         p=[0.4, 0.4, 0.2] if is_spike else [0.15, 0.6, 0.25])
            onm_data.append({
                'date': date, 'content': np.random.choice(onm_contents),
                'final_sentiment': sentiment, 'reach': np.random.randint(1000, 50000)
            })
    
    return pd.DataFrame(sosmed_data), pd.DataFrame(onm_data)

# =============================================================================
# LLM CONFIGURATION
# =============================================================================
def get_openai_client(api_key=None):
    """Initialize OpenAI client."""
    if api_key:
        return OpenAI(api_key=api_key)
    elif os.environ.get("OPENAI_API_KEY"):
        return OpenAI()
    return None

# =============================================================================
# TOPIC EXTRACTION (LLM-POWERED)
# =============================================================================
def extract_topics_llm_batch(contents, api_key=None):
    """Extract topics using OpenAI LLM directly."""
    client = get_openai_client(api_key)
    if not client:
        return extract_topics_fallback(contents)
    
    topics = []
    explanations = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, content in enumerate(contents):
        try:
            status_text.text(f"Processing content {i+1}/{len(contents)}...")
            
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are an economic policy analyst for Bank Indonesia. Extract the main policy-related topic from content. Respond in JSON format with 'topic_label' (max 5 words) and 'explanation' fields."},
                    {"role": "user", "content": f"Content: {content}"}
                ],
                temperature=0.3,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            topics.append(result.get('topic_label', 'General Economic Policy'))
            explanations.append(result.get('explanation', 'No explanation provided'))
            
        except Exception as e:
            topic, explanation = extract_single_topic_fallback(content)
            topics.append(topic)
            explanations.append(explanation)
        
        progress_bar.progress((i + 1) / len(contents))
    
    progress_bar.empty()
    status_text.empty()
    return topics, explanations

def extract_topics_fallback(contents):
    """Fallback keyword-based topic extraction."""
    topic_keywords = {
        'suku bunga': 'Interest Rate Policy', 'bunga': 'Interest Rate Policy', 'bi rate': 'Interest Rate Policy',
        'inflasi': 'Inflation Control', 'digital rupiah': 'Digital Currency', 'digitalisasi': 'Digital Transformation',
        'rupiah': 'Currency Stability', 'dolar': 'Currency Stability', 'ekonomi': 'Economic Growth',
        'pertumbuhan': 'Economic Growth', 'perbankan': 'Banking Sector', 'kredit': 'Banking Sector',
        'pariwisata': 'Tourism Recovery', 'umkm': 'MSME Support', 'cadangan devisa': 'Foreign Reserves',
        'sistem keuangan': 'Financial Stability', 'pembayaran': 'Payment Systems', 'kebijakan moneter': 'Monetary Policy'
    }
    
    topics = []
    explanations = []
    for content in contents:
        content_lower = content.lower()
        found_topics = [topic for keyword, topic in topic_keywords.items() if keyword in content_lower]
        
        if found_topics:
            topic = max(set(found_topics), key=found_topics.count)
            explanation = f"Content discusses {topic.lower()} based on detected keywords."
        else:
            topic = "General Economic Policy"
            explanation = "General economic policy discussion without specific topic keywords."
        
        topics.append(topic)
        explanations.append(explanation)
    return topics, explanations

def extract_single_topic_fallback(content):
    """Fallback for single content."""
    topics, explanations = extract_topics_fallback([content])
    return topics[0], explanations[0]

# =============================================================================
# TOPIC CLUSTERING
# =============================================================================
def cluster_topics(df):
    """Cluster topics and calculate aggregates."""
    if 'topic' not in df.columns or df.empty:
        return pd.DataFrame()
    
    topic_stats = df.groupby('topic').agg({
        'content': 'count',
        'final_sentiment': lambda x: (x == 'negative').mean()
    }).reset_index()
    topic_stats.columns = ['topic', 'volume', 'negative_ratio']
    return topic_stats.sort_values('volume', ascending=False)

# =============================================================================
# RISK CALCULATION
# =============================================================================
def calculate_risk_metrics(daily_df, raw_df=None):
    """Calculate risk metrics with dynamic spike detection."""
    if daily_df.empty:
        return daily_df
    
    # Dynamic spike detection: negative_ratio > mean + std
    mean_neg = daily_df['negative_ratio'].mean()
    std_neg = daily_df['negative_ratio'].std()
    threshold = mean_neg + std_neg if std_neg > 0 else mean_neg * 1.2
    
    daily_df['is_spike'] = daily_df['negative_ratio'] > threshold
    daily_df['velocity'] = daily_df['volume'].pct_change().fillna(0).abs().clip(0, 1)
    
    # Deterministic misinformation score (reproducible)
    daily_df['misinformation_score'] = (
        daily_df['negative_ratio'] * daily_df['velocity']
    ).clip(0, 1)
    
    # Calculate influencer impact from raw data if available
    # Uses followers count and engagement metrics
    if raw_df is not None and 'influencer_score' in raw_df.columns:
        # Group by date and calculate average influencer score per day
        daily_influencer = raw_df.groupby(raw_df['date'].dt.date)['influencer_score'].mean()
        
        # Map to daily_df dates
        daily_df['date_normalized'] = daily_df['date'].dt.date
        daily_df['influencer_impact'] = daily_df['date_normalized'].map(daily_influencer).fillna(0.5)
        daily_df.drop(columns=['date_normalized'], inplace=True)
    else:
        # Fallback: use volume-based ranking
        daily_df['influencer_impact'] = daily_df['volume'].rank(pct=True)
    
    daily_df['risk_score'] = (
        0.30 * daily_df['negative_ratio'] +      # Negative sentiment ratio (30%)
        0.25 * daily_df['velocity'] +            # Volume velocity/acceleration (25%)
        0.25 * daily_df['influencer_impact'] +   # Influencer impact score (25%)
        0.20 * daily_df['misinformation_score']  # Misinformation potential (20%)
    )
    return daily_df

def smooth_timeseries(df):
    """Smooth time series data for better visualization."""
    if df.empty:
        return df
    
    df = df.sort_values('date')
    
    # Smooth volume (moving average)
    df['volume_smooth'] = (
        df['volume']
        .rolling(window=5, center=True, min_periods=1)
        .mean()
    )
    
    # Smooth sentiment (EMA)
    df['negative_ratio_smooth'] = (
        df['negative_ratio']
        .ewm(span=7, adjust=False)
        .mean()
    )
    
    # Positive from smooth
    df['positive_ratio_smooth'] = 1 - df['negative_ratio_smooth']
    
    return df

def get_risk_class(risk_score):
    """Classify risk level."""
    if risk_score >= 0.7:
        return 'High', 'risk-high'
    elif risk_score >= 0.4:
        return 'Moderate', 'risk-moderate'
    else:
        return 'Low', 'risk-low'

# =============================================================================
# RECOMMENDATION GENERATION (LLM-POWERED)
# =============================================================================
def generate_recommendation_llm(dominant_topic, negative_ratio, risk_level, narrative_summary, api_key=None):
    """Generate strategic recommendation using OpenAI LLM."""
    client = get_openai_client(api_key)
    if not client:
        return generate_recommendation_fallback(dominant_topic, negative_ratio, risk_level, narrative_summary)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a strategic communication advisor for Bank Indonesia (Central Bank of Indonesia). Generate communication strategy recommendations in JSON format."},
                {"role": "user", "content": f"""Given:
- Dominant Topic: {dominant_topic}
- Negative Sentiment Ratio: {negative_ratio:.1%}
- Risk Level: {risk_level}
- Key Narrative Summary: {narrative_summary}

Generate a JSON response with these fields:
- strategy: Recommended Communication Strategy
- channel: Recommended Channel
- tone: Suggested Tone
- stakeholder: Target Stakeholder
- messaging: Array of 4 Key Messaging Points"""}
            ],
            temperature=0.4,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        if isinstance(result.get('messaging'), str):
            result['messaging'] = [m.strip() for m in result['messaging'].split(',')]
        return result
        
    except Exception as e:
        st.error(f"LLM Recommendation Error: {str(e)}. Using fallback.")
        return generate_recommendation_fallback(dominant_topic, negative_ratio, risk_level, narrative_summary)

def generate_recommendation_fallback(dominant_topic, negative_ratio, risk_level, narrative_summary):
    """Fallback template-based recommendation."""
    recommendations = {
        'High': {
            'strategy': 'Crisis Communication Protocol',
            'channel': 'Official Press Conference + Social Media Blitz',
            'tone': 'Empathetic, Authoritative, Transparent',
            'stakeholder': 'General Public, Media, Financial Markets',
            'messaging': [
                'Acknowledge concerns and provide reassurance',
                'Present clear action plan with timeline',
                'Deploy senior spokesperson for credibility',
                'Monitor and respond to misinformation actively'
            ]
        },
        'Moderate': {
            'strategy': 'Proactive Engagement Strategy',
            'channel': 'Social Media + Industry Webinar',
            'tone': 'Informative, Reassuring, Professional',
            'stakeholder': 'Banking Community, Policy Makers',
            'messaging': [
                'Increase transparency through educational content',
                'Engage key stakeholders in dialogue',
                'Clarify misconceptions with factual data',
                'Maintain consistent communication cadence'
            ]
        },
        'Low': {
            'strategy': 'Maintenance & Monitoring',
            'channel': 'Regular Communication Channels',
            'tone': 'Professional, Consistent',
            'stakeholder': 'All Stakeholders',
            'messaging': [
                'Continue regular information dissemination',
                'Monitor early warning signals',
                'Maintain positive narrative momentum',
                'Prepare contingency communications'
            ]
        }
    }
    return recommendations.get(risk_level, recommendations['Low'])

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    st.markdown("<div class='main-header'>Bank Indonesia – Perception Engineering Tool (POC)</div>", 
                unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Perception Intelligence & Decision Support System</div>", 
                unsafe_allow_html=True)
    
    # Sidebar: Refresh button — clears cache and forces reload from disk
    st.sidebar.markdown("### 📊 Data Status")
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    df_sosmed, df_onm = load_data()

    _sosmed_src = "parquet" if (
        os.path.exists('data_sosmed.parquet') and
        (not os.path.exists('data_sosmed.csv') or
         os.path.getmtime('data_sosmed.parquet') >= os.path.getmtime('data_sosmed.csv'))
    ) else "CSV"
    _onm_src = "parquet" if (
        os.path.exists('data_onm.parquet') and
        (not os.path.exists('data_onm.csv') or
         os.path.getmtime('data_onm.parquet') >= os.path.getmtime('data_onm.csv'))
    ) else "CSV"
    st.sidebar.info(f"Social Media: {len(df_sosmed):,} records [{_sosmed_src}]")
    st.sidebar.info(f"Online Media: {len(df_onm):,} records [{_onm_src}]")
    st.sidebar.caption("Ganti CSV lalu klik Refresh Data untuk update data.")
    
    # =============================================================================
    # TOP FILTER SECTION
    # =============================================================================
    st.markdown("**Filters**")
    
    filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 2])
    
    with filter_col1:
        data_source = st.selectbox("Data Source", options=['Social Media', 'Online Media', 'All'], index=0, key="data_source_sel")
    
    with filter_col2:
        # Safe date range calculation
        try:
            if not df_sosmed.empty and not df_onm.empty:
                min_date = min(df_sosmed['date'].min(), df_onm['date'].min()).date()
                max_date = max(df_sosmed['date'].max(), df_onm['date'].max()).date()
            elif not df_sosmed.empty:
                min_date = df_sosmed['date'].min().date()
                max_date = df_sosmed['date'].max().date()
            elif not df_onm.empty:
                min_date = df_onm['date'].min().date()
                max_date = df_onm['date'].max().date()
            else:
                min_date = datetime(2024, 1, 1).date()
                max_date = datetime(2024, 1, 31).date()
        except:
            min_date = datetime(2024, 1, 1).date()
            max_date = datetime(2024, 1, 31).date()
        
        # Initialize session state for filters if not exists
        if 'filter_date_range' not in st.session_state:
            st.session_state.filter_date_range = (min_date, min(min_date + timedelta(days=30), max_date))
        
        date_range = st.date_input("Date Range", value=st.session_state.filter_date_range, 
                                   min_value=min_date, max_value=max_date, key="date_range_input")
        
        # Only update session state when user actually changes the value
        if date_range != st.session_state.get('filter_date_range'):
            st.session_state.filter_date_range = date_range
    
    with filter_col3:
        # Topic filter - will be populated after data load
        topic_options = ['All Topics']
        if data_source == 'Social Media' and 'final_topic' in df_sosmed.columns:
            topic_options.extend(sorted(df_sosmed['final_topic'].dropna().unique().tolist()))
        elif data_source == 'Online Media' and 'final_topic' in df_onm.columns:
            topic_options.extend(sorted(df_onm['final_topic'].dropna().unique().tolist()))
        elif data_source == 'All':
            all_topics = set()
            if 'final_topic' in df_sosmed.columns:
                all_topics.update(df_sosmed['final_topic'].dropna().unique())
            if 'final_topic' in df_onm.columns:
                all_topics.update(df_onm['final_topic'].dropna().unique())
            topic_options.extend(sorted(all_topics))
        
        # Initialize session state for topic if not exists
        if 'filter_topic' not in st.session_state:
            st.session_state.filter_topic = 'All Topics'
        
        selected_topic = st.selectbox("Filter by Topic", options=topic_options, index=0, key="topic_select")
        
        # Only update session state when user actually changes the value
        if selected_topic != st.session_state.get('filter_topic'):
            st.session_state.filter_topic = selected_topic
    
    # Use session state values for filtering (prevents constant reloading)
    data_source = st.session_state.get("data_source_sel", 'Social Media')
    date_range = st.session_state.get("filter_date_range", (min_date, min_date + timedelta(days=30)))
    selected_topic = st.session_state.get("filter_topic", 'All Topics')
    
    # Apply filters — no unnecessary copies to save memory
    if data_source == 'Social Media':
        df_filtered = df_sosmed
    elif data_source == 'Online Media':
        df_filtered = df_onm
    else:
        df_filtered = pd.concat([df_sosmed, df_onm], ignore_index=True)

    # Filter by topic
    if selected_topic != 'All Topics' and 'final_topic' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['final_topic'] == selected_topic]

    if len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered['date'].dt.date >= date_range[0]) &
            (df_filtered['date'].dt.date <= date_range[1])
        ]
    
    # Show filter status with cache info
    st.caption(f"✓ Filters cached | Data Source: {data_source} | Topic: {selected_topic} | Date: {date_range[0]} to {date_range[1]}")

    # Use all filtered data (no BI mention filter)
    df_data = df_filtered
    
    # Ensure final_sentiment exists (check 'sentiment' column from CSV)
    if 'sentiment' in df_data.columns and 'final_sentiment' not in df_data.columns:
        df_data['final_sentiment'] = df_data['sentiment']
    elif 'final_sentiment' not in df_data.columns:
        df_data['final_sentiment'] = 'neutral'
    
    # Calculate metrics
    total_mentions = len(df_data)
    
    negative_pct = (df_data['final_sentiment'] == 'negative').mean() * 100 if total_mentions > 0 else 0
    positive_pct = (df_data['final_sentiment'] == 'positive').mean() * 100 if total_mentions > 0 else 0
    
    # Show data info
    if total_mentions == 0:
        st.warning("No data found in the selected filters.")
    else:
        st.info(f"Showing {total_mentions:,} total data. Negative: {negative_pct:.1f}%, Positive: {positive_pct:.1f}%")
    
    # Daily aggregation - group by date only (not datetime)
    if not df_data.empty and 'date' in df_data.columns:
        # Use external series so we don't modify df_data in-place
        _date_only = df_data['date'].dt.floor('D')
        daily_metrics = df_data.groupby(_date_only).agg({
            'content': 'count',
            'final_sentiment': lambda x: (x == 'negative').mean()
        }).reset_index()
        daily_metrics.columns = ['date', 'volume', 'negative_ratio']
        # Pass raw df_data to calculate influencer impact from followers/engagement
        daily_metrics = calculate_risk_metrics(daily_metrics, raw_df=df_data)
        daily_metrics = smooth_timeseries(daily_metrics)
    else:
        # Create empty dataframe with required columns
        daily_metrics = pd.DataFrame(columns=['date', 'volume', 'negative_ratio', 'is_spike', 'risk_score'])
    
    is_spike_active = daily_metrics['is_spike'].any() if not daily_metrics.empty else False
    current_risk = daily_metrics['risk_score'].iloc[-1] if not daily_metrics.empty else 0
    risk_level, risk_class = get_risk_class(current_risk)
    
    # =============================================================================
    # SECTION 1 – OVERVIEW ANALYTICS
    # =============================================================================
    st.markdown("<div class='section-header'>📊 Overview Analytics</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Data</div>"
                    f"<div class='metric-value'>{total_mentions:,}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>% Negative</div>"
                    f"<div class='metric-value' style='color: #dc3545;'>{negative_pct:.1f}%</div></div>", 
                    unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>% Positive</div>"
                    f"<div class='metric-value' style='color: #28a745;'>{positive_pct:.1f}%</div></div>", 
                    unsafe_allow_html=True)
    with col4:
        # Spike info with threshold
        if not daily_metrics.empty:
            mean_neg = daily_metrics['negative_ratio'].mean()
            std_neg = daily_metrics['negative_ratio'].std()
            threshold = mean_neg + std_neg if std_neg > 0 else mean_neg * 1.2
            spike_count = daily_metrics['is_spike'].sum()
        else:
            spike_count = 0
            threshold = 0
        
        alert_class = 'spike-alert' if is_spike_active else 'spike-normal'
        alert_text = f'{spike_count} Days' if is_spike_active else 'Normal'
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Spike Alert</div>"
                    f"<div style='margin-top: 10px;'><span class='{alert_class}'>● {alert_text}</span></div>"
                    f"<div style='font-size: 0.75rem; color: #6c757d; margin-top: 5px;'>Threshold: {threshold:.1%}</div></div>", 
                    unsafe_allow_html=True)
        
        # Spike explanation tooltip-style
        st.markdown(f"""
        <small style='color: #6c757d; font-size: 0.7rem;'>
        ℹ️ Spike detected when negative sentiment {'>'} mean ({mean_neg:.1%}) + std dev ({std_neg:.1%}). 
        Indicates unusual surge in negative conversations requiring attention.
        </small>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Risk Level</div>"
                    f"<div style='margin-top: 10px;'><span class='risk-badge {risk_class}'>{risk_level}</span></div></div>", 
                    unsafe_allow_html=True)
    
    
    # Emotion distribution - Time series view
    if data_source in ['Social Media', 'All'] and 'final_emotion' in df_filtered.columns:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("**Emotion Distribution**")
        emotion_data = df_filtered[df_filtered['final_emotion'].notna()]
        if not emotion_data.empty:
            # Group by date and emotion for timeseries
            daily_emotion = emotion_data.groupby(['date', 'final_emotion']).size().reset_index(name='count')
            emotion_pivot = daily_emotion.pivot(index='date', columns='final_emotion', values='count').fillna(0)
            emotion_pivot_pct = emotion_pivot.div(emotion_pivot.sum(axis=1), axis=0) * 100
            
            # Get emotions present (exclude neutral/no-emotion)
            emotions_present = [col for col in emotion_pivot.columns if col.lower() not in ['neutral', 'no-emotion', 'noemotion']]
            
            fig_emotion = go.Figure()
            emotion_colors = {'joy': '#28a745', 'trust': '#0066cc', 'fear': '#ffc107', 'anger': '#dc3545'}
            
            for emotion in emotions_present:
                if emotion in emotion_pivot_pct.columns:
                    fig_emotion.add_trace(go.Scatter(
                        x=emotion_pivot_pct.index, 
                        y=emotion_pivot_pct[emotion],
                        mode='lines',
                        name=emotion.capitalize(),
                        line=dict(color=emotion_colors.get(emotion, '#6c757d'), width=2)
                    ))
            
            # Add spike overlays
            if 'is_spike' in daily_metrics.columns:
                spike_dates = daily_metrics[daily_metrics['is_spike']]['date'].tolist()
                for spike_date in spike_dates:
                    fig_emotion.add_vrect(x0=spike_date - timedelta(hours=12), x1=spike_date + timedelta(hours=12),
                                         fillcolor="rgba(255, 0, 0, 0.1)", layer="below", line_width=0)
        
        fig_emotion.update_layout(margin=dict(l=20, r=20, t=30, b=20),
                                legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
                                yaxis_title='% of Total')
        st.plotly_chart(fig_emotion, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Emotion distribution - Time series view (like Sentiment)
    if data_source in ['Social Media', 'All'] and 'final_emotion' in df_filtered.columns:
        st.markdown("**Emotion Trend Over Time**")
        
        # Aggregate emotion by date - TIME SERIES view with smoothing
        emotion_data = df_filtered[df_filtered['final_emotion'].notna()]
        if not emotion_data.empty:
            # Group by date and emotion for timeseries
            daily_emotion = emotion_data.groupby(['date', 'final_emotion']).size().reset_index(name='count')
            emotion_pivot = daily_emotion.pivot(index='date', columns='final_emotion', values='count').fillna(0)
            emotion_pivot_pct = emotion_pivot.div(emotion_pivot.sum(axis=1), axis=0) * 100
            
            # Get top 4-5 emotions only (exclude neutral/no-emotion) for cleaner chart
            emotions_present = [col for col in emotion_pivot.columns if col.lower() not in ['neutral', 'no-emotion', 'noemotion']]
            
            # Calculate total occurrence per emotion to find top ones
            emotion_totals = {emotion: emotion_pivot[emotion].sum() for emotion in emotions_present}
            sorted_emotions = sorted(emotion_totals.items(), key=lambda x: x[1], reverse=True)
            top_emotions = [e[0] for e in sorted_emotions[:5]]  # Show top 5 emotions only
            
            fig_emotion = go.Figure()
            emotion_colors = {
                'anger': '#dc3545', 
                'fear': '#ffc107', 
                'joy': '#28a745', 
                'trust': '#0066cc',
                'proud': '#9b59b6',
                'sadness': '#95a5a6',
                'surprised': '#17a2b8'
            }
            
            # Add traces for top emotions with smoothed lines
            for emotion in top_emotions:
                if emotion in emotion_pivot_pct.columns:
                    # Apply rolling average for smoother line (window=3 days)
                    emotion_series = emotion_pivot_pct[emotion].rolling(window=3, center=True, min_periods=1).mean()
                    
                    fig_emotion.add_trace(go.Scatter(
                        x=emotion_pivot_pct.index, 
                        y=emotion_series,
                        mode='lines',
                        name=emotion.capitalize(),
                        line=dict(color=emotion_colors.get(emotion, '#6c757d'), width=2.5),
                        hovertemplate=f'<b>{emotion.capitalize()}</b><br>Date: %{{x|%b %d}}<br>% of Total: %{{y:.1f}}%<extra></extra>'
                    ))
            
            # Add spike overlays
            if 'is_spike' in daily_metrics.columns:
                for spike_date in daily_metrics[daily_metrics['is_spike']]['date'].tolist():
                    fig_emotion.add_vrect(x0=spike_date - timedelta(hours=12), x1=spike_date + timedelta(hours=12),
                                         fillcolor="rgba(255, 0, 0, 0.1)", layer="below", line_width=0)
            
            fig_emotion.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
                yaxis_title='% of Total',
                xaxis_title='Date',
                hovermode='x unified'
            )
            st.plotly_chart(fig_emotion, use_container_width=True, config={'displayModeBar': False})
            
            # No-emotion info below chart
            no_emotion_count = df_filtered[(df_filtered['final_emotion'].isna()) | (df_filtered['final_emotion'].astype(str).str.lower().isin(['neutral', 'no-emotion', 'noemotion']))].shape[0]
            no_emotion_pct = (no_emotion_count / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
            st.markdown(f"<small style='color: #6c757d;'>ℹ️ Neutral/No-Emotion: {no_emotion_count:,} ({no_emotion_pct:.1f}% of data) - Hidden for clarity</small>", unsafe_allow_html=True)
        
    
    # =============================================================================
    # SECTION 2 – TOPIC ANALYSIS (Using pre-computed final_topic from Colab)
    # =============================================================================
    st.markdown("<div class='section-header'>Topic Analysis</div>", unsafe_allow_html=True)
    
    if 'final_topic' in df_filtered.columns:
        topic_stats = df_filtered.groupby('final_topic').agg({
            'content': 'count',
            'final_sentiment': lambda x: (x == 'negative').mean()
        }).reset_index()
        topic_stats.columns = ['topic', 'volume', 'negative_ratio']
        topic_stats = topic_stats.sort_values('volume', ascending=False)
        
        # Top N selector for pie chart
        top_n = st.selectbox("Show Top N Topics in Chart", options=[5, 10, 15, 20], index=1)
        topic_stats_display = topic_stats.head(top_n)
        
        # Topic dropdown filter (all topics)
        all_topics = ['All Topics'] + topic_stats['topic'].tolist()
        selected_topic_detail = st.selectbox("Select Topic for Details", options=all_topics, index=0)
        
        # Show total topics info
        st.markdown(f"**Total Topics:** {len(topic_stats):,} | **Showing Top {min(top_n, len(topic_stats))} in Chart**")
        
        if not topic_stats.empty:
            # Single row layout: Pie chart (left) + Topic metrics (right)
            pie_col, metrics_col = st.columns([1.2, 1])
            
            with pie_col:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                st.markdown(f"**Top {top_n} Topics Distribution**")
                colors = px.colors.qualitative.Set3[:len(topic_stats_display)]
                fig_topics = go.Figure(data=[go.Pie(
                    labels=topic_stats_display['topic'],
                    values=topic_stats_display['volume'],
                    hole=0.4,
                    marker_colors=colors,
                    textinfo='percent',
                    textposition='inside',
                    hovertemplate='<b>%{label}</b><br>Data: %{value:,}<br>Share: %{percent}<extra></extra>'
                )])
                fig_topics.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    showlegend=True,
                    legend=dict(orientation='v', x=1.02, y=0.5, font=dict(size=9))
                )
                st.plotly_chart(fig_topics, use_container_width=True, config={'displayModeBar': False})
                st.markdown("</div>", unsafe_allow_html=True)
            
            with metrics_col:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                st.markdown("**📊 Topic Metrics**")
                
                current_topic = selected_topic_detail if selected_topic_detail != 'All Topics' else topic_stats.iloc[0]['topic']
                topic_data = topic_stats[topic_stats['topic'] == current_topic] if selected_topic_detail != 'All Topics' else topic_stats.iloc[0:1]
                
                st.markdown(f"<span style='font-size: 0.9rem;'><strong>Selected:</strong> <span class='topic-tag'>{current_topic}</span></span>", unsafe_allow_html=True)
                
                topic_volume = topic_data['volume'].values[0] if not topic_data.empty else 0
                topic_negative_pct = topic_data['negative_ratio'].values[0] * 100 if not topic_data.empty else 0
                
                anger_pct, fear_pct, joy_pct, trust_pct = 0, 0, 0, 0
                if 'final_emotion' in df_filtered.columns:
                    topic_emotions = df_filtered[
                        (df_filtered['final_topic'] == current_topic) & 
                        (df_filtered['final_emotion'].notna()) &
                        (df_filtered['final_emotion'] != 'neutral')
                    ]
                    if not topic_emotions.empty:
                        emotion_counts = topic_emotions['final_emotion'].value_counts()
                        total_emotion = emotion_counts.sum()
                        anger_pct = (emotion_counts.get('anger', 0) / total_emotion * 100) if total_emotion > 0 else 0
                        fear_pct = (emotion_counts.get('fear', 0) / total_emotion * 100) if total_emotion > 0 else 0
                        joy_pct = (emotion_counts.get('joy', 0) / total_emotion * 100) if total_emotion > 0 else 0
                        trust_pct = (emotion_counts.get('trust', 0) / total_emotion * 100) if total_emotion > 0 else 0
                
                st.markdown(f"<br><strong>📈 Total Data:</strong> {topic_volume:,}", unsafe_allow_html=True)
                st.markdown(f"<strong>📉 Negative:</strong> {topic_negative_pct:.1f}%", unsafe_allow_html=True)
                
                emotion_html = f"""
                <div style='display: flex; gap: 0.8rem; margin-top: 0.7rem; flex-wrap: wrap;'>
                    <div style='flex: 1; min-width: 70px; text-align: center; background: rgba(220, 53, 69, 0.1); padding: 0.4rem; border-radius: 6px;'>
                        <span style='color: #dc3545; font-size: 1.1rem;'>😠</span><br>
                        <small style='color: #dc3545;'><strong>Anger</strong></small><br>
                        <strong style='color: #dc3545;'>{anger_pct:.1f}%</strong>
                    </div>
                    <div style='flex: 1; min-width: 70px; text-align: center; background: rgba(255, 193, 7, 0.1); padding: 0.4rem; border-radius: 6px;'>
                        <span style='color: #ffc107; font-size: 1.1rem;'>😨</span><br>
                        <small style='color: #ffc107;'><strong>Fear</strong></small><br>
                        <strong style='color: #ffc107;'>{fear_pct:.1f}%</strong>
                    </div>
                    <div style='flex: 1; min-width: 70px; text-align: center; background: rgba(40, 167, 69, 0.1); padding: 0.4rem; border-radius: 6px;'>
                        <span style='color: #28a745; font-size: 1.1rem;'>😊</span><br>
                        <small style='color: #28a745;'><strong>Joy</strong></small><br>
                        <strong style='color: #28a745;'>{joy_pct:.1f}%</strong>
                    </div>
                    <div style='flex: 1; min-width: 70px; text-align: center; background: rgba(0, 102, 204, 0.1); padding: 0.4rem; border-radius: 6px;'>
                        <span style='color: #0066cc; font-size: 1.1rem;'>🤝</span><br>
                        <small style='color: #0066cc;'><strong>Trust</strong></small><br>
                        <strong style='color: #0066cc;'>{trust_pct:.1f}%</strong>
                    </div>
                </div>
                """
                st.markdown(emotion_html, unsafe_allow_html=True)
                
                st.markdown(f"<br><strong>⚖️ Risk:</strong>", unsafe_allow_html=True)
                if topic_negative_pct > 50:
                    st.markdown("<span class='spike-alert'>High Risk</span>", unsafe_allow_html=True)
                elif topic_negative_pct > 30:
                    st.markdown("<span style='background-color: #ffc107; color: #1a1a2e; padding: 0.4rem 0.9rem; border-radius: 20px;'>Moderate Risk</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='spike-normal'>✅ Low Risk</span>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.session_state['current_topic'] = current_topic
            st.session_state['topic_negative_pct'] = topic_negative_pct
            st.session_state['topic_stats'] = topic_stats
            st.session_state['topic_emotion_risk'] = (anger_pct * 0.50 + fear_pct * 0.30 - joy_pct * 0.20 - trust_pct * 0.10)
    else:
        st.info("No topic data available. Please ensure 'final_topic' column exists from Colab clustering.")

    # Emotion distribution (below Topic Analysis)
    if data_source in ['Social Media', 'All'] and 'final_emotion' in df_filtered.columns:
        st.markdown("**Emotion Distribution**")
        emotion_data = df_filtered[df_filtered['final_emotion'].notna()]
        if not emotion_data.empty:
            emotion_counts = emotion_data['final_emotion'].value_counts()
            no_emotion_mask = emotion_counts.index.str.lower().str.replace('-', '').str.replace(' ', '') == 'noemotion'
            no_emotion_count = int(emotion_counts[no_emotion_mask].sum())
            no_emotion_pct = no_emotion_count / len(emotion_data) * 100
            emotion_counts_filtered = emotion_counts[~no_emotion_mask]
            emotion_colors = {
                'joy': '#28a745', 'trust': '#0066cc', 'fear': '#ffc107', 'anger': '#dc3545',
                'neutral': '#6c757d', 'proud': '#9b59b6', 'sadness': '#95a5a6', 'surprised': '#17a2b8'
            }
            fig_emotion = go.Figure(data=[go.Bar(
                x=emotion_counts_filtered.index.str.upper(),
                y=emotion_counts_filtered.values,
                marker_color=[emotion_colors.get(e.lower(), '#6c757d') for e in emotion_counts_filtered.index]
            )])
            fig_emotion.update_layout(margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
            st.plotly_chart(fig_emotion, use_container_width=True, config={'displayModeBar': False})
            if no_emotion_count > 0:
                st.markdown(f"<small style='color: #6c757d;'>No-Emotion: {no_emotion_count:,} ({no_emotion_pct:.1f}% dari data)</small>", unsafe_allow_html=True)

    # =============================================================================
    # SECTION 4 – SPIKE & RISK ENGINE
    # =============================================================================
    st.markdown("<div class='section-header'>⚠️ Spike & Risk Engine</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        risk_color = "#dc3545" if current_risk > 0.7 else "#ffc107" if current_risk > 0.4 else "#28a745"
        risk_html = f"""<div class='metric-card' style='text-align: center; padding: 2rem;'>
            <div class='metric-label'>Current Risk Score</div>
            <div style='font-size: 3rem; font-weight: 700; color: {risk_color};'>{current_risk:.2f}</div>
            <div style='margin-top: 1rem;'><span class='risk-badge {risk_class}'>{risk_level}</span></div>
        </div>"""
        st.markdown(risk_html, unsafe_allow_html=True)
        
        decision = "Action Required" if current_risk > 0.6 else "Monitor"
        decision_class = "action-required" if current_risk > 0.6 else "monitor"
        st.markdown(f"<div style='margin-top: 1rem;'><div class='decision-gate {decision_class}'>🔔 {decision}</div></div>", 
                    unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("**Risk Breakdown**")
        if not daily_metrics.empty and len(daily_metrics) > 0:
            try:
                latest = daily_metrics.iloc[-1]
                
                # Check if columns exist before accessing
                required_cols = ['negative_ratio', 'velocity', 'influencer_impact', 'misinformation_score']
                if all(col in latest for col in required_cols):
                    risk_components = [
                        {"Component": "Negative Ratio", "Weight": "30%", "Value": f"{latest['negative_ratio']:.3f}", "Score": f"{latest['negative_ratio'] * 0.30:.3f}"},
                        {"Component": "Velocity", "Weight": "25%", "Value": f"{latest['velocity']:.3f}", "Score": f"{latest['velocity'] * 0.25:.3f}"},
                        {"Component": "Influencer Impact", "Weight": "25%", "Value": f"{latest['influencer_impact']:.3f}", "Score": f"{latest['influencer_impact'] * 0.25:.3f}"},
                        {"Component": "Misinformation Score", "Weight": "20%", "Value": f"{latest['misinformation_score']:.3f}", "Score": f"{latest['misinformation_score'] * 0.20:.3f}"}
                    ]
                    risk_df = pd.DataFrame(risk_components)
                    
                    fig_risk = go.Figure(data=[go.Table(
                        header=dict(values=['<b>Component</b>', '<b>Weight</b>', '<b>Value</b>', '<b>Weighted</b>'],
                                    fill_color='#0066cc', align='left', font=dict(color='white', size=11)),
                        cells=dict(values=[risk_df['Component'], risk_df['Weight'], risk_df['Value'], risk_df['Score']],
                                   fill_color=[['#f8f9fa', 'white'] * 3], align='left', font=dict(size=10))
                    )])
                    fig_risk.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=220)
                    st.plotly_chart(fig_risk, use_container_width=True, config={'displayModeBar': False})
                else:
                    st.info("Risk metrics not calculated yet")
            except Exception as e:
                st.error(f"Error displaying risk breakdown: {str(e)}")
        else:
            st.info("No risk data available")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # =============================================================================
    # SECTION 5 – DECISION TRIGGER & RECOMMENDATION
    # =============================================================================
    st.markdown("<div class='section-header'>Decision Trigger & AI Recommendation</div>", unsafe_allow_html=True)
    
    # Decision trigger based on selected topic (combining sentiment + emotion risk)
    topic_risk = "Low"
    if 'topic_negative_pct' in st.session_state:
        topic_negative_pct = st.session_state['topic_negative_pct']
        
        # Get emotion risk if available
        emotion_risk = st.session_state.get('topic_emotion_risk', 0)
        
        # Combined risk assessment: sentiment (60%) + emotion (40%)
        combined_risk_score = (topic_negative_pct * 0.60) + (emotion_risk * 0.40)
        
        if combined_risk_score >= 50 or topic_negative_pct > 50:
            topic_risk = "High"
        elif combined_risk_score >= 30 or topic_negative_pct > 30:
            topic_risk = "Moderate"
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("**🎯 Topic Risk Assessment**")
        
        current_topic = st.session_state.get('current_topic', 'Unknown')
        st.markdown(f"<span style='font-size: 0.9rem;'><strong>Topic:</strong> <span class='topic-tag'>{current_topic}</span></span>", unsafe_allow_html=True)
        
        # Show combined risk breakdown
        if 'topic_emotion_risk' in st.session_state:
            emotion_risk_display = st.session_state['topic_emotion_risk']
            st.markdown(f"<small style='color: #6c757d;'>Sentiment Risk: {topic_negative_pct:.1f}% | Emotion Risk: {emotion_risk_display:.1f}/100</small>", unsafe_allow_html=True)
        
        # Risk status without explanatory text
        if topic_risk == "High":
            st.markdown("<br><div class='decision-gate action-required'>⚠️ ACTION REQUIRED</div>", unsafe_allow_html=True)
        elif topic_risk == "Moderate":
            st.markdown("<br><div class='decision-gate monitor' style='background-color: #ffc107; color: #1a1a2e;'>MONITOR & ACT</div>", unsafe_allow_html=True)
        else:
            st.markdown("<br><div class='decision-gate monitor'>✅ MONITOR</div>", unsafe_allow_html=True)
        
        # Two action buttons
        st.markdown("<br>")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            followup_clicked = st.button("📌 Follow-up as Issue", type="primary", use_container_width=True, key="followup_btn")
            if followup_clicked:
                st.success("✅ Topic marked for follow-up as high-impact issue")
                st.session_state['followup_marked'] = True
                st.session_state['followup_topic'] = current_topic
        
        with col_btn2:
            generate_clicked = st.button("💡 Generate Recommendation", type="secondary", use_container_width=True, key="generate_btn")
            if generate_clicked:
                st.session_state['show_recommendation'] = True
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("**📋 Communication Strategy**")
        
        # Show recommendation only when Generate button is clicked
        if st.session_state.get('show_recommendation', False):
            if topic_risk == "High":
                st.markdown("<div class='recommendation-box' style='margin-top: 1rem;'>", unsafe_allow_html=True)
                st.markdown("**🔴 High Risk - Crisis Communication Protocol**")
                st.markdown("**Channel:** Official Press Conference + Social Media Blitz")
                st.markdown("**Tone:** Empathetic, Authoritative, Transparent")
                st.markdown("**Target Stakeholder:** General Public, Media, Financial Markets")
                st.markdown("**Key Messaging Points:**")
                st.markdown("• Acknowledge concerns and provide reassurance")
                st.markdown("• Present clear action plan with timeline")
                st.markdown("• Deploy senior spokesperson for credibility")
                st.markdown("• Monitor and respond to misinformation actively")
                st.markdown("</div>", unsafe_allow_html=True)
            elif topic_risk == "Moderate":
                st.markdown("<div class='recommendation-box' style='margin-top: 1rem;'>", unsafe_allow_html=True)
                st.markdown("**🟡 Moderate Risk - Proactive Engagement Strategy**")
                st.markdown("**Channel:** Social Media + Industry Webinar")
                st.markdown("**Tone:** Informative, Reassuring, Professional")
                st.markdown("**Target Stakeholder:** Banking Community, Policy Makers")
                st.markdown("**Key Messaging Points:**")
                st.markdown("• Increase transparency through educational content")
                st.markdown("• Engage key stakeholders in dialogue")
                st.markdown("• Clarify misconceptions with factual data")
                st.markdown("• Maintain consistent communication cadence")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='recommendation-box' style='margin-top: 1rem;'>", unsafe_allow_html=True)
                st.markdown("**🟢 Low Risk - Maintenance & Monitoring**")
                st.markdown("**Channel:** Regular Communication Channels")
                st.markdown("**Tone:** Professional, Consistent")
                st.markdown("**Target Stakeholder:** All Stakeholders")
                st.markdown("**Key Messaging Points:**")
                st.markdown("• Continue regular information dissemination")
                st.markdown("• Monitor early warning signals")
                st.markdown("• Maintain positive narrative momentum")
                st.markdown("• Prepare contingency communications")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("💡 Click 'Generate Recommendation' button to view communication strategy")
        
        # Show follow-up status if marked
        if st.session_state.get('followup_marked', False):
            st.markdown(f"<br><div class='info-box'><strong>📌 Follow-up Status:</strong> Topic '{st.session_state.get('followup_topic', current_topic)}' marked for BI issue tracking.</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    

if __name__ == "__main__":
    main()

