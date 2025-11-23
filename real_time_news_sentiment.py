# -*- coding: utf-8 -*-
"""
Real-Time News Sentiment Dashboard with Streamlit + TextBlob
"""

import os
import time
import uuid
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from textblob import TextBlob
from datetime import datetime

# ===================== CONFIG =====================
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "2ef1f5123905ae6f327d09fd011d1318")
PRED_DIR = "predictions_parquet"
os.makedirs(PRED_DIR, exist_ok=True)

# ===================== NEWS FETCHERS =====================
def fetch_news_newsapi(limit=20):
    """Fetch news from NewsAPI"""
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "apiKey": NEWSAPI_KEY,
        "language": "en",
        "country": "us",
        "pageSize": limit
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        articles = r.json().get("articles", [])
        
        if not articles:
            st.warning("No articles returned from NewsAPI")
            return pd.DataFrame()
        
        rows = []
        for a in articles:
            if a.get("title") and a.get("title") != "[Removed]":
                rows.append({
                    "id": a.get("url", str(uuid.uuid4())),
                    "source": (a.get("source") or {}).get("name", "Unknown"),
                    "title": a.get("title"),
                    "publishedAt": a.get("publishedAt", datetime.now().isoformat())
                })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"NewsAPI error: {str(e)}")
        return pd.DataFrame()

# ===================== SENTIMENT PREDICTION =====================
def classify_sentiment(df):
    """Classify sentiment using TextBlob"""
    if df.empty:
        return None
    
    sentiments = []
    prob_pos = []
    
    for title in df['title']:
        try:
            blob = TextBlob(str(title))
            polarity = blob.sentiment.polarity
            
            # Classify sentiment
            if polarity > 0.1:
                sentiment = "Positive"
            elif polarity < -0.1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            sentiments.append(sentiment)
            # Normalize polarity to 0-1 range for probability
            prob_pos.append((polarity + 1) / 2)
        except:
            sentiments.append("Neutral")
            prob_pos.append(0.5)
    
    df['sentiment'] = sentiments
    df['prob_pos'] = prob_pos
    
    # Save to parquet
    fname = os.path.join(PRED_DIR, f"pred_{uuid.uuid4().hex}.parquet")
    df.to_parquet(fname, index=False)
    
    return df

# ===================== LOAD RECENT DATA =====================
def load_recent(n=200):
    """Load recent predictions from parquet files"""
    import glob
    
    files = sorted(
        glob.glob(os.path.join(PRED_DIR, "*.parquet")), 
        key=os.path.getmtime, 
        reverse=True
    )[:50]
    
    if not files:
        return pd.DataFrame(columns=["id", "source", "title", "publishedAt", "sentiment", "prob_pos"])
    
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except:
            continue
    
    if not dfs:
        return pd.DataFrame(columns=["id", "source", "title", "publishedAt", "sentiment", "prob_pos"])
    
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['id'], keep='first')
    df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    
    return df.sort_values('publishedAt', ascending=False).head(n)

# ===================== DASHBOARD =====================
st.set_page_config(page_title="Real-Time News Sentiment", layout="wide", page_icon="📰")

st.title("📰 Real-Time News Sentiment Dashboard")
st.markdown("Analyze sentiment of top headlines using TextBlob")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    refresh_interval = st.slider("Refresh interval (seconds)", 10, 120, 30)
    fetch_limit = st.slider("Number of articles to fetch", 5, 50, 20)
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("This dashboard fetches real-time news and performs sentiment analysis using TextBlob.")

# Fetch button
col1, col2 = st.columns([1, 4])
with col1:
    fetch_button = st.button("🔄 Fetch News", type="primary", use_container_width=True)

with col2:
    if fetch_button:
        with st.spinner("Fetching latest news..."):
            df_new = fetch_news_newsapi(fetch_limit)
            
            if not df_new.empty:
                out = classify_sentiment(df_new)
                if out is not None:
                    st.success(f"✅ Processed {len(out)} headlines")
                else:
                    st.error("❌ Failed to classify sentiment")
            else:
                st.warning("⚠️ No headlines fetched")

# Load and display data
df = load_recent()

# Metrics
if not df.empty:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Headlines", len(df))
    
    with col2:
        positive_count = len(df[df['sentiment'] == 'Positive'])
        st.metric("Positive", positive_count)
    
    with col3:
        negative_count = len(df[df['sentiment'] == 'Negative'])
        st.metric("Negative", negative_count)
    
    with col4:
        neutral_count = len(df[df['sentiment'] == 'Neutral'])
        st.metric("Neutral", neutral_count)

st.markdown("---")

# Latest Headlines Table
st.subheader("📋 Latest Headlines")
if not df.empty:
    display_df = df[['publishedAt', 'source', 'title', 'sentiment', 'prob_pos']].copy()
    display_df.columns = ['Published', 'Source', 'Headline', 'Sentiment', 'Positivity Score']
    display_df['Published'] = display_df['Published'].dt.strftime('%Y-%m-%d %H:%M')
    display_df['Positivity Score'] = display_df['Positivity Score'].round(3)
    
    st.dataframe(
        display_df,
        height=400,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("👆 Click 'Fetch News' to load headlines")

# Visualizations
if not df.empty:
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Sentiment Distribution")
        counts = df['sentiment'].value_counts().reset_index()
        counts.columns = ['sentiment', 'count']
        
        fig = px.bar(
            counts, 
            x='sentiment', 
            y='count', 
            color='sentiment',
            title="Distribution of Sentiments",
            color_discrete_map={
                'Positive': '#2ecc71',
                'Negative': '#e74c3c',
                'Neutral': '#95a5a6'
            }
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Positivity Trend")
        df_sorted = df.sort_values('publishedAt')
        
        fig2 = px.line(
            df_sorted, 
            x='publishedAt', 
            y='prob_pos',
            title="Sentiment Over Time"
        )
        fig2.update_layout(
            xaxis_title="Time",
            yaxis_title="Positivity Score",
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("Powered by NewsAPI & TextBlob")