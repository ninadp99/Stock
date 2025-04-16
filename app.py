import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import yfinance as yf
import praw
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Reddit
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# NLTK sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Function to fetch Reddit posts
def get_reddit_posts(stock_symbol, limit=50):
    posts = []
    for post in reddit.subreddit("stocks+investing+wallstreetbets").search(f"{stock_symbol} stock", limit=limit):
        posts.append({
            'title': post.title,
            'text': post.selftext,
            'score': post.score,
            'created_utc': datetime.fromtimestamp(post.created_utc),
            'url': f"https://www.reddit.com{post.permalink}"
        })
    return pd.DataFrame(posts)

# NewsAPI fetch function
def fetch_news(query, page_size=20):
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(url, params=params)
    return response.json().get("articles", [])

# Analyze sentiment

def analyze_sentiment(text):
    return sia.polarity_scores(text)['compound']

# Fetch news + sentiment

def get_news_sentiments(stock_symbol, page_size=20):
    articles = fetch_news(stock_symbol, page_size)
    sentiments = []
    for article in articles:
        if not article or not isinstance(article, dict):
            continue
        title = str(article.get('title', ''))
        description = str(article.get('description', ''))
        full_text = title + ' ' + description
        score = analyze_sentiment(full_text)
        sentiments.append((article, score))
    return sentiments

# App starts here
st.set_page_config(page_title="Stock Sentiment Analyzer", layout="wide")
st.title("📊 Stock Sentiment Analyzer")

stock_symbol = st.text_input("Enter Stock Symbol (e.g., TSLA)", "TSLA")
if st.button("Analyze"):
    with st.spinner("Fetching data and analyzing sentiment..."):
        reddit_df = get_reddit_posts(stock_symbol, limit=50)
        reddit_df['sentiment'] = reddit_df['text'].apply(analyze_sentiment)

        news_data = get_news_sentiments(stock_symbol, page_size=20)
        news_scores = [score for _, score in news_data]

        reddit_score = reddit_df['sentiment'].mean() if not reddit_df.empty else 0
        news_score = np.mean(news_scores) if news_scores else 0
        combined_score = np.mean([reddit_score, news_score])

        if combined_score > 0.2:
            trend = "📈 Bullish"
        elif combined_score < -0.2:
            trend = "📉 Bearish"
        else:
            trend = "📋 Neutral"

        st.success(f"**Predicted Market Trend for {stock_symbol}: {trend}**")
        st.metric("Average Sentiment", f"{combined_score:.2%}")
        st.metric("Reddit Sentiment", f"{reddit_score:.2%}")
        st.metric("News Sentiment", f"{news_score:.2%}")

        st.subheader("🗣️ Reddit Posts")
        for _, post in reddit_df.head(5).iterrows():
            st.markdown(f"**{post['title']}**  \n[View Post]({post['url']})")
            st.markdown(f"Sentiment: {post['sentiment']:.2f}")
            st.text_area("Content", post['text'], height=100)

        st.subheader("📰 News Articles")
        for article, score in news_data[:5]:
            title = article.get('title', 'N/A')
            url = article.get('url', 'N/A')
            desc = article.get('description', 'No description available')
            st.markdown(f"**{title}**  \n[Read More]({url})")
            st.markdown(f"Sentiment: {score:.2f}")
            st.text_area("Summary", desc, height=100)
