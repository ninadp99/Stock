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
import matplotlib.pyplot as plt
from io import StringIO
from collections import Counter
import re

# Load environment variables
load_dotenv()

# Initialize Reddit
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# API Keys
nytimes_api_key = os.getenv("NYTIMES_API_KEY")
guardian_api_key = os.getenv("GUARDIAN_API_KEY")

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

# Analyze sentiment
def analyze_sentiment(text):
    return sia.polarity_scores(text)['compound']

# Fetch news from NYTimes and The Guardian
def fetch_news_articles(stock_symbol):
    nyt_url = f"https://api.nytimes.com/svc/search/v2/articlesearch.json"
    nyt_params = {
        "q": stock_symbol,
        "api-key": nytimes_api_key,
        "sort": "newest"
    }
    nyt_response = requests.get(nyt_url, params=nyt_params).json()
    nyt_articles = nyt_response.get("response", {}).get("docs", [])

    guardian_url = "https://content.guardianapis.com/search"
    guardian_params = {
        "q": stock_symbol,
        "api-key": guardian_api_key,
        "order-by": "newest",
        "show-fields": "trailText"
    }
    guardian_response = requests.get(guardian_url, params=guardian_params).json()
    guardian_articles = guardian_response.get("response", {}).get("results", [])

    combined = []
    for item in nyt_articles:
        combined.append({
            'title': item.get('headline', {}).get('main', ''),
            'description': item.get('snippet', ''),
            'publishedAt': item.get('pub_date', '')
        })
    for item in guardian_articles:
        combined.append({
            'title': item.get('webTitle', ''),
            'description': item.get('fields', {}).get('trailText', ''),
            'publishedAt': item.get('webPublicationDate', '')
        })
    return combined

# Parse and score sentiment
def get_news_sentiments(stock_symbol):
    articles = fetch_news_articles(stock_symbol)
    sentiments = []
    for article in articles:
        title = article.get('title', '')
        description = article.get('description', '')
        full_text = f"{title} {description}"
        score = analyze_sentiment(full_text)
        published_at = article.get('publishedAt', '')[:10]
        try:
            date = datetime.strptime(published_at, '%Y-%m-%d').date()
            sentiments.append({'date': date, 'sentiment': score})
        except:
            continue
    return pd.DataFrame(sentiments)

# --- Streamlit UI Block ---
st.set_page_config(page_title="Stock Sentiment Analyzer", layout="wide")
st.title("📊 Stock Sentiment Analyzer")

stock_symbol = st.text_input("Enter Stock Symbol (e.g., TSLA)", "TSLA")
if st.button("Analyze"):
    with st.spinner("Fetching data and analyzing sentiment..."):
        reddit_df = get_reddit_posts(stock_symbol)
        reddit_df['sentiment'] = reddit_df['text'].apply(analyze_sentiment)

        news_df = get_news_sentiments(stock_symbol)

        reddit_score = reddit_df['sentiment'].mean() if not reddit_df.empty else 0
        news_score = news_df['sentiment'].mean() if not news_df.empty else 0
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

        st.subheader("📉 Sentiment vs. Stock Price (Last 30 Days)")
        stock_data = yf.Ticker(stock_symbol).history(period="30d")
        stock_data = stock_data[['Close']].reset_index()
        stock_data['Date'] = stock_data['Date'].dt.date

        reddit_df['date'] = reddit_df['created_utc'].dt.date
        reddit_daily = reddit_df.groupby('date')['sentiment'].mean().reset_index()
        reddit_daily.columns = ['date', 'reddit_sentiment']

        news_daily = news_df.groupby('date')['sentiment'].mean().reset_index()
        news_daily.columns = ['date', 'news_sentiment']

        sentiment_df = pd.merge(reddit_daily, news_daily, on='date', how='outer')
        sentiment_df = sentiment_df.set_index('date').asfreq('D')
        sentiment_df = sentiment_df.fillna(method='ffill')
        sentiment_df['avg_sentiment'] = sentiment_df[['reddit_sentiment', 'news_sentiment']].mean(axis=1)
        sentiment_df = sentiment_df.reset_index()

        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
        merged = pd.merge(stock_data, sentiment_df, left_on='Date', right_on='date', how='left')

        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = ax1.twinx()

        ax1.plot(merged['Date'], merged['Close'], color='green', label='Stock Price')
        ax2.plot(merged['Date'], merged['avg_sentiment'], color='blue', label='Avg Sentiment')

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Price ($)', color='green')
        ax2.set_ylabel('Reddit + News Sentiment', color='blue')
        ax1.tick_params(axis='y', labelcolor='green')
        ax2.tick_params(axis='y', labelcolor='blue')
        plt.title(f"{stock_symbol} - Sentiment vs Stock Price")
        fig.tight_layout()
        st.pyplot(fig)

        st.subheader("🗣️ Reddit Posts")
        for _, post in reddit_df.head(5).iterrows():
            st.markdown(f"**{post['title']}**  \n[View Post]({post['url']})")
            st.markdown(f"Sentiment: {post['sentiment']:.2f}")
            st.text_area("Content", post['text'], height=100)

        st.subheader("📰 News Articles")
        nyt_data = []
        guardian_data = []

        articles = fetch_news_articles(stock_symbol)
        enriched_news = []
        for article in articles:
            title = article.get('title', 'N/A')
            description = article.get('description', 'No summary available')
            sentiment = article.get('sentiment', 0)
            date = article.get('date', 'N/A')
            url = article.get('url') or article.get('web_url') or article.get('webUrl', '')
            source = 'nytimes' if 'nytimes.com' in str(url) or 'nyt' in str(url) else 'guardian'
            link = f"[{title}]({url})" if url else title

            if source == 'nytimes':
                nyt_data.append([link, date, round(sentiment, 2), description])
            else:
                guardian_data.append([link, date, round(sentiment, 2), description])

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📰 NYTimes")
            nyt_df = pd.DataFrame(nyt_data, columns=["Headline", "Date Posted", "Sentiment", "Description"])
            st.write(nyt_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        with col2:
            st.markdown("### 🗞 The Guardian")
            guardian_df = pd.DataFrame(guardian_data, columns=["Headline", "Date Posted", "Sentiment", "Description"])
            st.write(guardian_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        st.subheader("⬇️ Download Data")
        csv_buffer = StringIO()
        export_df = reddit_df[['title', 'text', 'sentiment', 'url']]
        export_df.to_csv(csv_buffer, index=False)
        st.download_button("Download Reddit Data as CSV", csv_buffer.getvalue(), file_name=f"{stock_symbol}_reddit_sentiment.csv", mime="text/csv")

        st.subheader("🧠 Insight Summary")
        if reddit_score > news_score:
            source = "Reddit discussions"
        else:
            source = "News media"

        if combined_score > 0.2:
            tone = "positive"
        elif combined_score < -0.2:
            tone = "negative"
        else:
            tone = "mixed"

        st.write(f"Overall sentiment for **{stock_symbol}** is {tone}. {source} currently shows the strongest influence on market perception.")
