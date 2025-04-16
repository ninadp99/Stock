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

        st.subheader("📉 Sentiment vs. Stock Price (Last 30 Days)")
        stock_data = yf.Ticker(stock_symbol).history(period="30d")
        stock_data = stock_data[['Close']].reset_index()
        stock_data['Date'] = stock_data['Date'].dt.date

        # Reddit daily sentiment
        reddit_df['date'] = reddit_df['created_utc'].dt.date
        reddit_daily = reddit_df.groupby('date')['sentiment'].mean().reset_index()
        reddit_daily.columns = ['date', 'reddit_sentiment']

        # News daily sentiment
        news_rows = []
        for article, score in news_data:
            date_str = article.get('publishedAt', '')[:10]
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d').date()
                news_rows.append({'date': date, 'news_sentiment': score})
            except:
                continue

        news_df = pd.DataFrame(news_rows)
        news_daily = news_df.groupby('date')['news_sentiment'].mean().reset_index()

        # Merge and average
        sentiment_df = pd.merge(reddit_daily, news_daily, on='date', how='outer').fillna(0)
        sentiment_df['avg_sentiment'] = sentiment_df[['reddit_sentiment', 'news_sentiment']].mean(axis=1)

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
        for article, score in news_data[:5]:
            title = article.get('title', 'N/A')
            url = article.get('url', 'N/A')
            desc = article.get('description', 'No description available')
            st.markdown(f"**{title}**  \n[Read More]({url})")
            st.markdown(f"Sentiment: {score:.2f}")
            st.text_area("Summary", desc, height=100)

        # 📁 CSV Export
        st.subheader("⬇️ Download Data")
        csv_buffer = StringIO()
        export_df = reddit_df[['title', 'text', 'sentiment', 'url']]
        export_df.to_csv(csv_buffer, index=False)
        st.download_button("Download Reddit Data as CSV", csv_buffer.getvalue(), file_name=f"{stock_symbol}_reddit_sentiment.csv", mime="text/csv")

        # 🧠 Simple Summary Generator (Keyword-based)
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
