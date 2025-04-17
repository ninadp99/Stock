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
from fin_news import FinNewsClient
import pdblp

# Load environment variables
load_dotenv()

# Initialize Reddit
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# Initialize FinNews and Bloomberg
finnews_api_key = os.getenv("THENEWS_API_KEY")
nytimes_api_key = os.getenv("NYTIMES_API_KEY")
fin_client = FinNewsClient(api_key=finnews_api_key)
bloomberg_client = pdblp.BCon(timeout=5000)
bloomberg_client.start()

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

# Fetch news from FinNews + NYTimes
def fetch_news_articles(stock_symbol):
    finnews_articles = fin_client.get_news(symbol=stock_symbol, limit=10)

    nyt_url = f"https://api.nytimes.com/svc/search/v2/articlesearch.json"
    params = {
        "q": stock_symbol,
        "api-key": nytimes_api_key,
        "sort": "newest"
    }
    nyt_response = requests.get(nyt_url, params=params).json()
    nyt_articles = nyt_response.get("response", {}).get("docs", [])

    combined = []
    for item in finnews_articles:
        combined.append({
            'title': item.get('title', ''),
            'description': item.get('summary', ''),
            'publishedAt': item.get('published', '')
        })

    for item in nyt_articles:
        combined.append({
            'title': item.get('headline', {}).get('main', ''),
            'description': item.get('snippet', ''),
            'publishedAt': item.get('pub_date', '')
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

# App starts here
st.set_page_config(page_title="Stock Sentiment Analyzer", layout="wide")
st.title("📊 Stock Sentiment Analyzer")

stock_symbol = st.text_input("Enter Stock Symbol (e.g., TSLA)", "TSLA")
if st.button("Analyze"):
    with st.spinner("Fetching data and analyzing sentiment..."):
        reddit_df = get_reddit_posts(stock_symbol, limit=50)
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
        for _, row in news_df.head(5).iterrows():
            st.markdown(f"**Date: {row['date']}**")
            st.markdown(f"Sentiment: {row['sentiment']:.2f}")

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
