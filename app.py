import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime
import yfinance as yf
import praw
import requests
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from io import StringIO
import uuid

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

# Sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    return sia.polarity_scores(text)['compound']

def get_ticker_symbol(company_name):
    try:
        query = f"https://query1.finance.yahoo.com/v1/finance/search?q={company_name}"
        response = requests.get(query).json()
        for quote in response.get('quotes', []):
            if 'symbol' in quote and 'shortname' in quote:
                return quote['symbol']
        return company_name
    except:
        return company_name

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

def fetch_news_articles(stock_symbol):
    nyt_data = []
    guardian_data = []

    try:
        nyt_url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
        nyt_params = {
            "q": stock_symbol,
            "sort": "newest",
            "api-key": nytimes_api_key
        }
        nyt_response = requests.get(nyt_url, params=nyt_params)
        if nyt_response.status_code == 200:
            data = nyt_response.json()
            articles = data.get("response", {}).get("docs", [])
            if articles:
                for article in articles:
                    nyt_data.append({
                        "title": article.get("headline", {}).get("main", ""),
                        "description": article.get("abstract") or article.get("snippet") or "",
                        "publishedAt": article.get("pub_date", ""),
                        "url": article.get("web_url", "")
                    })
        else:
            st.warning(f"NYTimes API error: {nyt_response.status_code}")
    except Exception as e:
        st.error(f"NYTimes fetch error: {e}")

    try:
        guardian_url = "https://content.guardianapis.com/search"
        guardian_params = {
            "q": stock_symbol,
            "api-key": guardian_api_key,
            "order-by": "newest",
            "show-fields": "trailText"
        }
        guardian_response = requests.get(guardian_url, params=guardian_params)
        if guardian_response.status_code == 200:
            guardian_json = guardian_response.json()
            guardian_articles = guardian_json.get("response", {}).get("results", [])
            for i, item in enumerate(guardian_articles):
                guardian_data.append({
                    'title': item.get('webTitle', ''),
                    'description': item.get('fields', {}).get('trailText', ''),
                    'publishedAt': item.get('webPublicationDate', ''),
                    'url': item.get('webUrl', '')
                })
        else:
            st.warning(f"Guardian API error: {guardian_response.status_code}")
    except Exception as e:
        st.error(f"Guardian fetch error: {e}")

    return nyt_data, guardian_data

def display_reddit_posts(reddit_df):
    st.subheader("🗣️ Reddit Posts")
    for i, post in reddit_df.head(5).iterrows():
        unique_key = f"reddit_post_{i}_{uuid.uuid4()}"
        st.markdown(f"**{post['title']}**  \n[View Post]({post['url']})")
        st.markdown(f"Sentiment: {post['sentiment']:.2f}")
        st.text_area(f"Content for post {i}", post['text'], height=100, key=unique_key)


        
# Streamlit UI
st.set_page_config(page_title="Stock Sentiment Analyzer", layout="wide")
st.title("\U0001F4CA Stock Sentiment Analyzer")

stock_symbol = st.text_input("Enter Stock Symbol (e.g., TSLA)", "TSLA")
if st.button("Analyze"):
    with st.spinner("Fetching data and analyzing sentiment..."):
        reddit_df = get_reddit_posts(stock_symbol)
        reddit_df['sentiment'] = reddit_df['text'].apply(analyze_sentiment)
        reddit_score = reddit_df['sentiment'].mean() if not reddit_df.empty else 0

        nyt_articles, guardian_articles = fetch_news_articles(stock_symbol)
        nyt_data, guardian_data = [], []
        news_sentiments, news_dates = [], []

        for source_articles, source_list in [(nyt_articles, nyt_data), (guardian_articles, guardian_data)]:
            for article in source_articles:
                title = article.get('title', 'N/A')
                description = article.get('description', 'No summary available')
                full_text = f"{title} {description}"
                sentiment = analyze_sentiment(full_text)
                published_at = article.get('publishedAt', '')[:10]
                try:
                    date = datetime.strptime(published_at, '%Y-%m-%d').date()
                    news_dates.append(date)
                except:
                    date = None
                url = article.get('url', '')
                link = f"[{title}]({url})" if url else title
                news_sentiments.append(sentiment)
                row = [link, published_at if published_at else 'N/A', round(sentiment, 2), description]
                source_list.append(row)

        news_score = np.mean(news_sentiments) if news_sentiments else 0
        combined_score = np.mean([reddit_score, news_score])

        trend = "\U0001F4C8 Bullish" if combined_score > 0.2 else "\U0001F4C9 Bearish" if combined_score < -0.2 else "\U0001F4CB Neutral"
        st.success(f"**Predicted Market Trend for {stock_symbol}: {trend}**")
        st.metric("Average Sentiment", f"{combined_score:.2%}")
        st.metric("Reddit Sentiment", f"{reddit_score:.2%}")
        st.metric("News Sentiment", f"{news_score:.2%}")

        st.subheader("\U0001F4C9 Sentiment vs. Stock Price (Last 30 Days)")
        stock_data = yf.Ticker(stock_symbol).history(period="30d")[['Close']].reset_index()
        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date

        reddit_df['date'] = reddit_df['created_utc'].dt.date
        reddit_daily = reddit_df.groupby('date')['sentiment'].mean().reset_index()
        reddit_daily.columns = ['date', 'reddit_sentiment']

        news_df = pd.DataFrame({'date': news_dates, 'sentiment': news_sentiments})
        news_daily = news_df.groupby('date')['sentiment'].mean().reset_index()
        news_daily.columns = ['date', 'news_sentiment']

        sentiment_df = pd.merge(reddit_daily, news_daily, on='date', how='outer')
        sentiment_df = sentiment_df.set_index('date').asfreq('D').fillna(method='ffill').reset_index()
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
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

        st.subheader("\U0001F5E3️ Reddit Posts")
        for _, post in reddit_df.head(5).iterrows():
            st.markdown(f"**{post['title']}**  \n[View Post]({post['url']})")
            st.markdown(f"Sentiment: {post['sentiment']:.2f}")
            st.text_area("Content", post['text'], height=100)

        st.subheader("\U0001F4F0 News Articles")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### \U0001F4F0 NYTimes")
            nyt_df = pd.DataFrame(nyt_data, columns=["Headline", "Date Posted", "Sentiment", "Description"])
            st.write(nyt_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        with col2:
            st.markdown("### \U0001F5DE The Guardian")
            guardian_df = pd.DataFrame(guardian_data, columns=["Headline", "Date Posted", "Sentiment", "Description"])
            st.write(guardian_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        st.subheader("\u2B07\uFE0F Download Data")
        csv_buffer = StringIO()
        export_df = reddit_df[['title', 'text', 'sentiment', 'url']]
        export_df.to_csv(csv_buffer, index=False)
        st.download_button("Download Reddit Data as CSV", csv_buffer.getvalue(), file_name=f"{stock_symbol}_reddit_sentiment.csv", mime="text/csv")

        st.subheader("\U0001F9E0 Insight Summary")
        dominant = "Reddit discussions" if reddit_score > news_score else "News media"
        tone = "positive" if combined_score > 0.2 else "negative" if combined_score < -0.2 else "mixed"
        st.write(f"Overall sentiment for **{stock_symbol}** is {tone}. {dominant} currently shows the strongest influence on market perception.")
