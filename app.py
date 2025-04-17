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

# Load environment variables
load_dotenv()

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# API Keys
nytimes_api_key = os.getenv("NYTIMES_API_KEY")
guardian_api_key = os.getenv("GUARDIAN_API_KEY")

# Initialize VADER sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text: str) -> float:
    """Return VADER compound sentiment score for given text."""
    return sia.polarity_scores(text or "")['compound']


def get_reddit_posts(stock_symbol: str, limit: int = 50) -> pd.DataFrame:
    """Fetch recent Reddit posts mentioning the stock symbol."""
    posts = []
    for post in reddit.subreddit("stocks+investing+wallstreetbets").search(
        f"{stock_symbol} stock", limit=limit
    ):
        posts.append({
            'title': post.title,
            'text': post.selftext,
            'score': post.score,
            'created_utc': datetime.fromtimestamp(post.created_utc),
            'url': f"https://www.reddit.com{post.permalink}"
        })
    return pd.DataFrame(posts)


def fetch_news_articles(stock_symbol: str):
    """Fetch NYTimes and Guardian articles about the stock symbol."""
    # NYTimes
    nyt_url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
    nyt_params = {"q": stock_symbol, "api-key": nytimes_api_key, "sort": "newest"}
    nyt_resp = requests.get(nyt_url, params=nyt_params).json()
    nyt_docs = nyt_resp.get("response", {}).get("docs", []) or []

    # Guardian
    guard_url = "https://content.guardianapis.com/search"
    guard_params = {"q": stock_symbol, "api-key": guardian_api_key, "order-by": "newest", "show-fields": "trailText"}
    guard_resp = requests.get(guard_url, params=guard_params).json()
    guard_results = guard_resp.get("response", {}).get("results", []) or []

    nyt_data, guardian_data = [], []
    # Parse NYTimes
    for item in nyt_docs:
        title = item.get('headline', {}).get('main', '')
        description = item.get('abstract') or item.get('snippet') or item.get('lead_paragraph', '')
        publishedAt = item.get('pub_date', '')
        url = item.get('web_url', '')
        nyt_data.append({'title': title, 'description': description, 'publishedAt': publishedAt, 'url': url})
    # Parse Guardian
    for item in guard_results:
        title = item.get('webTitle', '')
        description = item.get('fields', {}).get('trailText', '')
        publishedAt = item.get('webPublicationDate', '')
        url = item.get('webUrl', '')
        guardian_data.append({'title': title, 'description': description, 'publishedAt': publishedAt, 'url': url})

    return nyt_data, guardian_data

# Streamlit UI
st.set_page_config(page_title="Stock Sentiment Analyzer", layout="wide")
st.title("📊 Stock Sentiment Analyzer")

symbol = st.text_input("Enter Stock Symbol (e.g., TSLA)", "TSLA").upper()
if st.button("Analyze"):
    with st.spinner("Fetching and analyzing..."):
        # Reddit analysis
        reddit_df = get_reddit_posts(symbol, limit=100)
        reddit_df['sentiment'] = reddit_df['text'].apply(analyze_sentiment)
        reddit_score = reddit_df['sentiment'].mean() if not reddit_df.empty else 0

        # News analysis
        nyt_articles, guardian_articles = fetch_news_articles(symbol)
        news_sentiments, news_dates = [], []
        nyt_rows, guard_rows = [], []
        # process NYTimes
        for art in nyt_articles:
            score = analyze_sentiment(f"{art['title']} {art['description']}")
            date_str = art['publishedAt'][:10]
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except:
                continue
            news_sentiments.append(score)
            news_dates.append(date)
            link = f"[{art['title']}]({art['url']})" if art['url'] else art['title']
            nyt_rows.append([link, date, round(score,2), art['description']])
        # process Guardian
        for art in guardian_articles:
            score = analyze_sentiment(f"{art['title']} {art['description']}")
            date_str = art['publishedAt'][:10]
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except:
                continue
            news_sentiments.append(score)
            news_dates.append(date)
            link = f"[{art['title']}]({art['url']})" if art['url'] else art['title']
            guard_rows.append([link, date, round(score,2), art['description']])

        news_score = np.mean(news_sentiments) if news_sentiments else 0
        combined_score = np.mean([reddit_score, news_score])

        # Display metrics
        trend = "📈 Bullish" if combined_score>0.2 else "📉 Bearish" if combined_score< -0.2 else "📋 Neutral"
        st.success(f"**Trend for {symbol}: {trend}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Sentiment", f"{combined_score:.2%}")
        c2.metric("Reddit Sentiment", f"{reddit_score:.2%}")
        c3.metric("News Sentiment", f"{news_score:.2%}")

        # Chart
        st.subheader("📉 Sentiment vs Stock Price (30d)")
        stock_df = yf.Ticker(symbol).history(period="30d")[['Close']].reset_index()
        stock_df['Date'] = stock_df['Date'].dt.date
        reddit_daily = reddit_df.groupby(reddit_df['created_utc'].dt.date)['sentiment'].mean().reset_index(name='reddit_sent')
        news_daily = pd.DataFrame({'date': news_dates, 'sentiment': news_sentiments}).groupby('date')['sentiment'].mean().reset_index(name='news_sent')
        senti_df = pd.merge(reddit_daily, news_daily, left_on='created_utc', right_on='date', how='outer')
        senti_df = senti_df.rename(columns={'created_utc':'date'}).set_index('date').asfreq('D').fillna(method='ffill').reset_index()
        senti_df['avg_sent'] = senti_df[['reddit_sent','news_sent']].mean(axis=1)
        merged = pd.merge(stock_df, senti_df, left_on='Date', right_on='date', how='left')
        fig, ax1 = plt.subplots(figsize=(10,4))
        ax2 = ax1.twinx()
        ax1.plot(merged['Date'], merged['Close'], color='green')
        ax2.plot(merged['Date'], merged['avg_sent'], color='blue')
        ax1.set_ylabel('Price',color='green'); ax2.set_ylabel('Sentiment',color='blue')
        st.pyplot(fig)

                # Reddit posts preview
        st.subheader("🗣️ Top Reddit Posts")
        for _, r in reddit_df.head(5).iterrows():
            # Display title with clickable link
            st.markdown(f"**{r['title']}**  
[Link]({r['url']})"), unsafe_allow_html=True)
            # Display sentiment score
            st.write(f"Sentiment: {r['sentiment']:.2f}")

        # News tables
        st.subheader("📰 News Articles")
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("### 📰 NYTimes")
            df_nyt = pd.DataFrame(nyt_rows, columns=['Headline','Date','Sent','Description'])
            st.write(df_nyt.to_html(escape=False,index=False),unsafe_allow_html=True)
        with col2:
            st.markdown("### 🗞 Guardian")
            df_guard = pd.DataFrame(guard_rows, columns=['Headline','Date','Sent','Description'])
            st.write(df_guard.to_html(escape=False,index=False),unsafe_allow_html=True)

        # Download
        st.download_button("Download Reddit CSV", reddit_df.to_csv(index=False),"reddit.csv")
        st.download_button("Download News CSV", pd.DataFrame({'date':news_dates,'sentiment':news_sentiments}).to_csv(index=False),"news.csv")

        # Insight
        st.subheader("🧠 Insight Summary")
        dom = "Reddit" if reddit_score>news_score else "News"
        tone = "positive" if combined_score>0.2 else "negative" if combined_score< -0.2 else "mixed"
        st.write(f"Overall sentiment for {symbol} is {tone}. Dominant source: {dom}.")
