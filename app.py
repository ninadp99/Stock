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
    nyt_url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
    nyt_params = {
        "q": stock_symbol,
        "api-key": nytimes_api_key,
        "sort": "newest",
        "fq": "document_type:(\"article\")"
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

    nyt_data = []
    for item in nyt_articles:
        nyt_data.append({
            'title': item.get('headline', {}).get('main', ''),
            'description': item.get('snippet', ''),
            'publishedAt': item.get('pub_date', ''),
            'url': item.get('web_url', '')
        })

    guardian_data = []
    for item in guardian_articles:
        guardian_data.append({
            'title': item.get('webTitle', ''),
            'description': item.get('fields', {}).get('trailText', ''),
            'publishedAt': item.get('webPublicationDate', ''),
            'url': item.get('webUrl', '')
        })

    return nyt_data, guardian_data

# The rest of the Streamlit app remains unchanged
