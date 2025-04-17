# 📊 Stock Sentiment Analyzer

This Streamlit web app helps users understand the market sentiment around publicly traded companies by analyzing social and news media content. The app uses sentiment analysis on Reddit discussions and online news articles from The Guardian and NYTimes APIs.

## 🔍 Features

- **Real-time Sentiment Analysis**: Gathers Reddit posts and news articles based on a stock ticker (e.g., `TSLA`, `AAPL`).
- **News Coverage**: Pulls and displays articles from:
  - 📰 The New York Times
  - 🗞 The Guardian
- **Interactive Visualizations**:
  - Dual-axis chart showing stock prices and sentiment trends over the past 30 days.
  - Keyword-tagged Reddit posts and news summaries.
- **Export Capability**: Download sentiment data as CSV.
- **Clear Market Mood**: Bullish / Bearish / Neutral sentiment indicator.

## ⚙️ How It Works

1. **Reddit Integration**:
   - Uses the `praw` library to scrape relevant posts from finance subreddits.
   - Applies NLTK’s VADER sentiment model to Reddit post text.

2. **News Integration**:
   - Queries the NYTimes and The Guardian APIs for recent news on the stock.
   - Extracts headline, date, description, URL, and sentiment for each article.

3. **Visualization**:
   - `matplotlib` is used to plot historical stock price vs average sentiment trend.

4. **Sentiment Analysis**:
   - VADER compound sentiment score is calculated and averaged per day.

## 📦 Tech Stack

- Python
- Streamlit
- yfinance
- praw
- NLTK (VADER Sentiment)
- NYTimes Article Search API
- The Guardian Content API

## 🛠 Setup Instructions

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
