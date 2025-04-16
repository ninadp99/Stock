Stock Sentiment Analyzer 📊

This web app analyzes Reddit and news sentiment for any given stock symbol using NLP and shows market mood as Bullish, Bearish, or Neutral.

## Features
- Fetches Reddit posts from r/stocks, r/wallstreetbets, and r/investing
- Fetches financial news using NewsAPI
- Analyzes sentiment using NLTK's VADER
- Combines sentiment scores to predict market mood

## How to Run
1. Clone this repo
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file:
```env
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_USER_AGENT=your_user_agent
NEWS_API_KEY=your_newsapi_key
```
4. Run the app:
```bash
streamlit run app.py
