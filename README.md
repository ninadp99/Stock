# 📊 Stock Sentiment Analyzer

This is a Streamlit-based application that analyzes public sentiment around stocks using Reddit posts and news articles. It uses NLP techniques to determine whether the sentiment is bullish 📈, bearish 📉, or neutral 📋, and visualizes this data alongside stock prices.

---

## 🔍 Features

- 🔎 **Live Sentiment Analysis** from:
  - Reddit (WallStreetBets, Investing, Stocks)
  - The Guardian
  - The New York Times
- 📈 **30-day Sentiment vs. Stock Price Chart**
- 💬 **Top Reddit Discussions and Article Summaries**
- 🧠 **AI-generated Market Mood Insight**
- ⬇️ **CSV Export** of Reddit sentiment data

---

## 🧪 How It Works

### 🔸 Data Sources

- **Reddit**: via `praw`, based on stock symbol search
- **News**: via The Guardian and NYTimes public APIs (including filtering and improved error handling)

### 🔸 Sentiment Engine

- VADER Sentiment Analysis from NLTK
- Aggregates daily average sentiment from Reddit and News
- Cross-references article metadata for relevance

### 🔸 Visualization

- Uses `matplotlib` to plot stock prices from `yfinance` alongside sentiment trends
- Displays tables of enriched news metadata with hyperlinks

---

## 🛠 Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/stock-sentiment-analyzer.git
cd stock-sentiment-analyzer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the root directory:

```env
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_reddit_user_agent
NYTIMES_API_KEY=your_nyt_api_key
GUARDIAN_API_KEY=your_guardian_api_key
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py                  # Streamlit app source
├── requirements.txt        # Python dependencies
├── .env                    # API keys and credentials (not tracked)
├── README.md               # Documentation
```

---

## ✅ Status

✔️ NYTimes and Guardian articles now display headlines, URLs, and summaries
✔️ NYTimes fallback logic based on test success
✔️ Error messages and warnings are more descriptive

---

## 🙌 Contributions

Feel free to fork and improve. Issues and PRs welcome!

---

## 📄 License

MIT License © 2025 Ninad Pawar

