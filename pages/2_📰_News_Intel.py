
import nltk

# --- NLTK DOWNLOAD FIX ---
# This forces the cloud to download the dictionary every time it runs.
def download_nltk_data():
    resources = ['brown', 'punkt', 'punkt_tab', 'wordnet', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                nltk.download(resource)

download_nltk_data()
# -------------------------
import streamlit as st
import yfinance as yf
from textblob import TextBlob
import pandas as pd
from collections import Counter
import datetime

# --- SETTINGS & CONFIGURATION ---
st.set_page_config(page_title="Global Finance Sentiment", layout="wide")

# STOPWORDS for Keyword Extraction
STOPWORDS = set([
    'the', 'to', 'stock', 'shares', 'of', 'in', 'and', 'for', 'a', 'on', 'with', 'is', 'at', 
    'as', 'from', 'by', 'market', 'stocks', 'are', 'that', 'it', 'be', 'this', 'will', 'an',
    'has', 'was', 'have', 'but', 'or', 'which', 'up', 'down', 'new', 'after', 'today', 'why',
    'about', 'more', 'when', 'what', 'inc', 'ltd', 'share', 'markets', 'earnings', 'report',
    'price', 'buy', 'sell'
])

def get_sentiment(text):
    """
    Returns polarity and subjectivity using TextBlob.
    Polarity: -1 (Negative) to +1 (Positive)
    Subjectivity: 0 (Objective) to 1 (Subjective)
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def analyze_news(ticker_symbol):
    """
    Fetches news from yfinance, analyzes sentiment, and extracts keywords.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news
        
        if not news:
            return None, "No news found for this ticker."

        # Limit to top 10 articles
        news = news[:10]
        
        articles_data = []
        all_text = ""

        for article in news:
            # Handle nested structure - 'content' key seems to hold the main data now
            content = article.get('content', {})
            
            # TITLE
            title = content.get('title', article.get('title', ''))
            
            # TIME
            # Try to find time in content or root
            pub_time_raw = content.get('pubDate', article.get('providerPublishTime', None))
            if pub_time_raw:
                try:
                    # If it's a timestamp (int)
                    if isinstance(pub_time_raw, int):
                        pub_time = datetime.datetime.fromtimestamp(pub_time_raw).strftime('%Y-%m-%d %H:%M')
                    else:
                        # If it's a string, keep as is or try to parse
                        pub_time = str(pub_time_raw)
                except:
                    pub_time = str(pub_time_raw)
            else:
                pub_time = "N/A"
            
            # PUBLISHER
            publisher_data = content.get('provider', article.get('provider', {}))
            if isinstance(publisher_data, dict):
                publisher = publisher_data.get('displayName', 'Unknown')
            else:
                publisher = str(publisher_data) if publisher_data else 'Unknown'

            # LINK
            # Try clickThroughUrl or canonicalUrl in content
            click_url = content.get('clickThroughUrl', {})
            if isinstance(click_url, dict):
                link = click_url.get('url', '#')
            else:
                link = article.get('link', '#')

            # Skip if no title found
            if not title:
                continue

            polarity, subjectivity = get_sentiment(title)
            
            articles_data.append({
                'Headline': title,
                'Sentiment Score': polarity,
                'Subjectivity': subjectivity,
                'Source': publisher,
                'Time': pub_time,
                'Link': link
            })
            
            all_text += " " + title.lower()

        if not articles_data:
             return None, "No readable news articles found (titles parsed as empty)."

        df = pd.DataFrame(articles_data)

        # Keyword Extraction
        words = TextBlob(all_text).words
        # Filter stopwords and non-alphanumeric
        filtered_words = [w for w in words if w.isalpha() and w not in STOPWORDS and len(w) > 2]
        word_counts = Counter(filtered_words)
        try:
            top_keyword, _ = word_counts.most_common(1)[0]
        except IndexError:
            top_keyword = "N/A"

        return df, top_keyword

    except Exception as e:
        return None,str(e)

# --- SIDEBAR (THE ROUTER) ---
st.sidebar.header("Global Ticker Router")
region = st.sidebar.selectbox(
    "Select Market/Region",
    options=[
        "US (Default)", 
        "India (NSE)", 
        "London (LSE)", 
        "Hong Kong (HKSE)", 
        "Japan (Tokyo)", 
        "Europe (Paris/Euronext)"
    ]
)

suffix_map = {
    "US (Default)": "",
    "India (NSE)": ".NS",
    "London (LSE)": ".L",
    "Hong Kong (HKSE)": ".HK",
    "Japan (Tokyo)": ".T",
    "Europe (Paris/Euronext)": ".PA"
}

suffix = suffix_map[region]

st.sidebar.info(f"Suffix to be appended: '{suffix}'")

# --- MAIN APP ---
st.title("Global Financial News & Sentiment Analyzer 🌍")
st.markdown("Analyze news sentiment and key themes for companies worldwide.")

user_ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL, TATASTEEL, 9988):")

if user_ticker:
    full_ticker = f"{user_ticker.strip().upper()}{suffix}"
    st.write(f"Analyzing: **{full_ticker}**")

    with st.spinner("Fetching news..."):
        df_news, error_or_keyword = analyze_news(full_ticker)

    if df_news is not None:
        top_keyword = error_or_keyword
        avg_sentiment = df_news['Sentiment Score'].mean()

        # --- SECTION: BIG METRICS ---
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
        with col2:
            st.metric("Top Keyword", top_keyword.title())

        st.divider()

        # --- SECTION: VISUALIZATION ---
        st.subheader("Sentiment Analysis per Article")
        # Bar chart of sentiment scores
        st.bar_chart(df_news.set_index('Headline')['Sentiment Score'])

        # --- SECTION: DATA TABLE ---
        st.subheader("Latest News")
        
        # Format the dataframe for display (hide link if cleaner, or use column config)
        display_df = df_news[['Time', 'Source', 'Headline', 'Sentiment Score']].copy()
        
        st.dataframe(
            display_df, 
            use_container_width=True,
            column_config={
                "Sentiment Score": st.column_config.ProgressColumn(
                    "Sentiment",
                    format="%.2f",
                    min_value=-1,
                    max_value=1,
                    help="Polarity score from -1 (Neg) to +1 (Pos)"
                )
            }
        )

    else:
        st.warning(f"Could not fetch data. Reason: {error_or_keyword}")

else:
    st.info("Please enter a ticker symbol to begin.")

