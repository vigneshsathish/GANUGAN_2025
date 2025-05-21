import streamlit as st
import requests
from datetime import datetime, timedelta
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import os

# ------------------ CONFIG ------------------

NEWS_API_KEY = ""  # 🔁 Replace with your actual key
OLLAMA_MODEL = "mixtral"  # 🔁 Change to llama3, mistral etc. if needed

# ------------------ FETCH NEWS ------------------

def fetch_news(query="stock market", from_days_ago=1, language="en", max_articles=5):
    url = "https://newsapi.org/v2/everything"
    from_date = (datetime.now() - timedelta(days=from_days_ago)).strftime('%Y-%m-%d')

    params = {
        "q": query,
        "from": from_date,
        "language": language,
        "sortBy": "relevancy",
        "pageSize": max_articles,
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(url, params=params)
    articles = response.json().get("articles", [])
    headlines = [f"{a['title']} - {a.get('description', '')}" for a in articles if a.get("title")]
    return "\n".join(headlines)

# ------------------ GENERATE STOCK SUGGESTIONS ------------------

def suggest_value_stocks(news_text, model=OLLAMA_MODEL):
    prompt = PromptTemplate.from_template("""
You are a financial analyst. Based on the following market news (current and past), strictly recommend only 5 future multibagger stocks with high growth potential not on large cap stocks.
Provide:
1. Stock name
2. Short reasoning for each
3. Rank them from best to least

Market News:
{news_text}
""")
    llm = OllamaLLM(model=model)
    chain = prompt | llm
    return chain.invoke({"news_text": news_text})

# ------------------ STREAMLIT UI ------------------

def main():
    st.set_page_config(page_title="AI Value Stock Recommender", layout="wide")
    st.title("📈 AI-Powered Value Stock Recommender")
    st.markdown("Daily stock picks using Ollama + Real-time news.")

    # Inputs
    stock_focus = st.selectbox("Market Segment", ["Indian Stock Market", "US Stock Market", "Bitcoin", "All"], index=0)
    today_days = st.slider("How many days of recent news?", 1, 90, 1)
    past_days = st.slider("How many days ago for historical news?", 15, 800, 30)

    if st.button("📊 Recommend Top 5 Stocks"):
        with st.spinner("Fetching and analyzing news..."):

            # Fetch and combine news
            today_news = fetch_news(query=stock_focus.lower(), from_days_ago=today_days)
            past_news = fetch_news(query=stock_focus.lower(), from_days_ago=past_days)
            full_news = f"### Recent News:\n{today_news}\n\n### Historical News:\n{past_news}"

            st.subheader("📰 News Summary")
            st.text_area("Combined News Context", full_news, height=300)

            # LLM recommendation
            response = suggest_value_stocks(full_news)
            st.subheader("💡 Top 5 Value Stock Picks")
            st.markdown(response)

if __name__ == "__main__":
    main()
