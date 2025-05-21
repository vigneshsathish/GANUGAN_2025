# investment_bot.py — Your Personalized Investment Predictor using Ollama

from langchain_ollama import OllamaLLM

from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from datetime import datetime, timedelta
from prophet import Prophet
import pandas as pd

# ========== STEP 1: Summarize News from Economic Times + WSJ ==========
def fetch_news():
    headlines = []

    # Economic Times
    try:
        et_url = 'https://economictimes.indiatimes.com/markets'
        et_html = requests.get(et_url).text
        soup_et = BeautifulSoup(et_html, 'html.parser')
        for tag in soup_et.select('h3')[:5]:
            headlines.append("ET: " + tag.text.strip())
    except Exception as e:
        print("ET Error:", e)

    # Wall Street Journal
    try:
        wsj_url = 'https://www.wsj.com/news/markets'
        wsj_html = requests.get(wsj_url).text
        soup_wsj = BeautifulSoup(wsj_html, 'html.parser')
        for tag in soup_wsj.select('h3')[:5]:
            headlines.append("WSJ: " + tag.text.strip())
    except Exception as e:
        print("WSJ Error:", e)

    return headlines


# ========== STEP 2: Use Ollama to Summarize and Recommend =============
def summarize_with_ollama(headlines):
    llm = OllamaLLM(model="mixtral") # or llama3, mistral, etc.

    prompt_template = PromptTemplate.from_template(
        """
        Analyze the following financial headlines:
        {headlines}

        1. Summarize the market trends.
        2. Suggest top 3 investment areas (India/US/BTC).
        3. Highlight possible multibagger sectors.
        """
    )
    prompt = prompt_template.format(headlines="\n".join(headlines))
    result = llm.invoke(prompt)
    return result


# ========== STEP 3: Forecast Bitcoin Price ============================
def forecast_bitcoin():
    print("Forecasting Bitcoin price for next month...")
    btc = yf.download("BTC-USD", period="2y", interval="1d")
    btc = btc.reset_index()

    # Prepare data for Prophet
    btc = btc.rename(columns={"Date": "ds", "Close": "y"})
    btc = btc[['ds', 'y']]

    # Ensure no missing or non-numeric values
    btc = btc.dropna()
    btc['y'] = pd.to_numeric(btc['y'], errors='coerce')
    btc = btc.dropna()

    model = Prophet()
    model.fit(btc)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Print last 5 days forecast
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# ========== STEP 4: Main Pipeline =====================================
def main():
    print("\n🔍 Fetching latest news from Economic Times & WSJ...")
    headlines = fetch_news()

    print("\n🤖 Summarizing with Mixtral via Ollama...")
    summary = summarize_with_ollama(headlines)
    print("\n🧠 Investment Summary:\n", summary)

    print("\n📈 Forecasting Bitcoin price for next month...")
    btc_forecast = forecast_bitcoin()
    print(f"\n📊 BTC Predicted Price on {btc_forecast['ds'].date()}: ${btc_forecast['yhat']:.2f}")

if __name__ == "__main__":
    main()
