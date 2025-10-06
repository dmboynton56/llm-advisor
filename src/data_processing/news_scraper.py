import requests
from bs4 import BeautifulSoup
import os
import sys
from datetime import datetime

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the LLM client to use for summarization
from src.api_clients.llm_client import get_ai_analysis 

# Define the target URL and headers to mimic a browser
NEWS_URL = "https://finviz.com/news.ashx"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def fetch_news_headlines():
    """
    Fetches the latest financial news headlines from Finviz.
    Returns a list of headlines.
    """
    print("  Fetching latest financial news headlines...")
    try:
        response = requests.get(NEWS_URL, headers=HEADERS)
        response.raise_for_status()  # Raise an exception for bad status codes

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Finviz news headlines are in 'a' tags with class 'nn-tab-link'
        news_table = soup.find('table', class_='news')
        if not news_table:
            print("  Warning: Could not find the news table on Finviz.")
            return []
            
        headlines = [a.text for a in news_table.find_all('a', class_='nn-tab-link')]
        
        # We only need the most recent headlines for pre-market sentiment
        recent_headlines = headlines[:20] # Get the top 20
        print(f"  Successfully fetched {len(recent_headlines)} recent headlines.")
        return recent_headlines

    except requests.exceptions.RequestException as e:
        print(f"  Error fetching news: {e}")
        return []

def summarize_news_with_llm(headlines: list):
    """
    Uses an LLM to summarize a list of headlines into a single paragraph
    describing the overall market sentiment.
    """
    if not headlines:
        return "No news headlines were found to summarize."

    print("  Sending headlines to LLM for sentiment summarization...")
    
    # Format the headlines into a single string for the prompt
    headlines_text = "\n".join(f"- {h}" for h in headlines)
    
    # This is the prompt we designed earlier
    prompt_content = (
        "You are a financial news analyst. Your task is to read the following headlines "
        "and synthesize them into a single, concise paragraph (max 3 sentences) that "
        "summarizes the overall market sentiment for the upcoming trading session. "
        "Focus on macroeconomic news and sentiment for the major indices (S&P 500, Nasdaq).\n\n"
        f"**News Headlines:**\n{headlines_text}\n\n"
        "**Your Summary:**"
    )
    
    try:
        # We'll use a generic system prompt for this simple task
        system_prompt = "You are a helpful assistant that provides concise summaries."
        summary = get_ai_analysis(system_prompt, prompt_content) # Assumes llm_client returns just the text content
        
        print("  LLM summary received.")
        return summary

    except Exception as e:
        print(f"  Error getting LLM summary: {e}")
        return "Could not generate news summary due to an error."

def get_daily_news_summary():
    """
    The main function for this module. Fetches and summarizes daily news.
    """
    print("\n--- Starting Daily News & Sentiment Analysis ---")
    headlines = fetch_news_headlines()
    summary = summarize_news_with_llm(headlines)
    print("--- News Analysis Complete ---")
    return summary

if __name__ == '__main__':
    # This allows you to test the script directly
    summary = get_daily_news_summary()
    print("\n--- TEST SUMMARY ---")
    print(summary)
