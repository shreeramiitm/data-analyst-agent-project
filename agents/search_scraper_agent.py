# agents/search_scraper_agent.py
# This worker agent is specialized in finding and scraping data from the web.

import pandas as pd
import requests
from bs4 import BeautifulSoup

def run(url: str) -> pd.DataFrame | str:
    """
    Entry point for the SearchAndScrapeAgent.
    It scrapes a URL to find either the largest HTML table or the main text content.
    Returns a pandas DataFrame if a table is found, otherwise returns a string.
    """
    print(f"SearchAndScrapeAgent: Running on URL -> {url}")
    
    if not url:
        raise ValueError("SearchAndScrapeAgent requires a URL to run.")
        
    try:
        # Use a common user-agent to mimic a real browser and avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }
        response = requests.get(url, headers=headers, timeout=25) # Increased timeout for slow sites
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
        
        html_content = response.text

        # --- Strategy 1: Find all tables and return the largest one as a DataFrame ---
        # This is the preferred method as it provides structured data.
        try:
            tables = pd.read_html(html_content)
            if tables:
                # Find the table with the most cells (rows * columns)
                main_table = max(tables, key=lambda df: df.size)
                print(f"SearchAndScrapeAgent: Success (found table). Returning table with shape {main_table.shape}.")
                return main_table
        except ValueError:
            # pandas.read_html raises a ValueError if no tables are found
            print("SearchAndScrapeAgent: No HTML tables found. Switching to text extraction.")

        # --- Strategy 2: If no tables, find the main body of text and return it as a string ---
        # This is the fallback for articles, blogs, or other non-tabular pages.
        soup = BeautifulSoup(html_content, 'lxml')
        if soup.body:
            # A simple heuristic: get all text from the body, strip excess whitespace.
            # A more advanced version could prioritize <main> or <article> tags.
            main_text = soup.body.get_text(separator=' ', strip=True)
            if len(main_text) > 150: # Ensure there's meaningful content
                print(f"SearchAndScrapeAgent: Success (found text). Returning {len(main_text)} characters.")
                return main_text

        # If neither strategy works, raise an error.
        raise ValueError(f"No usable tables or significant text content could be extracted from the URL: {url}")

    except requests.exceptions.RequestException as e:
        print(f"SearchAndScrapeAgent Error: Network request failed. {e}")
        raise
    except Exception as e:
        print(f"SearchAndScrapeAgent Error: An unexpected error occurred. {e}")
        raise