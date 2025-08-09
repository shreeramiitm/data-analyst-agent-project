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
        response = requests.get(url, headers=headers, timeout=25)
        response.raise_for_status()
        
        html_content = response.text

        # --- Strategy 1: Find all tables and return the largest one as a DataFrame ---
        try:
            # Tell pandas that the header is the first row (index 0)
            tables = pd.read_html(html_content, header=0)
            if tables:
                main_table = max(tables, key=lambda df: df.size)

                # --- NEW: Clean the DataFrame ---
                # 1. Set the first row as the header (if read_html failed)
                if all(isinstance(col, int) for col in main_table.columns):
                    main_table.columns = main_table.iloc[0]
                    main_table = main_table[1:]

                # 2. Reset the index after promoting the header and removing a row
                main_table.reset_index(drop=True, inplace=True)

                print(f"SearchAndScrapeAgent: Success (found table). Returning table with columns: {main_table.columns.tolist()}")
                return main_table
        except ValueError:
            print("SearchAndScrapeAgent: No HTML tables found. Switching to text extraction.")

        # --- Strategy 2: Fallback to text extraction ---
        soup = BeautifulSoup(html_content, 'lxml')
        if soup.body:
            main_text = soup.body.get_text(separator=' ', strip=True)
            if len(main_text) > 150:
                print(f"SearchAndScrapeAgent: Success (found text). Returning {len(main_text)} characters.")
                return main_text

        raise ValueError(f"No usable tables or significant text content could be extracted from the URL: {url}")

    except requests.exceptions.RequestException as e:
        print(f"SearchAndScrapeAgent Error: Network request failed. {e}")
        raise
    except Exception as e:
        print(f"SearchAndScrapeAgent Error: An unexpected error occurred. {e}")
        raise
