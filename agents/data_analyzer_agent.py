# agents/search_scraper_agent.py
# This worker agent is specialized in finding and scraping data from the web.

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

def run(url: str) -> pd.DataFrame | str:
    """
    Entry point for the SearchAndScrapeAgent.
    It scrapes a URL to find the largest HTML table, then robustly cleans it.
    Returns a pandas DataFrame if a table is found, otherwise returns a string.
    """
    print(f"SearchAndScrapeAgent: Running on URL -> {url}")
    
    if not url:
        raise ValueError("SearchAndScrapeAgent requires a URL to run.")
        
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=25)
        response.raise_for_status()
        
        html_content = response.text

        # --- Strategy 1: Find and Clean HTML Table ---
        try:
            # Scrape all tables from the page
            tables = pd.read_html(html_content)
            if tables:
                main_table = max(tables, key=lambda df: df.size)

                # --- NEW: Robust Header Cleaning Process ---
                # Step 1: Handle multi-level headers by collapsing them.
                if isinstance(main_table.columns, pd.MultiIndex):
                    main_table.columns = ['_'.join(map(str, col)).strip() for col in main_table.columns.values]

                # Step 2: If columns are still not strings, the header is likely in the first row.
                if all(isinstance(c, int) for c in main_table.columns):
                    # Promote the first row to header
                    main_table.columns = main_table.iloc[0]
                    main_table = main_table[1:].reset_index(drop=True)

                # Step 3: Clean the final column names.
                # Remove non-alphanumeric characters, extra spaces, and citation brackets like [a], [b].
                def clean_col_name(name):
                    if not isinstance(name, str):
                        name = str(name)
                    name = re.sub(r'\[.*?\]', '', name) # Remove content in brackets
                    name = re.sub(r'[^A-Za-z0-9_ ]+', '', name) # Remove non-alphanumeric chars except underscore/space
                    return name.strip()

                main_table.columns = [clean_col_name(col) for col in main_table.columns]
                
                print(f"SearchAndScrapeAgent: Success (table found). Cleaned columns: {main_table.columns.tolist()}")
                return main_table
                
        except ValueError:
            print("SearchAndScrapeAgent: No HTML tables found. Switching to text extraction.")

        # --- Strategy 2: Fallback to text extraction ---
        soup = BeautifulSoup(html_content, 'lxml')
        if soup.body:
            main_text = soup.body.get_text(separator=' ', strip=True)
            if len(main_text) > 150:
                print(f"SearchAndScrapeAgent: Success (text found). Returning {len(main_text)} characters.")
                return main_text

        raise ValueError(f"No usable tables or significant text content could be extracted from the URL: {url}")

    except requests.exceptions.RequestException as e:
        print(f"SearchAndScrapeAgent Error: Network request failed. {e}")
        raise
    except Exception as e:
        print(f"SearchAndScrapeAgent Error: An unexpected error occurred. {e}")
        raise
