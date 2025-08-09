# AI Data Analysis Agent

This project is a multi-agent API that can source, prepare, analyze, and visualize data from any public webpage. The application is built with FastAPI and uses a powerful LLM to orchestrate a team of specialized agents for scraping, data analysis, and visualization.

This project is designed to fulfill the requirements of the "Data Analyst Agent" assignment.

## Features

-   **Generic Web Scraping**: Can extract data from any URL, automatically detecting and parsing HTML tables or extracting raw text if no tables are found.
-   **Multi-Agent Architecture**: Uses a master "Orchestrator" agent to create a dynamic plan and delegate tasks to specialized worker agents.
-   **AI-Powered Data Analysis**: Leverages an LLM to translate natural language questions into complex SQL queries to analyze structured data.
-   **Versatile Visualization**: Capable of generating various plots like scatter plots (with regression lines), bar charts, line charts, and histograms.
-   **Flexible API**: Accepts requests via a simple file upload (`questions.txt`) and returns results in a structured JSON format.

## Tech Stack

-   **Backend**: FastAPI, Gunicorn, Uvicorn
-   **AI/LLM**: OpenAI
-   **Data Handling**: Pandas, DuckDB
-   **Web Scraping**: Requests, BeautifulSoup4, lxml
-   **Plotting**: Matplotlib, Seaborn

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file:**
    Create a file named `.env` in the root directory and add your API credentials:
    ```env
    OPENAI_API_KEY="your_api_key_here"
    OPENAI_BASE_URL="your_base_url_here"
    ```

## How to Run Locally

1.  **Start the server:**
    ```bash
    uvicorn main:app --reload
    ```
    The application will be running at `http://127.0.0.1:8000`.

2.  **Send a request:**
    You can use `curl` to send a request to the API. Create a file named `questions.txt` with your query.

    **Example `questions.txt`:**
    ```text
    Scrape the list of highest grossing films from Wikipedia at [https://en.wikipedia.org/wiki/List_of_highest-grossing_films](https://en.wikipedia.org/wiki/List_of_highest-grossing_films).
    1. How many films grossed more than $2 billion?
    2. Draw a scatterplot of Rank vs. Peak with a dotted red regression line.
    ```

    **Send the request using `curl`:**
    ```bash
    curl -X POST -F "questions.txt=@questions.txt" [http://127.0.0.1:8000/api/](http://127.0.0.1:8000/api/)
    ```

## Deployment

This application is containerized with a `Dockerfile` and can be deployed on any platform that supports Docker, such as:
-   Render
-   Railway
-   AWS Elastic Beanstalk
-   Google Cloud Run