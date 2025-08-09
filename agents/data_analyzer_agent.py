# agents/data_analyzer_agent.py
# This worker agent is an expert in data analysis using SQL or natural language understanding.

import pandas as pd
import duckdb
from openai import OpenAI

def run(question: str, llm_client: OpenAI, df: pd.DataFrame = None, text_data: str = None) -> any:
    """
    Entry point for the DataAnalysisAgent.
    It answers a question by either:
    1. Converting it to a SQL query and running it on a DataFrame.
    2. Answering it directly from provided text content using an LLM.
    """
    print("DataAnalysisAgent: Running...")
    
    if df is not None:
        # If a DataFrame is provided, use SQL-based analysis
        return _run_sql_analysis(df, question, llm_client)
    elif text_data is not None:
        # If raw text is provided, use LLM-based text analysis
        return _run_text_analysis(text_data, question, llm_client)
    else:
        # This case should be prevented by the orchestrator
        raise ValueError("DataAnalysisAgent requires either a DataFrame or text_data to be provided.")

def _run_sql_analysis(df: pd.DataFrame, question: str, llm_client: OpenAI) -> any:
    """Analyzes a pandas DataFrame using a generated DuckDB SQL query."""
    print("DataAnalysisAgent: Performing SQL analysis...")
    
    if df.empty:
        return "Cannot analyze: The provided DataFrame is empty."

    # Connect to an in-memory DuckDB database and register the DataFrame as a table
    con = duckdb.connect(database=':memory:')
    con.register("data_table", df)
    
    try:
        # Attempt to get the table schema to provide more context to the LLM
        schema = con.execute("PRAGMA table_info('data_table');").fetchdf().to_string()
    except Exception as e:
        schema = f"Could not retrieve schema. Please infer from column names: {df.columns.tolist()}"
        print(f"Warning: Could not get schema from DuckDB: {e}")

    # A detailed prompt to guide the LLM in writing a correct DuckDB query
    sql_generation_prompt = f"""
You are an expert DuckDB SQL writer. Your ONLY job is to convert the user's question into a single, valid DuckDB SQL query.
The data is in a table named `data_table`. The table schema is:
{schema}

CRITICAL Rules:
1.  You MUST respond with ONLY the SQL query. Do not add any explanation, markdown, or other text.
2.  Always use double quotes for column names that contain spaces or special characters (e.g., "Worldwide gross").
3.  To use monetary strings (like '$2,123,456' or '€1.5B') in numeric calculations, you must first clean them. Remove currency symbols ('$', '€'), commas (','), and letters ('B' for billion, 'M' for million), and then cast to a numeric type. For example: `CAST(REPLACE(REPLACE("Worldwide gross", '$', ''), ',', '') AS BIGINT)`.
4.  If asked for a correlation, use the `CORR(column_a, column_b)` function.
5.  If asked for a specific row (e.g., 'the earliest film'), select the relevant columns from that single row. Do not use LIMIT 1 without an ORDER BY clause.
"""
    
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sql_generation_prompt},
            {"role": "user", "content": question}
        ]
    )
    # Clean up the response to get just the SQL
    sql_query = response.choices[0].message.content.strip().replace("```sql", "").replace("```", "")
    print(f"DataAnalysisAgent: Generated SQL -> {sql_query}")

    try:
        # Execute the query and fetch the result as a DataFrame
        result_df = con.execute(sql_query).fetchdf()
        
        # Smartly format the return value based on the result's shape
        if result_df.empty:
            return None
        if result_df.size == 1:
            return result_df.iloc[0, 0] # Return a single value directly
        if result_df.shape[0] == 1:
            return result_df.iloc[0].to_dict() # Return a single row as a dictionary
        
        return result_df.to_dict(orient='records') # Return multiple rows as a list of dictionaries

    except Exception as e:
        print(f"DataAnalysisAgent: SQL execution error: {e}")
        return f"Error executing query: {e}"
    finally:
        # Ensure the database connection is always closed
        con.close()

def _run_text_analysis(text_data: str, question: str, llm_client: OpenAI) -> str:
    """Answers a question based on a body of text using an LLM."""
    print("DataAnalysisAgent: Performing text analysis...")
    
    system_prompt = """
You are an expert Data Analyst. Your job is to answer the user's question based *only* on the provided text content.
Be concise and extract the specific information requested. If the answer cannot be found in the text, state that clearly.
"""
    
    # We truncate the text to avoid exceeding the LLM's context window limit
    max_chars = 8000
    
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Based on the following text, please answer this question:\n\nQuestion: {question}\n\nText: {text_data[:max_chars]}"}
        ]
    )
    return response.choices[0].message.content.strip()