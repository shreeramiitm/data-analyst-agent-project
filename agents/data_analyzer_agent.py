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
        return _run_sql_analysis(df, question, llm_client)
    elif text_data is not None:
        return _run_text_analysis(text_data, question, llm_client)
    else:
        raise ValueError("DataAnalysisAgent requires either a DataFrame or text_data.")

def _run_sql_analysis(df: pd.DataFrame, question: str, llm_client: OpenAI) -> any:
    """Analyzes a pandas DataFrame using a generated DuckDB SQL query."""
    print("DataAnalysisAgent: Performing SQL analysis...")
    
    if df.empty:
        return "Cannot analyze: The provided DataFrame is empty."

    con = duckdb.connect(database=':memory:')
    con.register("data_table", df)
    
    try:
        schema = con.execute("PRAGMA table_info('data_table');").fetchdf().to_string()
    except Exception as e:
        schema = f"Could not retrieve schema. Please infer from column names: {df.columns.tolist()}"
        print(f"Warning: Could not get schema from DuckDB: {e}")

    # A more robust and stricter prompt to prevent context leakage
    sql_generation_prompt = f"""
You are an expert DuckDB SQL writer. Your ONLY job is to convert the user's question into a single, valid DuckDB SQL query based on the provided table schema.

The data is in a table named `data_table`. The table schema is:
{schema}

CRITICAL Rules:
1.  **You MUST only use column names that are explicitly listed in the schema above.** Do not invent or assume column names from previous conversations or examples (e.g., do not use "Worldwide gross" unless it is in the schema).
2.  You MUST respond with ONLY the SQL query. Do not add any explanation, markdown, or other text.
3.  Always use double quotes for column names that contain spaces or special characters (e.g., "Fert. Rate").
4.  If the user's question involves strings that need to be converted to numbers for calculations (e.g., population numbers with commas or currency), first clean them by removing symbols and letters, then `CAST` to a numeric type like `BIGINT`. For example: `CAST(REPLACE(REPLACE("Worldwide gross", '$', ''), ',', '') AS BIGINT)`.
5.  If a user's question seems to refer to a column that doesn't exist, use the most semantically similar column from the provided schema.
"""
    
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sql_generation_prompt},
            {"role": "user", "content": question}
        ]
    )
    sql_query = response.choices[0].message.content.strip().replace("```sql", "").replace("```", "")
    print(f"DataAnalysisAgent: Generated SQL -> {sql_query}")

    try:
        result_df = con.execute(sql_query).fetchdf()
        
        if result_df.empty:
            return None
        if result_df.size == 1:
            return result_df.iloc[0, 0]
        if result_df.shape[0] == 1:
            return result_df.iloc[0].to_dict()
        
        return result_df.to_dict(orient='records')

    except Exception as e:
        print(f"DataAnalysisAgent: SQL execution error: {e}")
        return f"Error executing query: {e}"
    finally:
        con.close()

def _run_text_analysis(text_data: str, question: str, llm_client: OpenAI) -> str:
    """Answers a question based on a body of text using an LLM."""
    print("DataAnalysisAgent: Performing text analysis...")
    
    system_prompt = """
You are an expert Data Analyst. Your job is to answer the user's question based *only* on the provided text content.
Be concise and extract the specific information requested. If the answer cannot be found in the text, state that clearly.
"""
    
    max_chars = 8000
    
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Based on the following text, please answer this question:\n\nQuestion: {question}\n\nText: {text_data[:max_chars]}"}
        ]
    )
    return response.choices[0].message.content.strip()
