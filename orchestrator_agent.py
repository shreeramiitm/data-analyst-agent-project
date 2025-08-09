# orchestrator_agent.py
# This is the "manager" or "master" agent. It orchestrates the entire workflow
# by delegating tasks to specialized worker agents.

import os
import json
import pandas as pd
from openai import OpenAI

# Import the specialized worker agents
from agents import search_scraper_agent
from agents import data_analyzer_agent
from agents import visualization_agent

class OrchestratorAgent:
    """The master agent that manages the entire data analysis task."""

    def __init__(self):
        """Initializes the Orchestrator and the OpenAI client using environment variables."""
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        
        if not api_key or not base_url:
            raise ValueError("OPENAI_API_KEY or OPENAI_BASE_URL not found in environment variables.")
        
        self.llm_client = OpenAI(api_key=api_key, base_url=base_url)
        print("OrchestratorAgent initialized with provided credentials.")

    async def run(self, prompt: str):
        """The main execution method for the entire workflow."""
        
        # Step 1: Create a high-level plan using the LLM.
        plan = self._generate_plan(prompt)
        print("--- Orchestrator Plan ---")
        print(json.dumps(plan, indent=2))
        print("-------------------------")

        # This will hold the data as it's passed between agents
        shared_context = {"original_prompt": prompt}
        final_results = []

        # Step 2: Execute the plan by calling the worker agents
        for i, task in enumerate(plan.get("tasks", [])):
            agent_name = task.get("agent")
            task_goal = task.get("goal")
            print(f"\nExecuting Task {i+1}: Delegating to '{agent_name}'")
            print(f"  Goal: {task_goal}")

            if agent_name == "SearchAndScrapeAgent":
                data_url = task.get("url")
                if not data_url:
                    raise ValueError("URL not found in the plan for SearchAndScrapeAgent.")
                
                scraped_data = search_scraper_agent.run(url=data_url)
                
                # Store the scraped data (either table or text) in the shared context
                if isinstance(scraped_data, pd.DataFrame):
                    shared_context["dataframe"] = scraped_data
                    shared_context["data_type"] = "table"
                    print("  -> Stored a DataFrame in shared context.")
                else:
                    shared_context["text_data"] = scraped_data
                    shared_context["data_type"] = "text"
                    print("  -> Stored text data in shared context.")

            elif agent_name == "DataAnalysisAgent":
                data_type = shared_context.get("data_type")
                analysis_result = None
                
                if data_type == "table":
                    df = shared_context.get("dataframe")
                    if df is None: raise ValueError("DataAnalysisAgent cannot run without a dataframe.")
                    analysis_result = data_analyzer_agent.run(df=df, question=task_goal, llm_client=self.llm_client)
                
                elif data_type == "text":
                    text = shared_context.get("text_data")
                    if text is None: raise ValueError("DataAnalysisAgent cannot run without text data.")
                    analysis_result = data_analyzer_agent.run(text_data=text, question=task_goal, llm_client=self.llm_client)
                else:
                    raise ValueError("No data found in context for DataAnalysisAgent to analyze.")

                final_results.append(analysis_result)
                print(f"  -> Orchestrator received result: {str(analysis_result)[:150]}...")

            elif agent_name == "VisualizationAgent":
                df = shared_context.get("dataframe")
                if df is None:
                    raise ValueError("VisualizationAgent cannot run without a dataframe. Ensure the data source contains a table.")
                
                plot_params = task.get("params", {})
                viz_result = visualization_agent.run(df=df, params=plot_params)
                final_results.append(viz_result)
                print("  -> Orchestrator received a base64 image.")

        # Step 3: Return the aggregated results
        return final_results


    def _generate_plan(self, prompt: str) -> dict:
        """Uses an LLM to decompose a prompt into a multi-agent JSON plan."""
        print("Orchestrator: Generating multi-agent plan...")
        
        system_prompt = """
You are a master orchestrator agent. Your primary function is to create a detailed, step-by-step JSON plan to fulfill a user's request by delegating tasks to specialized worker agents.

You have access to the following agents:
1.  `SearchAndScrapeAgent`: Given a URL, it scrapes data. It can return either a data table (if a `<table>` is found) or the main body of text from the page.
2.  `DataAnalysisAgent`: Answers a specific question about data. It can analyze a data table using SQL or analyze plain text to extract information.
3.  `VisualizationAgent`: Creates a plot (e.g., scatter, bar, line) from a data table.

**Your Task:**
Based on the user's entire prompt, create a JSON object containing a list of tasks.

**Critical Guidelines:**
- **First Task is always Scraping**: The first task in the list must be `SearchAndScrapeAgent` to retrieve the data from the URL mentioned in the prompt. Extract the URL accurately.
- **One Task Per Question**: For each distinct question or instruction the user asks about the data, create a separate task object for the appropriate agent (`DataAnalysisAgent` or `VisualizationAgent`).
- **Formulate Clear Goals**: For `DataAnalysisAgent`, the `goal` must be a clear, self-contained question. The agent is powerful, so you can ask complex questions involving filtering, aggregation, correlation, or finding specific rows.
- **Specify Visualization Details**: For `VisualizationAgent`, the `goal` should describe the plot. The `params` object must specify `plot_type`, `x_column`, `y_column`, and any styling like `color` or `linestyle` if mentioned.
- **Strict JSON Output**: Generate ONLY the JSON plan. Do not add any conversational text, explanations, or markdown formatting.

**Example User Request:**
"Please scrape the data from the page at https://en.wikipedia.org/wiki/List_of_highest-grossing_films.
1. How many films grossed more than $2 billion and were released before the year 2000?
2. What's the correlation between the 'Rank' and 'Peak' columns?
3. Draw a scatterplot of Rank versus Peak, and include a dotted red regression line."

**Example Plan Structure:**
{
  "tasks": [
    {
      "agent": "SearchAndScrapeAgent",
      "goal": "Fetch the data from the provided Wikipedia URL.",
      "url": "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    },
    {
      "agent": "DataAnalysisAgent",
      "goal": "Count how many films have a 'Worldwide gross' greater than $2 billion and a 'Year' before 2000."
    },
    {
      "agent": "DataAnalysisAgent",
      "goal": "Calculate the Pearson correlation coefficient between the 'Rank' column and the 'Peak' column."
    },
    {
      "agent": "VisualizationAgent",
      "goal": "Plot Rank vs. Peak with a dotted red regression line.",
      "params": {
        "plot_type": "scatter", 
        "x_column": "Rank", 
        "y_column": "Peak", 
        "regression_line": true,
        "color": "red",
        "linestyle": "dotted"
      }
    }
  ]
}
"""
        response = self.llm_client.chat.completions.create(
            model="gpt-4o", # Using a powerful model for robust planning
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        plan_str = response.choices[0].message.content
        return json.loads(plan_str)