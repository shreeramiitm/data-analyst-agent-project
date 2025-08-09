# main.py
# API server and main entry point for the application.

import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import orchestrator_agent

# Load environment variables from a .env file
load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Data Analysis Agent API",
    description="A multi-agent API that analyzes and visualizes data from any source.",
    version="2.0.0" 
)

# Create a single, reusable instance of our agent
try:
    data_agent = orchestrator_agent.OrchestratorAgent()
except Exception as e:
    # If the agent fails to initialize (e.g., missing API key),
    # we should know immediately.
    print(f"FATAL: Could not initialize OrchestratorAgent: {e}")
    data_agent = None

@app.get("/", tags=["Health Check"])
async def read_root():
    """A simple endpoint to check if the server is running."""
    return {"status": "ok", "message": "Data Analysis Agent is running."}

# --- Main API Endpoint (Updated for File Uploads) ---
@app.post("/api/", tags=["Core Functionality"])
async def analyze_data(questions_file: UploadFile = File(..., alias="questions.txt")):
    """
    Accepts a 'questions.txt' file via multipart/form-data.
    This endpoint processes the file to scrape, analyze, and visualize data.
    """
    if not data_agent:
        raise HTTPException(
            status_code=503,
            detail="Agent is not available due to an initialization error. Check server logs."
        )

    try:
        # Read the content of the uploaded file
        prompt = (await questions_file.read()).decode("utf-8")
        print(f"Received task from file: {prompt[:250]}...") # Log first 250 chars

        # Execute the task using the agent instance
        print("--- Handing off task to agent ---")
        result = await data_agent.run(prompt=prompt)
        print("--- Agent finished task ---")
        
        # The orchestrator is designed to return a JSON-serializable list or dict
        return JSONResponse(content=result)

    except ValueError as ve:
        # Handle specific value errors, which are often user-input related
        print(f"A value error occurred: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Handle unexpected server errors
        print(f"An unexpected error occurred while processing the task: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")