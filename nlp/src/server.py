import os
import json
import logging
from typing import Any, List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ReAct System Prompt with Strict JSON constraints
SYSTEM_PROMPT = """You are an autonomous problem-solving agent for the TIL-AI competition.
You must solve the user's task by running in a strict loop of Thought, Action, PAUSE, and Observation.

1. Use "Thought" to describe your reasoning about the current state.
2. Use "Action" to execute ONE of the available tools - then immediately return "PAUSE".
3. Wait for the "Observation" (provided by the system environment) before continuing.

AVAILABLE TOOLS:
- `fetch_context`: fetches more information based on a query.
- `calculate`: performs math calculations.

CRITICAL RULE: Before returning your final answer, you must output strictly valid JSON. Do not include markdown formatting like ```json.
Format your final answer exactly as: {"answer": "Your final result here"}
"""

app = FastAPI()

# Make sure to set OPENAI_API_KEY in your environment before running this Docker container
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your_openai_api_key"))

class NLPRequest(BaseModel):
    instances: List[Dict[str, Any]]

class NLPResponse(BaseModel):
    predictions: List[str]

def run_react_agent(query: str) -> str:
    """Executes the ReAct loop using OpenAI."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
    
    max_loops = 5
    for _ in range(max_loops):
        response = client.chat.completions.create(
            model="gpt-4o",  # or gpt-4-turbo depending on preference
            messages=messages,
            temperature=0.0
        )
        
        reply = response.choices[0].message.content
        logger.info(f"Agent Reply: {reply}")
        messages.append({"role": "assistant", "content": reply})
        
        # Check if the agent paused waiting for an observation
        if "PAUSE" in reply and "Action:" in reply:
            # Here you would normally parse the Action and run a tool.
            # For hackathon purposes, we mock an observation.
            action_str = reply.split("Action:")[1].split("PAUSE")[0].strip()
            observation = f"Observation for {action_str}: Execution successful."
            logger.info(f"Observation: {observation}")
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            # Assume it's the final output, enforce JSON strictness
            try:
                # Basic cleanup in case it added markdown block
                cleaned = reply.replace("```json", "").replace("```", "").strip()
                result = json.loads(cleaned)
                return result.get("answer", cleaned)
            except json.JSONDecodeError:
                # Self-correction: Ask the model to format it correctly
                messages.append({"role": "user", "content": "Format Error: Output strictly valid JSON without markdown blocks."})
                
    return "Error: Agent exceeded max loops or failed JSON validation."

@app.post("/nlp")
def nlp_endpoint(request: NLPRequest):
    predictions = []
    for instance in request.instances:
        # Assuming the query is passed inside 'query' key in instance
        query = instance.get("query", "")
        if not query:
            predictions.append("INSUFFICIENT_DATA")
            continue
            
        final_answer = run_react_agent(query)
        predictions.append(final_answer)
        
    return {"predictions": predictions}

@app.get("/health")
def health():
    return {"status": "ok"}
