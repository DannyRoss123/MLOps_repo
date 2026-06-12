"""
Week 5: FastAPI endpoints for the TechCorp AI Agent.
"""

import os
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent import Agent

app = FastAPI(
    title="TechCorp AI Agent",
    description="AI agent that answers TechCorp business questions using LLM + tools.",
    version="1.0.0",
)

_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "techcorp.db",
)

_agent: Agent = None


def _get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent(_DB_PATH)
    return _agent


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    user_role: str = "engineer"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/agent/query")
def query_agent(request: QueryRequest):
    """Submit a question to the AI agent and receive an answer."""
    try:
        return _get_agent().query(request.question, request.user_role)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/metrics")
def get_metrics():
    """Return cumulative token usage and cost metrics."""
    try:
        return _get_agent().get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": _get_agent().model,
        "tools": list(_get_agent().tools.keys()),
    }
