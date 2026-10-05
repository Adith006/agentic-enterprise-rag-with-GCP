# ============================================================
# CRITICAL: Configure Logfire before importing app modules
# ============================================================

import os
import logfire

from dotenv import load_dotenv

load_dotenv()

logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN")
)

# ============================================================
# App imports
# ============================================================

from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import Optional

from app.agents.graph import rag_agent


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Enterprise Agentic RAG API"
)


# ============================================================
# Request Model
# ============================================================

class QueryRequest(BaseModel):
    """
    Request model for user queries.
    """

    q: str
    thread_id: Optional[str] = "default_user"


# ============================================================
# Health Check
# ============================================================

@app.get("/")
def home():
    """
    Health check endpoint.
    """
    return {
        "message": "Enterprise Agentic RAG API is live."
    }


# ============================================================
# Graph Visualization
# ============================================================

@app.get("/graph")
def get_graph_image():
    """
    Returns the LangGraph workflow as a PNG image.
    """
    try:
        with logfire.span("Generate Graph Image"):

            png_bytes = (
                rag_agent
                .get_graph()
                .draw_mermaid_png()
            )

            return Response(
                content=png_bytes,
                media_type="image/png"
            )

    except Exception as e:

        logfire.exception(
            "Failed to generate graph image",
            error=str(e)
        )

        return {
            "error": "Could not generate graph image."
        }


# ============================================================
# Query Endpoint
# ============================================================

@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the Agentic RAG pipeline for a user query.
    """

    q = request.q
    thread_id = request.thread_id

    initial_state = {
        "messages": [
            {
                "role": "user",
                "content": q
            }
        ],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing...",
        "final_answer": ""
    }

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:

        with logfire.span(
            "Agentic RAG Query",
            thread_id=thread_id
        ):

            final_output = rag_agent.invoke(
                initial_state,
                config=config
            )

            answer = final_output.get(
                "final_answer",
                ""
            )

            return {
                "question": q,
                "answer": answer,
                "thought_process": final_output.get(
                    "plan",
                    []
                ),
                "status": final_output.get(
                    "status",
                    ""
                ),
                "sources": final_output.get(
                    "documents",
                    []
                )
            }

    except Exception as e:

        logfire.exception(
            "Agent execution failed",
            error=str(e),
            thread_id=thread_id
        )

        return {
            "question": q,
            "answer": (
                "I apologize, but I encountered "
                "an internal error. Please try again."
            ),
            "thought_process": [
                "Error encountered during execution."
            ],
            "status": "error",
            "sources": []
        }