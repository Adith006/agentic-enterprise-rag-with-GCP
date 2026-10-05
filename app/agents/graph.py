import os
import logfire

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import responder_node


def route_planner(state: AgentState) -> str:
    """
    Routes the agent based on the planner decision.

    Conversational queries go directly to the responder.
    Research queries go through retrieval first.
    """
    if state["current_query"] == "CONVERSATIONAL":
        return "responder"

    return "retriever"


def build_checkpointer():
    """
    Creates the LangGraph checkpointer.

    Uses MemorySaver locally and PostgresSaver
    for persistent cloud memory.
    """
    local_mode = os.getenv("LOCAL_MODE", "true").lower() == "true"

    if local_mode:
        logfire.info("Using MemorySaver")
        return MemorySaver()

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from app.services.gcp.database_service import get_db_pool

        pool = get_db_pool()

        if pool is None:
            logfire.warning(
                "Postgres unavailable, using MemorySaver"
            )
            return MemorySaver()

        checkpointer = PostgresSaver(pool)
        checkpointer.setup()

        logfire.info("Using PostgresSaver")

        return checkpointer

    except Exception as e:
        logfire.exception(
            "Failed to initialize PostgresSaver",
            error=str(e)
        )

        return MemorySaver()


def build_graph():
    """
    Builds and compiles the research agent graph.

    Flow:
        Planner
           |
           +-- Conversational --> Responder --> END
           |
           +-- Research --> Retriever --> Responder --> END
    """

    with logfire.span("Building Agent Graph"):

        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("planner", planner_node)
        workflow.add_node("retriever", retrieve_node)
        workflow.add_node("responder", responder_node)

        # Entry point
        workflow.set_entry_point("planner")

        # Planner routing
        workflow.add_conditional_edges(
            "planner",
            route_planner,
            {
                "retriever": "retriever",
                "responder": "responder"
            }
        )

        # Remaining edges
        workflow.add_edge("retriever", "responder")
        workflow.add_edge("responder", END)

        # Memory
        checkpointer = build_checkpointer()

        return workflow.compile(
            checkpointer=checkpointer
        )


rag_agent = build_graph()