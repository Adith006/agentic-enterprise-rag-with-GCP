from langchain_groq import ChatGroq
from app.agents.state import AgentState
from app.agents.token_budget import (
    MAX_PLANNER_COMPLETION_TOKENS,
    MAX_PLANNER_PROMPT_TOKENS,
    build_bounded_prompt,
)
from app.config import settings
import logfire
from app.gateway import get_langchain_llm

# initialise the groq model
llm = get_langchain_llm(feature="planner")

def planner_node(state: AgentState):
    """
    Analyze the conversation and determine whether the user
    needs a conversational response or technical research.
    For research requests, generate a focused search query.
    """
    try:
        with logfire.span("Planner Decision"):

            history = ""

            for msg in state["messages"][:-1]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history += f"{role}: {msg['content']}\n"

            user_message = (
                state["messages"][-1]["content"]
                if state["messages"]
                else ""
            )

            prompt_template = """
            You are a research agent planner.

            Conversation history:
            {history}

            Latest user message:
            {user_message}

            Determine the user's intent.

            If the user is:
            - Greeting or casual conversation
            - Asking something that can be answered from conversation history

            Return:
            CONVERSATIONAL

            If the user needs external research:
            - Technical questions
            - Kubernetes
            - Networking
            - Intel
            - Current information
            - Documentation
            - Any question requiring external knowledge

            Return ONLY a concise and optimized search query.

            Do not provide an answer.
            """

            prompt, prompt_token_count = build_bounded_prompt(
                prompt_template,
                {
                    "history": history,
                    "user_message": user_message,
                },
                token_limit=MAX_PLANNER_PROMPT_TOKENS,
                trim_order=("history", "user_message"),
            )
            logfire.info(
                "Prepared token-bounded planner request",
                prompt_tokens=prompt_token_count,
                max_completion_tokens=MAX_PLANNER_COMPLETION_TOKENS,
            )

            decision = llm.invoke(prompt).content.strip()

            logfire.info(
                "Planner decision",
                decision=decision
            )

            if decision == "CONVERSATIONAL":
                return {
                    "current_query": "CONVERSATIONAL",
                    "status": "Conversational response required",
                    "plan": [
                        "Intent: Conversational",
                        "Retrieval: Skipped"
                    ]
                }

            return {
                "current_query": decision,
                "status": "Research required",
                "plan": [
                    "Intent: Research",
                    f"Search Query: {decision}"
                ]
            }

    except Exception as e:
        logfire.exception(
            "Planner node failed",
            error=str(e)
        )

        return {
            "current_query": "",
            "status": "Planner error",
            "plan": []
        }