import logfire

from langchain_groq import ChatGroq
from app.agents.state import AgentState
from app.agents.token_budget import (
    MAX_RESPONSE_COMPLETION_TOKENS,
    MAX_RESPONSE_PROMPT_TOKENS,
    build_bounded_prompt,
)
from app.config import settings


# Initialize Groq LLM
llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model=settings.GROQ_MODEL,
    temperature=0.1,
    max_tokens=MAX_RESPONSE_COMPLETION_TOKENS,
)


def responder_node(state: AgentState):
    """
    Generates the final response using conversation history
    and retrieved research documents when available.
    """

    try:
        with logfire.span("Responder Node"):

            query = state["current_query"]

            # Build conversation history
            history_str = ""

            for msg in state["messages"][:-1]:
                role = (
                    "User"
                    if msg["role"] == "user"
                    else "Assistant"
                )
                history_str += (
                    f"{role}: {msg['content']}\n"
                )

            user_msg = (
                state["messages"][-1]["content"]
                if state["messages"]
                else ""
            )

            if query == "CONVERSATIONAL":

                logfire.info(
                    "Generating conversational response"
                )

                prompt_template = """
You are a friendly and helpful AI assistant.

Use the conversation history to understand the user
and keep your response short.

CONVERSATION HISTORY:
{history}

USER MESSAGE:
{user_msg}

Give a short,concise and helpful response.
"""

                prompt_parts = {
                    "history": history_str,
                    "context": "",
                    "user_msg": user_msg,
                }
            else:

                logfire.info(
                    "Generating research response"
                )

                context = "\n\n".join(
                    state.get("documents", [])
                )

                prompt_template = """
You are a Senior Technical Research Assistant.

Answer the user's question using the research
context provided below.

RESEARCH CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION:
{user_msg}

If the context does not contain enough information,
clearly state that instead of inventing information.
"""

                prompt_parts = {
                    "history": history_str,
                    "context": context,
                    "user_msg": user_msg,
                }

            prompt, prompt_token_count = build_bounded_prompt(
                prompt_template,
                prompt_parts,
                token_limit=MAX_RESPONSE_PROMPT_TOKENS,
                trim_order=("history", "context", "user_msg"),
            )
            logfire.info(
                "Prepared token-bounded answer request",
                prompt_tokens=prompt_token_count,
                max_completion_tokens=MAX_RESPONSE_COMPLETION_TOKENS,
                request_token_budget=(
                    prompt_token_count + MAX_RESPONSE_COMPLETION_TOKENS
                ),
            )

            # Generate response
            response = llm.invoke(prompt)

            content = response.content

            logfire.info(
                "Response generated successfully"
            )

            return {
                "final_answer": content,
                "status": "Response generated.",
                "messages": [
                    {
                        "role": "assistant",
                        "content": content
                    }
                ]
            }

    except Exception as e:

        logfire.exception(
            "Responder node failed",
            error=str(e)
        )

        raise