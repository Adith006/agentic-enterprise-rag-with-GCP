import logfire

from app.agents.state import AgentState
from app.agents.token_budget import (
    MAX_RESPONSE_COMPLETION_TOKENS,
    MAX_RESPONSE_PROMPT_TOKENS,
    build_bounded_prompt,
)
from app.gateway import portkey_client, extract_cache_status


def responder_node(state: AgentState):
    """
    Generates the final response using conversation history
    and retrieved research documents through Portkey.
    """

    try:
        with logfire.span("Responder Node"):

            query = state["current_query"]

            history_str = ""

            for msg in state["messages"][:-1]:
                role = (
                    "User"
                    if msg["role"] == "user"
                    else "Assistant"
                )
                history_str += f"{role}: {msg['content']}\n"

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

Give a short, concise and helpful response.
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
                    prompt_token_count
                    + MAX_RESPONSE_COMPLETION_TOKENS
                ),
            )

            # Generate response through Portkey
            response = portkey_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=MAX_RESPONSE_COMPLETION_TOKENS,
            )

            content = response.choices[0].message.content

            # Check Portkey cache
            cache_status = extract_cache_status(response)

            if cache_status == "HIT":
                logfire.info(
                    "Gateway Cache Hit — response served from Portkey cache."
                )

                plan_update = state["plan"] + [
                    "Cache: Hit"
                ]

                status = "Cache hit — instant response."

            else:
                logfire.info(
                    "Response generated through Portkey."
                )

                plan_update = state["plan"]

                status = "Response generated."

            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
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