from langchain_groq import ChatGroq
from app.agents.state import AgentState
from app.config import settings
import logfire

import logfire
from app.agents.state import AgentState
from app.gateway import portkey_client, extract_cache_status


def responder_node(state: AgentState):
    """
    Generates the final response using conversation history and,
    when available, retrieved research documents.
    """

    try:
        with logfire.span("Responder Node"):

            query = state["current_query"]

            # Build conversation history
            history_str = ""

            for msg in state["messages"][:-1]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_str += f"{role}: {msg['content']}\n"

            user_msg = (
                state["messages"][-1]["content"]
                if state["messages"]
                else ""
            )

            # Conversational response
            if query == "CONVERSATIONAL":

                logfire.info(
                    "Generating conversational response"
                )

                prompt = f"""
                You are a friendly and helpful AI assistant.

                Use the conversation history to understand the user
                and respond naturally.

                CONVERSATION HISTORY:
                {history_str}

                USER MESSAGE:
                {user_msg}

                Give a concise and helpful response.
                """

            # Research-based response
            else:

                logfire.info(
                    "Generating research-based response"
                )

                context = "\n\n".join(state.get("documents", []))

                prompt = f"""
                You are a Senior Technical Research Assistant.

                Answer the user's question using the research context
                provided below.

                RESEARCH CONTEXT:
                {context}

                CONVERSATION HISTORY:
                {history_str}

                USER QUESTION:
                {user_msg}

                If the context does not contain enough information,
                clearly state that instead of inventing information.
                """

            # Generate response
            response = portkey_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1
            )

            content = response.choices[0].message.content

            # Check Portkey cache
            cache_status = extract_cache_status(response)

            if cache_status == "HIT":
                logfire.info(
                    "Portkey cache hit"
                )
                status = "Response generated from cache."
            else:
                logfire.info(
                    "Response generated using LLM"
                )
                status = "Response generated."

            return {
                "final_answer": content,
                "status": status,
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