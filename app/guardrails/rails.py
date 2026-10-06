import logfire
import re
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails



from app.config import settings
from app.guardrails.colang_rules import (
    COLANG_CONTENT,
    OFF_TOPIC_EXAMPLES,
    OFF_TOPIC_REFUSAL,
    RAIL_INDICATORS,
    YAML_CONTENT,
)


_rails: LLMRails | None = None


def _normalize_text(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses the same configured Groq model as the planner and responder.
    """
    global _rails

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info(
        "NeMo Guardrails initialized",
        model=settings.GROQ_MODEL,
    )


    
def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    # Exact off-topic examples are handled locally before the LLM gate. This
    # keeps these policy decisions active even if NeMo fails to initialise.
    padded_message = f" {_normalize_text(message)} "
    matched_off_topic = next(
        (
            example
            for example in OFF_TOPIC_EXAMPLES
            if f" {_normalize_text(example)} " in padded_message
        ),
        None,
    )
    if matched_off_topic:
        logfire.info(
            "NeMo Guardrails fired",
            rail="off_topic",
            matched_example=matched_off_topic,
        )
        return True, OFF_TOPIC_REFUSAL

    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        padded_message = f" {_normalize_text(message)} "
        matched_off_topic = next(
            (
                example
                for example in OFF_TOPIC_EXAMPLES
                if f" {_normalize_text(example)} " in padded_message
            ),
            None,
        )
        if matched_off_topic:
            logfire.info(
                "NeMo Guardrails fired",
                rail="off_topic",
                matched_example=matched_off_topic,
            )
            return True, OFF_TOPIC_REFUSAL

        result = _rails.generate(messages=[{"role": "user", "content": message}])

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None
