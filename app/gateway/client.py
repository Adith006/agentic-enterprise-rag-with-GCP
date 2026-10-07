import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings
import os

# config for portkey
# {
#   "strategy": {
#     "mode": "fallback"
#   },
#   "cache": {
#     "mode": "simple"
#   },
#   "retry": {
#     "attempts": 2,
#     "on_status_codes": [429, 503]
#   },
#   "targets": [
#     {
#       "provider": "groq",
#       "override_params": {
#         "model": "@YOUR_GROQ_SLUG/openai/gpt-oss-20b"
#       }
#     },
#     {
#       "provider": "groq",
#       "override_params": {
#         "model": "@YOUR_GROQ_SLUG_2/openai/gpt-oss-120b"
#       }
#     }
#   ]
# }



PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID")

portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=PORTKEY_CONFIG_ID
)


def get_langchain_llm(feature: str = "groq_primary") -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.GROQ_SLUG}/openai/gpt-oss-20b",
        temperature=0,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=PORTKEY_CONFIG_ID,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production"
            }
        )
    )


def extract_cache_status(response) -> str:
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get(
                "x-portkey-cache-status", ""
            )
            if status:
                return status.upper()

    return "MISS"