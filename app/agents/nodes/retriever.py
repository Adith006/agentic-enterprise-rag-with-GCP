import logfire

from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents


def retrieve_node(state: AgentState):
    """
    Performs vector search against Qdrant and semantic reranking
    to retrieve the most relevant documents for the research query.
    """
    try:
        query = state["current_query"]

        with logfire.span("Knowledge Retrieval"):

            logfire.info(
                "Searching Qdrant",
                query=query
            )

            raw_results = search_enterprise_knowledge(
                query,
                limit=15
            )

            logfire.info(
                "Retrieved candidates",
                count=len(raw_results)
            )

            doc_contents = [
                doc["content"]
                for doc in raw_results
            ]

            with logfire.span("Semantic Reranking"):

                reranked_contents = rerank_documents(
                    query,
                    doc_contents,
                    top_n=5
                )

                logfire.info(
                    "Reranking completed",
                    documents_kept=len(reranked_contents)
                )

            formatted_docs = [
                f"CONTENT: {doc}"
                for doc in reranked_contents
            ]

        return {
            "documents": formatted_docs,
            "status": "Technical context retrieved.",
            "plan": state["plan"] + ["Context Retrieved"]
        }

    except Exception as e:
        logfire.exception(
            "Knowledge retrieval failed",
            error=str(e)
        )

        return {
            "documents": [],
            "status": "Knowledge retrieval failed.",
            "plan": state["plan"] + ["Retrieval Failed"]
        }