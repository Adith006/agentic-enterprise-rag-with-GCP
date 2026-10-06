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

            if not raw_results:
                logfire.warning("Qdrant returned no documents for the query")
                return {
                    "documents": [],
                    "retrieved_sources": [],
                    "status": "No Qdrant documents found.",
                    "plan": state["plan"] + ["No Qdrant documents found"]
                }

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

            # Preserve the Qdrant metadata for the UI while keeping the
            # text-only list that the responder uses to generate its answer.
            remaining_results = list(raw_results)
            retrieved_sources = []
            for content in reranked_contents:
                match_index = next(
                    (
                        index
                        for index, result in enumerate(remaining_results)
                        if result["content"] == content
                    ),
                    None,
                )

                if match_index is None:
                    retrieved_sources.append({
                        "source": "Unknown",
                        "content": content
                    })
                    continue

                retrieved_sources.append(remaining_results.pop(match_index))

        return {
            "documents": formatted_docs,
            "retrieved_sources": retrieved_sources,
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
            "retrieved_sources": [],
            "status": "Knowledge retrieval failed.",
            "plan": state["plan"] + ["Retrieval Failed"]
        }
