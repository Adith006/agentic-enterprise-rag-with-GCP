from typing import TypedDict, List, Annotated
import operator

from typing import TypedDict, List


class AgentState(TypedDict):
    """
    Represents the state of the agent during execution.
    """

    messages: Annotated[List[dict],operator.add]
    current_query:str
    documents: List[str]
    plan: List[str]
    status: str
    final_answer: str