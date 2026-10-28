
from typing import Any
from pydantic import ConfigDict
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from memora.hybrid_retriever import HybridMemoryRetriever


class HybridLangChainRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    memory_retriever: Any
    user_id: str
    result_limit: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:

        results = self.memory_retriever.search(
            user_id=self.user_id,
            query=query,
            limit=self.result_limit,
        )

        documents = []

        for memory in results:
            documents.append(
                Document(
                    page_content=memory["value"],
                    metadata={
                        "memory_id": memory["id"],
                        "user_id": memory["user_id"],
                        "slot": memory["slot"],
                        "category": memory["category"],
                        "scope": memory["scope"],
                        "retrieval_score": memory["retrieval_score"],
                    },
                )
            )

        return documents
