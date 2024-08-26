"""SenseNova embeddings file."""

import os
import time
from typing import Any, List, Optional

import sensenova
from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager

class SenseNovaEmbedding(BaseEmbedding):
    """SenseNova embeddings.

    Args:
        model_name (str): Model for embedding.
            Defaults to "nova-embedding-stable".

        api_key (Optional[str]): API key to access the model. Defaults to None.
        api_base (Optional[str]): API base to access the model. Defaults to Official Base.
    """
    task_type: Optional[str] = Field(
        default="retrieval_document",
        description="The task for embedding model.",
    )
    rate_limit_per_second: Optional[int] = Field(
        default=1,
        description="Rate limit for API calls per second.",
    )
    last_request_time: Optional[int] = Field(
        default=0,
        description="Last request time.",
    )

    def __init__(
        self,
        model_name: str = "nova-embedding-stable",
        task_type: Optional[str] = "retrieval_document",
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        rate_limit_per_second: Optional[int] = 1,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        sensenova.access_key_id = access_key_id or os.getenv("SENSENOVA_ACCESS_KEY_ID")
        sensenova.secret_access_key = secret_access_key or os.getenv("SENSENOVA_SECRET_ACCESS_KEY")

        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            task_type=task_type,
            **kwargs,
        )
        self.rate_limit_per_second = rate_limit_per_second
        self.last_request_time = 0
    @classmethod
    def class_name(cls) -> str:
        return "SenseNovaEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        response = sensenova.Embedding.create(model=self.model_name, input=[query])
        return response.embeddings[0].embedding

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        response = sensenova.Embedding.create(model=self.model_name, input=[text])
        return response.embeddings[0].embedding

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        """Get text embeddings with rate limiting."""
        embeddings = []
        for text in texts:
            # 计算需要等待的时间
            elapsed_time = time.time() - self.last_request_time
            wait_time = max(0, 1.0 / self.rate_limit_per_second - elapsed_time)
            time.sleep(wait_time)

            embedding = sensenova.Embedding.create(
                model=self.model_name,
                input=[text]
            ).embeddings[0].embedding
            embeddings.append(embedding)

            self.last_request_time = time.time()

        return embeddings

    ### Async methods ###
    # need to wait async calls from SenseNova side to be implemented.
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return self._get_text_embeddings(texts)