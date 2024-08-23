import os
import typing
from typing import Any, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM

import sensenova

DEFAULT_SENSENOVA_MAX_TOKENS = 512

MODEL_CONTEXT_WINDOWS = {
    "SenseChat-5": 131072,  
    "SenseChat": 4096,  
    "SenseChat-32K": 32768,
    "SenseChat-128K": 131072,
    "SenseChat-Turbo": 32768,  
    "SenseChat-5-Cantonese": 32768,  
}

class SenseNova(CustomLLM):
    """
    SenseNova LLM.

    Examples:
        `pip install sensenova`

        ```python
        from my_custom_llm import SenseNova

        llm = SenseNova(model="SenseChat-32K", access_key_id="YOUR_ACCESS_KEY_ID", secret_access_key="YOUR_SECRET_ACCESS_KEY")
        resp = llm.complete("Write a poem about a magic backpack")
        print(resp)
        ```
    """

    model: str = Field(default="SenseChat-32K", description="The SenseNova model to use.")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use during generation.",
        gte=0.0,
        lte=1.0,
    )
    max_tokens: int = Field(
        default=DEFAULT_SENSENOVA_MAX_TOKENS,
        description="The number of tokens to generate.",
        gt=0,
    )
    generate_kwargs: dict = Field(
        default_factory=dict, description="Kwargs for generation."
    )

    def __init__(
        self,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        model: Optional[str] = "SenseChat-32K",
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        **generate_kwargs: Any,
    ):
        """Creates a new SenseNova model interface."""
        try:
            import sensenova
        except ImportError:
            raise ValueError(
                "sensenova is not installed. Please install it with "
                "`pip install sensenova`."
            )

        if model not in MODEL_CONTEXT_WINDOWS:
            raise ValueError(f"Model {model} is not supported. Supported models are: {', '.join(MODEL_CONTEXT_WINDOWS.keys())}")

        sensenova.access_key_id = access_key_id or os.getenv("SENSENOVA_ACCESS_KEY_ID")
        sensenova.secret_access_key = secret_access_key or os.getenv("SENSENOVA_SECRET_ACCESS_KEY")

        if not max_tokens:
            max_tokens = DEFAULT_SENSENOVA_MAX_TOKENS
        else:
            max_tokens = min(max_tokens, DEFAULT_SENSENOVA_MAX_TOKENS)


        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            generate_kwargs=generate_kwargs,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        return "SenseNova_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        # Assuming SenseNova has a fixed context window size
        context_window = MODEL_CONTEXT_WINDOWS.get(self.model, 4096)  # 默认值为4096  # You may need to adjust this based on SenseNova's specifications
        return LLMMetadata(
            context_window=context_window,
            num_output=self.max_tokens,
            model_name=self.model,
            is_chat_model=True,
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        response = sensenova.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **self.generate_kwargs,
            **kwargs
        )
        text = response.data.choices[0].message
        return CompletionResponse(text=text, raw=response)
    
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        response = sensenova.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **self.generate_kwargs,
            **kwargs
        )
        for chunk in response:
            text = chunk.data.choices[0].message
            yield CompletionResponse(text=text, raw=chunk)
    
    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        sensenova_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        response = sensenova.ChatCompletion.create(
            model=self.model,
            messages=sensenova_messages,
            **self.generate_kwargs,
            **kwargs
        )
        text = response.data.choices[0].message
        return ChatResponse(message=ChatMessage(role="assistant", content=text), raw=response)
