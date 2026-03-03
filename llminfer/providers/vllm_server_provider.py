"""
VLLM provider implementation that talks to a served (HTTP) vLLM model.

This provider assumes you are running the vLLM OpenAI-compatible server, e.g.:

    python -m vllm.entrypoints.openai.api_server --model /path/to/model

By default it connects to ``http://localhost:8000/v1`` and does not require
authentication, but both the base URL and API key can be configured.

Configuration priority:
1. Explicit constructor arguments
2. Environment variables:
   - ``VLLM_BASE_URL``  (e.g. http://localhost:8000/v1)
   - ``VLLM_API_KEY``   (if your server enforces auth)
"""

import os
import concurrent.futures
from typing import List, Dict, Union, Optional, Any

from tqdm import tqdm
from .base import LLMProvider


class VLLMServerProvider(LLMProvider):
    """Provider that calls a vLLM OpenAI-compatible HTTP server."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install the openai package: pip install openai")

        # Resolve connection settings
        resolved_base_url = (
            base_url
            or os.getenv("VLLM_BASE_URL")
            or "http://localhost:8000/v1"
        )
        resolved_api_key = api_key or os.getenv("VLLM_API_KEY") or "EMPTY"

        self._client_cls = OpenAI
        self.client = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)

    def _normalize_conversations(
        self, conversations: Union[List[Dict], Dict, List[str], str]
    ) -> List[List[Dict[str, Any]]]:
        """
        Normalize various input formats into a list of OpenAI-style message lists.

        Supported inputs:
        - Single conversation list: [{"role": "user", "content": "..."}]
        - List of conversations: [[{...}], [{...}], ...]
        - Single prompt string: "What is AI?"
        - List of prompt strings: ["What is AI?", "Explain ML", ...]
        """
        if not isinstance(conversations, list):
            conversations = [conversations]

        normalized: List[List[Dict[str, Any]]] = []
        for conv in conversations:
            if isinstance(conv, str):
                # Single prompt string
                normalized.append([{"role": "user", "content": conv}])
            elif isinstance(conv, list) and conv and isinstance(conv[0], dict):
                # Already a conversation list
                normalized.append(conv)  # type: ignore[arg-type]
            else:
                raise ValueError(
                    f"Unsupported input format for VLLMServerProvider: {type(conv)}"
                )

        return normalized

    def _single_inference(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: str,
        return_json: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        **client_kwargs: Any,
    ) -> str:
        """Run inference on a single normalized conversation."""
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
            }

            if return_json:
                kwargs["response_format"] = {"type": "json_object"}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            # Handle extra_body / thinking configuration for compatible models (e.g., Qwen3.5)
            extra_body = client_kwargs.pop("extra_body", {}) or {}
            if enable_thinking is not None:
                chat_template_kwargs = extra_body.get("chat_template_kwargs", {}) or {}
                chat_template_kwargs["enable_thinking"] = enable_thinking
                extra_body["chat_template_kwargs"] = chat_template_kwargs

            if extra_body:
                kwargs["extra_body"] = extra_body

            # Forward any remaining custom parameters directly to the client
            kwargs.update({k: v for k, v in client_kwargs.items() if v is not None})

            # Remove None values just in case
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            completion = self.client.chat.completions.create(**kwargs)
            return completion.choices[0].message.content

        except Exception as e:
            print(f"Error during vLLM server inference: {str(e)}")
            return "[ERROR]"

    def infer(
        self,
        conversations: Union[List[Dict], Dict, List[str], str],
        model: str,
        return_json: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_workers: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        **client_kwargs: Any,
    ) -> List[str]:
        """
        Run inference against a vLLM OpenAI-compatible server.
        
        Args:
            conversations: Conversation(s) or prompt string(s).
            model: Model name as exposed by the vLLM server.
            return_json: Whether to request JSON-formatted responses.
            temperature: Sampling temperature.
            max_tokens: Maximum number of tokens to generate.
            max_workers: Number of parallel requests (default: min(32, len(conversations)+4)).
            enable_thinking: When not None, passed as
                ``extra_body['chat_template_kwargs']['enable_thinking']`` for
                Qwen3.5-series models only. Using this with other models will
                raise a ValueError.
            **client_kwargs: Additional parameters forwarded to the underlying
                OpenAI-compatible client (e.g., top_p, presence_penalty, extra_body, ...).
        
        Returns:
            List of generated responses (one per input item).
        """
        # Validate enable_thinking usage: only allow for Qwen3.5 series models
        if enable_thinking is not None:
            model_lower = model.lower()
            if "qwen3.5" not in model_lower:
                raise ValueError(
                    "enable_thinking is only supported for Qwen3.5-series models "
                    "(model name containing 'Qwen3.5'). "
                    f"Received model: {model!r}"
                )

        normalized = self._normalize_conversations(conversations)

        if max_workers is None:
            max_workers = min(32, len(normalized) + 4)

        results: List[str] = [""] * len(normalized)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {}
            for i, msgs in enumerate(normalized):
                future = executor.submit(
                    self._single_inference,
                    msgs,
                    model=model,
                    return_json=return_json,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                    **client_kwargs,
                )
                future_to_index[future] = i

            with tqdm(
                total=len(normalized),
                desc=f"vLLM server inference ({model})",
            ) as pbar:
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    results[index] = future.result()
                    pbar.update(1)

        return results

