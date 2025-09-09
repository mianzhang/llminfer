"""
llminfer - A simple package for LLM inference with JSONL file support

This package provides easy-to-use functions for running inference with various LLM providers
including OpenAI, Anthropic, Google Gemini, and VLLM.

Main functions:
- process_jsonl: Process a JSONL file with an LLM and save results (uses process_jsonl_batch internally)
- process_jsonl_batch: Process a JSONL file in batches for large files or intermediate saves
- infer: Run inference on conversations/prompts directly
"""

from .core import process_jsonl, process_jsonl_batch, infer
from .providers import OpenAIProvider, AnthropicProvider, GeminiProvider, VLLMProvider
from .utils import read_jsonl, write_jsonl
from .config import load_api_key, create_sample_config

__version__ = "0.1.0"
__all__ = [
    "process_jsonl", 
    "process_jsonl_batch",
    "infer",
    "OpenAIProvider", 
    "AnthropicProvider", 
    "GeminiProvider", 
    "VLLMProvider",
    "read_jsonl", 
    "write_jsonl",
    "load_api_key",
    "create_sample_config"
] 