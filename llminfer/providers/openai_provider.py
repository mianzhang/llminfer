"""
OpenAI API provider implementation.
"""

import concurrent.futures
from typing import List, Dict, Union, Optional
from tqdm import tqdm
from .base import LLMProvider
from ..config import load_api_key


class OpenAIProvider(LLMProvider):
    """OpenAI API provider supporting GPT models and reasoning models like o1, o3."""
    
    def __init__(self):
        try:
            from openai import OpenAI
            
            # Load API key from config.json or environment
            api_key = load_api_key('openai')
            if not api_key:
                raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or add 'openai' key to config.json")
            
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("Please install the openai package: pip install openai")
    
    def _single_inference(self, conv: Dict, model: str, return_json: bool = False, 
                         temperature: Optional[float] = None, reasoning_effort: str = 'medium', 
                         max_completion_tokens: Optional[int] = None) -> str:
        """Run inference on a single conversation."""
        try:
            # List of reasoning model name substrings (case-insensitive)
            reasoning_model_keys = ["o3", "o1", "o4-mini"]
            
            def is_reasoning_model(model_name):
                model_name = model_name.lower()
                return any(key in model_name for key in reasoning_model_keys)
            
            kwargs = {
                "model": model,
                "messages": conv,
            }
            
            if return_json:
                kwargs["response_format"] = {"type": "json_object"}
            
            if is_reasoning_model(model):
                # Only add supported params for reasoning models
                if max_completion_tokens is not None:
                    kwargs["max_completion_tokens"] = max_completion_tokens
                if reasoning_effort is not None:
                    kwargs["reasoning_effort"] = reasoning_effort
                # Do NOT add temperature for reasoning models
            else:
                # For non-reasoning models, keep temperature if provided
                if temperature is not None:
                    kwargs["temperature"] = temperature
            
            # Remove None values
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            
            completion = self.client.chat.completions.create(**kwargs)
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Error during OpenAI inference: {str(e)}")
            return '[ERROR]'
    
    def infer(self, conversations: Union[List[Dict], Dict], model: str, 
              return_json: bool = False, temperature: Optional[float] = None,
              reasoning_effort: str = 'medium', max_completion_tokens: Optional[int] = None,
              max_workers: Optional[int] = None) -> List[str]:
        """
        Run OpenAI inference with parallel processing.
        
        Args:
            conversations: Single conversation dict or list of conversation dicts
            model: Model name (e.g., 'gpt-4', 'o1-preview', 'o3-mini')
            return_json: Whether to request JSON format response
            temperature: Temperature for non-reasoning models
            reasoning_effort: Effort level for reasoning models ('low', 'medium', 'high')
            max_completion_tokens: Max tokens for reasoning models
            max_workers: Maximum number of concurrent threads (default: min(32, len(conversations) + 4))
            
        Returns:
            List of generated responses
        """
        if not isinstance(conversations, list):
            conversations = [conversations]
        
        # Use ThreadPoolExecutor for parallel API calls
        if max_workers is None:
            max_workers = min(32, len(conversations) + 4)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all conversations for parallel processing
            future_to_index = {}
            for i, conv in enumerate(conversations):
                future = executor.submit(
                    self._single_inference, 
                    conv, model, return_json, temperature, 
                    reasoning_effort, max_completion_tokens
                )
                future_to_index[future] = i
            
            # Collect results in the original order with progress bar
            results = [''] * len(conversations)
            with tqdm(total=len(conversations), desc=f"OpenAI inference ({model})") as pbar:
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    results[index] = future.result()
                    pbar.update(1)
        
        return results 