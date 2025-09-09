"""
Anthropic Claude API provider implementation.
"""

import concurrent.futures
from typing import List, Dict, Union, Optional
from .base import LLMProvider
from ..config import load_api_key


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self):
        try:
            import anthropic
            
            # Load API key from config.json or environment
            api_key = load_api_key('anthropic')
            if not api_key:
                raise ValueError("Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable or add 'anthropic' key to config.json")
            
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Please install the anthropic package: pip install anthropic")
    
    def _single_inference(self, conv: Dict, model: str, max_tokens: int = 8192, 
                         thinking_budget: Optional[int] = None, enable_thinking: bool = False) -> str:
        """Run inference on a single conversation."""
        try:
            # Check if model supports thinking (Claude 3.7+ and Claude 4+ models)
            def supports_thinking(model_name):
                model_lower = model_name.lower()
                return ('claude-3-7-sonnet' in model_lower or 
                        'claude-opus-4' in model_lower or 
                        'claude-sonnet-4' in model_lower or
                        'claude-4' in model_lower)
            
            # Extract system message if present
            system_message = None
            messages = []
            
            for msg in conv:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    # Ensure content is in the right format for Anthropic
                    if isinstance(msg["content"], str):
                        content = [{"type": "text", "text": msg["content"]}]
                    else:
                        content = msg["content"]
                    
                    messages.append({
                        "role": msg["role"],
                        "content": content
                    })
            
            # Create the API call
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages
            }
            
            if system_message:
                kwargs["system"] = system_message
            
            # Add thinking parameters for compatible models
            if supports_thinking(model) and enable_thinking:
                # Use provided budget or default to reasonable value
                budget = thinking_budget if thinking_budget is not None else min(max_tokens // 2, 10000)
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget
                }
            
            response = self.client.messages.create(**kwargs)
            
            # Extract text from response, handling thinking blocks if present
            if hasattr(response, 'content') and response.content:
                # For models with thinking, we might get thinking + text blocks
                text_parts = []
                thinking_parts = []
                
                for block in response.content:
                    if hasattr(block, 'type'):
                        if block.type == "text":
                            text_parts.append(block.text)
                        elif block.type == "thinking" and hasattr(block, 'thinking'):
                            thinking_parts.append(block.thinking)
                
                text_content = "\n".join(text_parts) if text_parts else response.content[0].text
                
                # If thinking was captured and enabled, return structured response
                if thinking_parts and enable_thinking:
                    return {
                        "text": text_content,
                        "thinking": "\n\n".join(thinking_parts),
                        "has_thinking": True
                    }
                else:
                    return text_content
            else:
                return response.content[0].text
            
        except Exception as e:
            print(f"Error during Anthropic inference: {str(e)}")
            return '[ERROR]'
    
    def infer(self, conversations: Union[List[Dict], Dict], model: str,
              max_tokens: int = 8192, thinking_budget: Optional[int] = None,
              enable_thinking: bool = False, max_workers: Optional[int] = None) -> List[str]:
        """
        Run Anthropic inference with parallel processing.
        
        Args:
            conversations: Single conversation dict or list of conversation dicts
            model: Model name (e.g., 'claude-3-5-sonnet-20241022', 'claude-3-opus-20240229')
            max_tokens: Maximum tokens to generate
            thinking_budget: Budget for thinking tokens (Claude 3.7+ only)
            enable_thinking: Whether to enable thinking mode
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
                    conv, model, max_tokens, thinking_budget, enable_thinking
                )
                future_to_index[future] = i
            
            # Collect results in the original order
            results = [''] * len(conversations)
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()
        
        return results 