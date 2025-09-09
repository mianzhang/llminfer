"""
Google Gemini API provider implementation.
"""

import os
import concurrent.futures
from typing import List, Dict, Union, Optional
from .base import LLMProvider
from ..config import load_api_key


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""
    
    def __init__(self):
        try:
            import google.generativeai as genai
            self.genai = genai
            
            # Load API key from config.json or environment
            api_key = load_api_key('gemini')
            if not api_key:
                raise ValueError("Google API key not found. Please set GOOGLE_API_KEY environment variable or add 'gemini' key to config.json")
            
            genai.configure(api_key=api_key)
        except ImportError:
            raise ImportError("Please install the google-generativeai package: pip install google-generativeai")
    
    def _single_inference(self, conv: Dict, model: str, max_tokens: int = 8192, 
                         thinking_budget: Optional[int] = None, include_thoughts: bool = False) -> str:
        """Run inference on a single conversation."""
        try:
            # Configure generation parameters
            generation_config = self.genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
            )
            
            # Configure thinking for Gemini 2.5 series models
            thinking_config = None
            if thinking_budget is not None and thinking_budget != 0 and "2.5" in model:
                try:
                    # Check if ThinkingConfig is available
                    if hasattr(self.genai.types, 'ThinkingConfig'):
                        thinking_config = self.genai.types.ThinkingConfig(
                            include_thoughts=include_thoughts,
                            thinking_budget=thinking_budget
                        )
                    else:
                        print(f"Warning: ThinkingConfig not available in this version of google-generativeai. Falling back to regular inference.")
                        thinking_config = None
                except AttributeError:
                    print(f"Warning: ThinkingConfig not available in this version of google-generativeai. Falling back to regular inference.")
                    thinking_config = None
            
            # Convert conversation format to Gemini format
            gemini_messages = []
            system_instruction = None
            
            for msg in conv:
                if msg["role"] == "system":
                    system_instruction = msg["content"]
                elif msg["role"] == "user":
                    gemini_messages.append({
                        "role": "user",
                        "parts": [msg["content"]]
                    })
                elif msg["role"] == "assistant":
                    gemini_messages.append({
                        "role": "model",
                        "parts": [msg["content"]]
                    })
            
            # Create the model
            gemini_model = self.genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
                system_instruction=system_instruction
            )
            
            # If there are multiple messages, use chat; otherwise use generate_content
            if len(gemini_messages) > 1:
                # Multi-turn conversation
                chat = gemini_model.start_chat(history=gemini_messages[:-1])
                if thinking_config is not None:
                    try:
                        response = chat.send_message(
                            gemini_messages[-1]["parts"][0],
                            generation_config=generation_config,
                            thinking_config=thinking_config
                        )
                    except TypeError as e:
                        # If thinking_config parameter is not supported, fall back to regular inference
                        print(f"Warning: thinking_config parameter not supported. Falling back to regular inference. Error: {e}")
                        response = chat.send_message(gemini_messages[-1]["parts"][0])
                else:
                    response = chat.send_message(gemini_messages[-1]["parts"][0])
            else:
                # Single message
                if thinking_config is not None:
                    try:
                        response = gemini_model.generate_content(
                            gemini_messages[0]["parts"][0],
                            generation_config=generation_config,
                            thinking_config=thinking_config
                        )
                    except TypeError as e:
                        # If thinking_config parameter is not supported, fall back to regular inference
                        print(f"Warning: thinking_config parameter not supported. Falling back to regular inference. Error: {e}")
                        response = gemini_model.generate_content(gemini_messages[0]["parts"][0])
                else:
                    response = gemini_model.generate_content(gemini_messages[0]["parts"][0])
            
            # Extract the response text
            if include_thoughts and hasattr(response, 'candidates') and response.candidates:
                # If including thoughts, concatenate thoughts and answer
                full_response = ""
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'thought') and part.thought:
                                full_response += f"**Thinking:** {part.text}\n\n"
                            else:
                                full_response += part.text
                return full_response
            else:
                return response.text
            
        except Exception as e:
            print(f"Error during Gemini inference: {str(e)}")
            return '[ERROR]'
    
    def infer(self, conversations: Union[List[Dict], Dict], model: str,
              max_tokens: int = 8192, thinking_budget: Optional[int] = None,
              include_thoughts: bool = False, max_workers: Optional[int] = None) -> List[str]:
        """
        Run Gemini inference with parallel processing.
        
        Args:
            conversations: Single conversation dict or list of conversation dicts
            model: Model name (e.g., 'gemini-pro', 'gemini-1.5-pro', 'gemini-2.5-flash')
            max_tokens: Maximum tokens to generate
            thinking_budget: Budget for thinking (Gemini 2.5+ only)
            include_thoughts: Whether to include thinking in output
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
                    conv, model, max_tokens, thinking_budget, include_thoughts
                )
                future_to_index[future] = i
            
            # Collect results in the original order
            results = [''] * len(conversations)
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()
        
        return results 