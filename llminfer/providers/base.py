"""
Base class for all LLM providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union


class LLMProvider(ABC):
    """Base class for all LLM providers."""
    
    @abstractmethod
    def infer(self, conversations: Union[List[Dict], Dict, List[str], str], **kwargs) -> List[str]:
        """
        Run inference on conversations or prompts.
        
        Args:
            conversations: Can be:
                - Single conversation dict: [{"role": "user", "content": "..."}]
                - List of conversation dicts: [[{"role": "user", "content": "..."}], ...]
                - Single prompt string: "What is AI?"
                - List of prompt strings: ["What is AI?", "Explain ML", ...]
            **kwargs: Provider-specific parameters
            
        Returns:
            List of generated responses
        """
        pass 