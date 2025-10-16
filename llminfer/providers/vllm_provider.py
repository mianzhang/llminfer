"""
VLLM provider implementation for local model inference.
"""

import os
from typing import List, Union, Dict, Optional
from .base import LLMProvider


class VLLMProvider(LLMProvider):
    """VLLM provider for local model inference with proper conversation handling."""
    
    def __init__(self, model_id: str, gpu_num: Optional[int] = None, **llm_kwargs):
        """
        Initialize VLLM provider.
        
        Args:
            model_id: Path or name of the model to load
            gpu_num: Number of GPUs to use for tensor parallelism. 
                    If None, auto-detects from CUDA_VISIBLE_DEVICES
            **llm_kwargs: Additional arguments to pass to VLLM LLM constructor
        """
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
            self.LLM = LLM
            self.SamplingParams = SamplingParams
            self.AutoTokenizer = AutoTokenizer
        except ImportError as e:
            if "vllm" in str(e):
                raise ImportError("Please install the vllm package: pip install vllm")
            else:
                raise ImportError("Please install the transformers package: pip install transformers")
        
        self.model_id = model_id
        self.gpu_num = gpu_num if gpu_num is not None else self._detect_gpu_count()
        self.llm_kwargs = llm_kwargs
        self._llm = None
        self._tokenizer = None
    
    def _detect_gpu_count(self) -> int:
        """Auto-detect GPU count from CUDA_VISIBLE_DEVICES environment variable."""
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_devices:
            # Count the number of GPUs specified in CUDA_VISIBLE_DEVICES
            # Format can be "0", "0,1", "0,1,2", etc.
            return len(cuda_devices.split(','))
        return 1  # Default to 1 GPU
    
    def _get_tokenizer(self):
        """Lazy initialization of tokenizer."""
        if self._tokenizer is None:
            try:
                self._tokenizer = self.AutoTokenizer.from_pretrained(self.model_id)
            except Exception as e:
                print(f"Warning: Could not load tokenizer for {self.model_id}: {e}")
                print("Conversations will need to be pre-formatted as strings.")
                self._tokenizer = None
        return self._tokenizer
    
    def _get_llm(self):
        """Lazy initialization of LLM to avoid loading until needed."""
        if self._llm is None:
            # Default VLLM parameters
            default_params = {
                'dtype': 'bfloat16',
                'tensor_parallel_size': self.gpu_num,
                'gpu_memory_utilization': 0.8,
                'trust_remote_code': True
            }
            
            # Merge with user-provided parameters
            llm_params = {**default_params, **self.llm_kwargs, 'model': self.model_id}
            
            print(f"Initializing VLLM with model: {self.model_id}")
            print(f"GPU count: {self.gpu_num}")
            print(f"LLM parameters: {llm_params}")
            
            self._llm = self.LLM(**llm_params)
        return self._llm
    
    def _conversation_to_prompt(self, conversation: List[Dict], enable_thinking: bool = False) -> str:
        """
        Convert a conversation to a prompt string using the tokenizer's chat template.
        
        Args:
            conversation: List of message dicts with 'role' and 'content' keys
            enable_thinking: Whether to enable thinking mode for models that support it
        """
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            # Fallback: simple concatenation if tokenizer not available
            prompt_parts = []
            for msg in conversation:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            return "\n".join(prompt_parts) + "\nAssistant:"
        
        try:
            # Check if this is a model that supports thinking (like Qwen3 or Gemini 2.5)
            model_lower = self.model_id.lower()
            
            if 'qwen3' in model_lower or ('gemini' in model_lower and '2.5' in model_lower):
                # These models support thinking parameter
                try:
                    prompt = tokenizer.apply_chat_template(
                        conversation, 
                        tokenize=False, 
                        add_generation_prompt=True, 
                        enable_thinking=enable_thinking
                    )
                except TypeError:
                    # Fallback if enable_thinking parameter is not supported
                    prompt = tokenizer.apply_chat_template(
                        conversation, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
            else:
                # Standard chat template application
                prompt = tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            
            return prompt
            
        except Exception as e:
            print(f"Warning: Failed to apply chat template: {e}")
            # Fallback to simple concatenation
            prompt_parts = []
            for msg in conversation:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            return "\n".join(prompt_parts) + "\nAssistant:"
    
    def infer(self, conversations: Union[List[Dict], Dict, List[str], str], 
              enable_thinking: bool = False, **sampling_kwargs) -> List[str]:
        """
        Run VLLM inference.
        
        Args:
            conversations: Can be:
                - Single conversation dict: [{"role": "user", "content": "..."}]
                - List of conversation dicts: [[{"role": "user", "content": "..."}], ...]
                - Single prompt string: "What is AI?"
                - List of prompt strings: ["What is AI?", "Explain ML", ...]
            enable_thinking: Whether to enable thinking mode for models that support it (e.g., Qwen3, Gemini 2.5)
            **sampling_kwargs: Additional sampling parameters for SamplingParams
            
        Returns:
            List of generated responses
        """
        # Normalize input to list format
        if not isinstance(conversations, list):
            conversations = [conversations]
        
        # Convert conversations to prompts
        prompts = []
        for conv in conversations:
            if isinstance(conv, str):
                # Already a string prompt
                prompts.append(conv)
            elif isinstance(conv, list) and len(conv) > 0 and isinstance(conv[0], dict):
                # Conversation format - convert to prompt
                try:
                    prompt = self._conversation_to_prompt(conv, enable_thinking=enable_thinking)
                    prompts.append(prompt)
                except Exception as e:
                    print(f"Error converting conversation to prompt: {e}")
                    prompts.append("[ERROR: Could not convert conversation]")
            else:
                print(f"Warning: Unknown input format: {type(conv)}")
                prompts.append("[ERROR: Unknown input format]")
        
        # Run VLLM inference
        try:
            # Set default sampling parameters
            default_sampling_params = {
                'max_tokens': 2048,
                'temperature': 0.7,
            }
            
            # Merge with user-provided parameters
            sampling_params_dict = {**default_sampling_params, **sampling_kwargs}
            sampling_params = self.SamplingParams(**sampling_params_dict)
            
            llm = self._get_llm()
            
            print(f"Running VLLM inference on {len(prompts)} prompts...")
            outputs = llm.generate(prompts, sampling_params)
            generations = [output.outputs[0].text for output in outputs]
            
            return generations
            
        except Exception as e:
            print(f"Error during VLLM inference: {str(e)}")
            return ['[ERROR]'] * len(prompts)
    
    def cleanup(self):
        """Clean up the LLM instance to free memory."""
        if self._llm is not None:
            print("Cleaning up VLLM instance...")
            del self._llm
            self._llm = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None 