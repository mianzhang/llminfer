"""
Core functions for LLM inference and JSONL processing.
"""

from typing import List, Dict, Any, Union, Optional
from .utils import read_jsonl, write_jsonl
from .providers import (
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    VLLMProvider,
    VLLMServerProvider,
)


def infer(conversations: Union[List[Dict], Dict], provider: str, model: str, **kwargs) -> List[str]:
    """
    Run inference on conversations using the specified provider.
    
    Args:
        conversations: Single conversation dict or list of conversation dicts
        provider: Provider name ('openai', 'anthropic', 'gemini', 'vllm', 'vllm_server')
        model: Model name
        **kwargs: Additional parameters specific to each provider
        
    Returns:
        List of generated responses
        
    Example:
        # OpenAI
        responses = infer(
            [{"role": "user", "content": "Hello!"}],
            provider="openai",
            model="gpt-4",
            temperature=0.7
        )
        
        # Anthropic
        responses = infer(
            [{"role": "user", "content": "Hello!"}],
            provider="anthropic", 
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000
        )
    """
    provider = provider.lower()
    
    if provider == "openai":
        llm = OpenAIProvider()
        return llm.infer(conversations, model=model, **kwargs)
    
    elif provider == "anthropic":
        llm = AnthropicProvider()
        return llm.infer(conversations, model=model, **kwargs)
    
    elif provider == "gemini":
        llm = GeminiProvider()
        return llm.infer(conversations, model=model, **kwargs)
    
    elif provider == "vllm":
        # VLLM requires model_id and gpu_num in constructor
        model_id = kwargs.pop('model_id', model)
        gpu_num = kwargs.pop('gpu_num', None)  # None for auto-detection
        
        # Extract VLLM-specific LLM constructor arguments
        llm_kwargs = {}
        vllm_llm_params = ['dtype', 'tensor_parallel_size', 'gpu_memory_utilization', 
                          'trust_remote_code', 'max_model_len', 'block_size', 
                          'swap_space', 'cpu_offload_gb', 'load_format']
        for param in vllm_llm_params:
            if param in kwargs:
                llm_kwargs[param] = kwargs.pop(param)
        
        llm = VLLMProvider(model_id=model_id, gpu_num=gpu_num, **llm_kwargs)
        # Local VLLM now handles both conversations and string prompts
        return llm.infer(conversations, **kwargs)

    elif provider in ("vllm_server", "vllm-server"):
        # vLLM served model over HTTP (OpenAI-compatible server)
        base_url = kwargs.pop("base_url", None)
        api_key = kwargs.pop("api_key", None)

        llm = VLLMServerProvider(base_url=base_url, api_key=api_key)
        return llm.infer(conversations, model=model, **kwargs)
    
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported: openai, anthropic, gemini, vllm, vllm_server"
        )


def process_jsonl(
    input_file: str,
    output_file: str,
    provider: str,
    model: str,
    input_key: str = "conversation",
    response_key: str = "response",
    **provider_kwargs
) -> None:
    """
    Process a JSONL file with LLM inference and save results.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        provider: Provider name ('openai', 'anthropic', 'gemini', 'vllm')
        model: Model name
        input_key: Key in JSON objects containing input data (prompt string or conversation list)
        response_key: Key to store the LLM response in output
        **provider_kwargs: Additional parameters for the provider
        
    The input JSONL should contain items with consistent input data type:
    - All string prompts: {"id": 1, "prompt": "Hello, world!"}
    - All conversation lists: {"id": 1, "conversation": [{"role": "user", "content": "Hello!"}]}
    
    For string prompts, they will be automatically converted to conversation format.
    Note: VLLM provider only accepts string prompts, not conversation lists.
    
    Example input JSONL formats:
    {"id": 1, "prompt": "What is the capital of France?"}
    {"id": 2, "prompt": "Explain quantum computing in simple terms."}
    
    OR
    
    {"id": 1, "conversation": [{"role": "user", "content": "Hello!"}]}
    {"id": 2, "conversation": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"}]}
    
    Example usage:
        # With string prompts
        process_jsonl(
            "prompts.jsonl",
            "responses.jsonl", 
            provider="openai",
            model="gpt-4",
            input_key="prompt",
            temperature=0.7
        )
        
        # With conversations
        process_jsonl(
            "conversations.jsonl",
            "responses.jsonl",
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            input_key="conversation"
        )
    """
    # Read data to determine batch size (process all at once)
    data = read_jsonl(input_file)
    if not data:
        print("No data found in input file")
        return
    
    # Call process_jsonl_batch with batch_size = total items (process all at once)
    process_jsonl_batch(
        input_file=input_file,
        output_file=output_file,
        provider=provider,
        model=model,
        batch_size=len(data),  # Process all items in a single batch
        input_key=input_key,
        response_key=response_key,
        **provider_kwargs
    )


def process_jsonl_batch(
    input_file: str,
    output_file: str,
    provider: str,
    model: str,
    batch_size: int = 10,
    input_key: str = "conversation",
    response_key: str = "response",
    **provider_kwargs
) -> None:
    """
    Process a JSONL file in batches to handle large files efficiently.
    
    This is useful for large files or when you want to save intermediate results.
    Results are appended to the output file after each batch.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        provider: Provider name ('openai', 'anthropic', 'gemini', 'vllm')
        model: Model name
        batch_size: Number of items to process in each batch
        input_key: Key in JSON objects containing input data (prompt string or conversation list)
        response_key: Key to store the LLM response in output
        **provider_kwargs: Additional parameters for the provider
        
    All items in the file must have the same input type (all strings or all conversations).
    Note: VLLM provider only supports string prompts, not conversation lists.
    """
    print(f"Reading input file: {input_file}")
    data = read_jsonl(input_file)
    
    if not data:
        print("No data found in input file")
        return
    
    print(f"Processing {len(data)} items in batches of {batch_size} with {provider} {model}")
    
    # Detect input type from first item
    first_item = data[0]
    if input_key not in first_item:
        raise KeyError(f"Key '{input_key}' not found in first item")
    
    first_input = first_item[input_key]
    is_string_input = isinstance(first_input, str)
    is_conversation_input = isinstance(first_input, list)
    
    if not (is_string_input or is_conversation_input):
        raise ValueError(f"Input data must be a string or list, got {type(first_input)}")
    
    # VLLM now supports both string prompts and conversation lists
    # No need to enforce string-only input for VLLM
    
    input_type = "string" if is_string_input else "conversation"
    print(f"Detected input type: {input_type}")
    
    # Determine if this is single-batch processing (like from process_jsonl)
    is_single_batch = batch_size >= len(data)
    
    if not is_single_batch:
        # Clear output file for multi-batch processing
        with open(output_file, 'w') as f:
            pass
    
    # Process in batches
    all_results = []  # Store all results for single-batch processing
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(data) - 1)//batch_size + 1
        
        if not is_single_batch:
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
        
        # Extract conversations for this batch
        conversations = []
        for j, item in enumerate(batch):
            try:
                input_data = item.get(input_key)
                if input_data is None:
                    raise KeyError(f"Key '{input_key}' not found in item {i + j}")
                
                # Process based on detected type
                if is_string_input:
                    if not isinstance(input_data, str):
                        raise ValueError(f"Expected string input in item {i + j}, got {type(input_data)}")
                    
                    # String prompt - convert to conversation format for consistency
                    # VLLM provider will handle the conversion back to prompt internally
                    conv = [{"role": "user", "content": input_data}]
                        
                else:  # is_conversation_input
                    if not isinstance(input_data, list):
                        raise ValueError(f"Expected list input in item {i + j}, got {type(input_data)}")
                    
                    # Conversation list - use directly
                    conv = input_data
                        
                conversations.append(conv)
            except Exception as e:
                print(f"Error processing item {i + j}: {e}")
                conversations.append([{"role": "user", "content": "[ERROR: Could not process item]"}])
        
        # Run inference on batch
        if not is_single_batch:
            print("Running inference on batch...")
        else:
            print("Running inference...")
            
        try:
            responses = infer(conversations, provider=provider, model=model, **provider_kwargs)
        except Exception as e:
            if not is_single_batch:
                print(f"Error during inference for batch {batch_num}: {e}")
            else:
                print(f"Error during inference: {e}")
            responses = ['[ERROR]'] * len(conversations)
        
        # Add responses to batch items
        for item, response in zip(batch, responses):
            item[response_key] = response
        
        if is_single_batch:
            # For single batch, collect all results
            all_results.extend(batch)
        else:
            # For multi-batch, append to file after each batch
            existing_data = []
            try:
                existing_data = read_jsonl(output_file)
            except (FileNotFoundError, ValueError):
                pass
            
            write_jsonl(output_file, existing_data + batch)
            print(f"Saved batch results. Total processed: {i + len(batch)}")
    
    if is_single_batch:
        # Write all results at once for single-batch processing
        print(f"Saving results to: {output_file}")
        write_jsonl(output_file, all_results)
    
    print(f"Done! Processed {len(data)} items in total.")


def _is_error_response(value: Any, error_prefix: str = "[ERROR]") -> bool:
    """Check if a response value indicates an error (e.g. '[ERROR]' or '[ERROR]: ...')."""
    if not isinstance(value, str):
        return False
    return value.strip().startswith(error_prefix)


def fix_jsonl(
    input_file: str,
    output_file: str,
    provider: str,
    model: str,
    input_key: str = "conversation",
    response_key: str = "response",
    error_prefix: str = "[ERROR]",
    **provider_kwargs
) -> None:
    """
    Re-run inference for JSONL rows whose response field indicates an error, and write results.

    Reads the JSONL file, finds items where the response_key value starts with error_prefix
    (default ``"[ERROR]"``), re-runs inference on those items only, and writes the updated
    data to output_file. Items without errors are left unchanged.

    Args:
        input_file: Path to input JSONL file (may contain [ERROR] responses)
        output_file: Path to output JSONL file (will be overwritten)
        provider: Provider name ('openai', 'anthropic', 'gemini', 'vllm', 'vllm_server')
        model: Model name
        input_key: Key containing input data (prompt string or conversation list)
        response_key: Key storing the response (default: "response")
        error_prefix: String prefix that marks a response as failed (default: "[ERROR]")
        **provider_kwargs: Additional parameters for the provider

    Example:
        # Fix failed items in a previously processed file
        llminfer.fix_jsonl(
            "output.jsonl",
            "output_fixed.jsonl",
            provider="openai",
            model="gpt-4",
            input_key="prompt",
            temperature=0.7,
        )
    """
    data = read_jsonl(input_file)
    if not data:
        print("No data found in input file")
        return

    # Find indices and items that need re-inference
    error_indices = [
        i for i, item in enumerate(data)
        if _is_error_response(item.get(response_key), error_prefix)
    ]

    if not error_indices:
        print("No error responses found; nothing to fix.")
        write_jsonl(output_file, data)
        return

    print(f"Found {len(error_indices)} items with error responses (out of {len(data)}). Re-running inference...")

    # Detect input type from first item
    first_item = data[0]
    if input_key not in first_item:
        raise KeyError(f"Key '{input_key}' not found in first item")

    first_input = first_item[input_key]
    is_string_input = isinstance(first_input, str)
    is_conversation_input = isinstance(first_input, list)

    if not (is_string_input or is_conversation_input):
        raise ValueError(f"Input data must be a string or list, got {type(first_input)}")

    # Build conversations only for error items
    conversations = []
    for i in error_indices:
        item = data[i]
        input_data = item.get(input_key)
        if input_data is None:
            raise KeyError(f"Key '{input_key}' not found in item {i}")

        if is_string_input:
            conv = [{"role": "user", "content": input_data}]
        else:
            conv = input_data
        conversations.append(conv)

    try:
        responses = infer(conversations, provider=provider, model=model, **provider_kwargs)
    except Exception as e:
        print(f"Error during inference: {e}")
        raise

    for data_idx, response in zip(error_indices, responses):
        data[data_idx][response_key] = response

    print(f"Saving fixed results to: {output_file}")
    write_jsonl(output_file, data)
    print(f"Done! Fixed {len(error_indices)} items.")