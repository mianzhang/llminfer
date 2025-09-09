#!/usr/bin/env python3
"""
Command-line interface for llminfer package.
"""

import argparse
import json
from typing import Dict, Any
from .core import process_jsonl, process_jsonl_batch
from .config import create_sample_config


def parse_kwargs(kwargs_str: str) -> Dict[str, Any]:
    """Parse key=value pairs from command line."""
    kwargs = {}
    if not kwargs_str:
        return kwargs
    
    for pair in kwargs_str.split(','):
        if '=' not in pair:
            continue
        key, value = pair.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        # Try to parse as JSON, fallback to string
        try:
            kwargs[key] = json.loads(value)
        except json.JSONDecodeError:
            kwargs[key] = value
    
    return kwargs


def main():
    parser = argparse.ArgumentParser(
        description="LLM inference on JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sample config file
  llminfer --create-config
  
  # Basic usage with OpenAI (string prompts)
  llminfer prompts.jsonl output.jsonl openai gpt-4 --input-key "prompt" --kwargs "temperature=0.7"
  
  # Basic usage with conversations
  llminfer conversations.jsonl output.jsonl openai gpt-4 --input-key "conversation"
  
  # Batch processing
  llminfer large.jsonl results.jsonl openai gpt-4o-mini --batch-size 20
  
  # Custom keys with mixed input types
  llminfer data.jsonl results.jsonl anthropic claude-3-5-sonnet-20241022 \\
    --input-key "messages" --response-key "answer"
  
  # Reasoning model
  llminfer reasoning.jsonl output.jsonl openai o1-preview \\
    --kwargs "reasoning_effort=high,max_completion_tokens=4000"
        """
    )
    
    parser.add_argument("--create-config", action="store_true",
                       help="Create a sample config.json.example file")
    
    parser.add_argument("input_file", nargs="?", help="Input JSONL file path")
    parser.add_argument("output_file", nargs="?", help="Output JSONL file path")
    parser.add_argument("provider", nargs="?", help="LLM provider (openai, anthropic, gemini, vllm)")
    parser.add_argument("model", nargs="?", help="Model name")
    
    parser.add_argument("--input-key", default="conversation",
                       help="Key containing input data (string prompt or conversation list) (default: conversation)")
    parser.add_argument("--response-key", default="response", 
                       help="Key to store response (default: response)")
    parser.add_argument("--batch-size", type=int,
                       help="Process in batches (useful for large files)")
    parser.add_argument("--kwargs", type=str, default="",
                       help="Provider-specific parameters as key=value,key2=value2")
    
    args = parser.parse_args()
    
    # Handle config creation
    if args.create_config:
        create_sample_config()
        return
    
    # Validate required arguments for processing
    if not all([args.input_file, args.output_file, args.provider, args.model]):
        parser.error("input_file, output_file, provider, and model are required (unless using --create-config)")
    
    # Parse additional kwargs
    provider_kwargs = parse_kwargs(args.kwargs)
    
    print(f"Processing {args.input_file} with {args.provider} {args.model}")
    if provider_kwargs:
        print(f"Additional parameters: {provider_kwargs}")
    
    try:
        if args.batch_size:
            process_jsonl_batch(
                args.input_file,
                args.output_file,
                args.provider,
                args.model,
                batch_size=args.batch_size,
                input_key=args.input_key,
                response_key=args.response_key,
                **provider_kwargs
            )
        else:
            process_jsonl(
                args.input_file,
                args.output_file,
                args.provider,
                args.model,
                input_key=args.input_key,
                response_key=args.response_key,
                **provider_kwargs
            )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 