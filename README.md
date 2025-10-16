# llminfer

A simple Python package for LLM inference with JSONL file support. Easily run inference with multiple LLM providers including OpenAI, Anthropic, Google Gemini, and VLLM.

## Features

- **Multiple Providers**: Support for OpenAI, Anthropic Claude, Google Gemini, and VLLM
- **JSONL Processing**: Built-in functions to process JSONL files with LLM inference
- **Parallel Processing**: Concurrent API calls for faster inference with multiple conversations
- **Batch Processing**: Handle large files efficiently with batch processing
- **Reasoning Models**: Support for OpenAI's reasoning models (o1, o3) with proper parameter handling
- **Thinking Mode**: Support for Claude's thinking mode and Gemini's thinking capabilities
- **Flexible Input**: Custom conversation extraction functions for different data formats
- **Simple API**: Easy-to-use functions without complex abstractions

## Installation

```bash
pip install openai anthropic google-generativeai vllm  # Install the providers you need
```

## Quick Start

### Basic Inference

```python
import llminfer

# Simple conversation
conversation = [{"role": "user", "content": "What is the capital of France?"}]

# OpenAI
responses = llminfer.infer(
    conversation,
    provider="openai",
    model="gpt-4",
    temperature=0.7
)

# Anthropic Claude
responses = llminfer.infer(
    conversation,
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000
)
```

### JSONL File Processing

The `process_jsonl` function supports both **string prompts** and **conversation lists** as input. All items in a file must use the same input type:

```python
# Process string prompts
llminfer.process_jsonl(
    "prompts.jsonl",
    "output.jsonl", 
    provider="openai",
    model="gpt-4",
    input_key="prompt",  # Key pointing to string prompts
    temperature=0.7
)

# Process conversation lists (backward compatible)
llminfer.process_jsonl(
    "conversations.jsonl",
    "output.jsonl", 
    provider="openai",
    model="gpt-4",
    input_key="conversation",  # Key pointing to conversation lists
    temperature=0.7
)
```

**Supported input formats:**

String prompts:
```jsonl
{"id": 1, "prompt": "What is the capital of France?"}
{"id": 2, "prompt": "Explain quantum computing in simple terms."}
```

Conversation lists (for OpenAI/Anthropic/Gemini only):
```jsonl
{"id": 1, "conversation": [{"role": "user", "content": "Hello!"}]}
{"id": 2, "messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "What is 2+2?"}]}
```

**Note**: VLLM will raise an error if you try to use conversation lists. Use string prompts instead.



**Output format:**
```jsonl
{"id": 1, "prompt": "What is the capital of France?", "response": "The capital of France is Paris."}
{"id": 2, "conversation": [{"role": "user", "content": "Hello!"}], "response": "Hello! How can I help you today?"}
```

## API Reference

### Core Functions

#### `infer(conversations, provider, model, **kwargs)`

Run inference on conversations using the specified provider.

**Parameters:**
- `conversations`: Single conversation dict or list of conversation dicts
- `provider`: Provider name ('openai', 'anthropic', 'gemini', 'vllm')
- `model`: Model name
- `**kwargs`: Provider-specific parameters

**Returns:** List of generated responses

#### `process_jsonl(input_file, output_file, provider, model, **kwargs)`

Process a JSONL file with LLM inference and save results.

**Parameters:**
- `input_file`: Path to input JSONL file
- `output_file`: Path to output JSONL file
- `provider`: Provider name
- `model`: Model name
- `input_key`: Key containing input data - string prompt or conversation list (default: "conversation")
- `response_key`: Key to store response (default: "response")


- `**kwargs`: Provider-specific parameters

#### `process_jsonl_batch(input_file, output_file, provider, model, batch_size=10, **kwargs)`

Process a JSONL file in batches for large files. This is the core implementation that `process_jsonl` uses internally.

### Providers

#### OpenAI Provider

Supports GPT models and reasoning models (o1, o3).

```python
responses = llminfer.infer(
    conversations,
    provider="openai",
    model="gpt-4",
    temperature=0.7,
    return_json=False,
    reasoning_effort="medium",  # For reasoning models
    max_completion_tokens=2000,  # For reasoning models
    max_workers=5  # Enable parallel processing (default: min(32, len(conversations) + 4))
)
```

#### Anthropic Provider

Supports Claude models with thinking mode.

```python
responses = llminfer.infer(
    conversations,
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    max_tokens=8192,
    thinking_budget=5000,  # For thinking mode
    enable_thinking=False,
    max_workers=5  # Enable parallel processing (default: min(32, len(conversations) + 4))
)
```

#### Gemini Provider

Supports Google Gemini models with thinking capabilities.

```python
responses = llminfer.infer(
    conversations,
    provider="gemini",
    model="gemini-1.5-pro",
    max_tokens=8192,
    thinking_budget=5000,  # For Gemini 2.5+
    include_thoughts=False,
    max_workers=5  # Enable parallel processing (default: min(32, len(conversations) + 4))
)
```

#### VLLM Provider

For local model inference. **Now supports both conversation lists and string prompts with automatic chat template handling.**

```python
# With conversation format (recommended)
responses = llminfer.infer(
    conversations,  # List of conversation dicts or single conversation
    provider="vllm",
    model="meta-llama/Llama-2-7b-chat-hf",
    gpu_num=None,  # Auto-detects from CUDA_VISIBLE_DEVICES
    temperature=0.7,
    max_tokens=512,
    # Additional VLLM LLM parameters
    dtype='bfloat16',
    gpu_memory_utilization=0.8,
    trust_remote_code=True
)

# With string prompts (legacy support)
responses = llminfer.infer(
    ["What is AI?", "Explain machine learning"],  # List of strings
    provider="vllm",
    model="meta-llama/Llama-2-7b-chat-hf"
)

# With thinking mode for supported models (Qwen3, Gemini 2.5)
responses = llminfer.infer(
    conversations,
    provider="vllm",
    model="Qwen/Qwen3-7B-Instruct",
    enable_thinking=True,  # Enable thinking mode
    temperature=0.7
)

# JSONL processing now works with both formats:
llminfer.process_jsonl(
    "conversations.jsonl",  # Can contain conversations or string prompts
    "vllm_responses.jsonl",
    provider="vllm",
    model="meta-llama/Llama-2-7b-chat-hf",
    input_key="conversation"  # Or "prompt" for strings
)
```

**VLLM Features:**
- **Auto-GPU Detection**: Automatically detects GPU count from `CUDA_VISIBLE_DEVICES`
- **Chat Template Support**: Uses model's tokenizer to properly format conversations
- **Thinking Models**: Special handling for Qwen3 and Gemini 2.5 models
- **Flexible Input**: Accepts both conversation lists and pre-formatted string prompts
- **Batch Processing**: Processes all inputs in a single efficient batch

## Advanced Usage

### Parallel Processing

All API-based providers (OpenAI, Anthropic, Gemini) now support parallel processing for multiple conversations, significantly improving performance when processing multiple requests:

```python
import llminfer

# Multiple conversations processed in parallel
conversations = [
    [{"role": "user", "content": "What is AI?"}],
    [{"role": "user", "content": "Explain machine learning"}],
    [{"role": "user", "content": "What is deep learning?"}],
    [{"role": "user", "content": "Define neural networks"}],
]

# Parallel processing with custom worker count
responses = llminfer.infer(
    conversations,
    provider="openai",
    model="gpt-4o-mini",
    max_workers=10  # Process up to 10 requests concurrently
)

# Default worker count is min(32, len(conversations) + 4)
responses = llminfer.infer(
    conversations,
    provider="anthropic", 
    model="claude-3-5-haiku-20241022"
    # max_workers automatically determined
)
```

**Performance Benefits:**
- **3-5x faster** for multiple conversations compared to sequential processing
- Automatic rate limit handling through concurrent futures
- Results returned in original order regardless of completion time
- Memory efficient - only active requests consume resources

**Notes:**
- VLLM provider handles batching internally - all conversations are processed in a single efficient batch
- API-based providers (OpenAI, Anthropic, Gemini) benefit most from parallel processing
- Optimal worker count depends on your API rate limits and system resources
- Too many workers may hit rate limits; start with default and adjust as needed

### Flexible Input Handling

The `process_jsonl` function automatically handles different input types:

```python
# String prompts - automatically converted to conversation format for most providers
llminfer.process_jsonl(
    "prompts.jsonl",
    "responses.jsonl",
    provider="openai",
    model="gpt-4",
    input_key="prompt"
)

# Conversation lists - used directly for chat-based providers
llminfer.process_jsonl(
    "conversations.jsonl",
    "responses.jsonl",
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    input_key="messages"
)

# Consistent input types within each file
string_data = [
    {"id": 1, "data": "What is AI?"},  # All string prompts
    {"id": 2, "data": "Explain machine learning"},
]
llminfer.write_jsonl("prompts.jsonl", string_data)

# Works with any provider
llminfer.process_jsonl(
    "prompts.jsonl",
    "prompt_responses.jsonl",
    provider="openai",
    model="gpt-4",
    input_key="data"  # Detects string type from first item
)

# VLLM works with both string prompts and conversations
llminfer.process_jsonl(
    "prompts.jsonl",
    "vllm_responses.jsonl", 
    provider="vllm",
    model="meta-llama/Llama-2-7b-chat-hf",
    input_key="data"  # Can be strings or conversations
)
```

**Provider-specific behavior:**
- **OpenAI/Anthropic/Gemini**: String prompts → converted to `[{"role": "user", "content": "prompt"}]`
- **VLLM**: Accepts both formats - conversations → chat template → prompt strings
- **Input type detection**: Automatically detects input type from first item and processes all items consistently

### Using Different Input Keys

```python
# If your data has questions in a "question" field
# {"id": 1, "question": "What is the capital of France?"}
llminfer.process_jsonl(
    "qa.jsonl",
    "qa_responses.jsonl",
    provider="openai",
    model="gpt-4",
    input_key="question",  # Automatically detects it's a string prompt
    response_key="answer"
)

# If your data has complex conversations in a "messages" field  
# {"id": 1, "messages": [{"role": "system", "content": "You are an expert."}, {"role": "user", "content": "Help me"}]}
llminfer.process_jsonl(
    "conversations.jsonl",
    "responses.jsonl",
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    input_key="messages"  # Automatically detects it's a conversation list
)
```

### Batch Processing for Large Files

```python
# For large files, use explicit batch processing to save intermediate results
llminfer.process_jsonl_batch(
    "large_dataset.jsonl",
    "results.jsonl",
    provider="openai",
    model="gpt-4o-mini",
    batch_size=50,  # Process in chunks of 50 items
    temperature=0.5
)

# Note: process_jsonl() internally uses process_jsonl_batch() with batch_size=total_items
# So both functions use the same core logic, just different batching strategies
```

### Using Reasoning Models

```python
# OpenAI o1/o3 models
responses = llminfer.infer(
    conversation,
    provider="openai",
    model="o1-preview",
    reasoning_effort="high",
    max_completion_tokens=4000
)
```

### Using Thinking Mode

```python
# Claude thinking mode
responses = llminfer.infer(
    conversation,
    provider="anthropic",
    model="claude-3-7-sonnet",
    enable_thinking=True,
    thinking_budget=10000
)

# Gemini thinking mode (Gemini 2.5+)
responses = llminfer.infer(
    conversation,
    provider="gemini",
    model="gemini-2.5-flash",
    thinking_budget=8000,
    include_thoughts=True
)

# VLLM thinking mode (for Qwen3, Gemini 2.5, etc.)
responses = llminfer.infer(
    conversation,
    provider="vllm",
    model="Qwen/Qwen3-7B-Instruct",
    enable_thinking=True,
    temperature=0.7
)

# OpenAI reasoning models (o1, o3)
responses = llminfer.infer(
    conversation,
    provider="openai",
    model="o1-preview",
    reasoning_effort="high",
    max_completion_tokens=4000
)
```

## Configuration

### Option 1: config.json (Recommended)

Create a `config.json` file in your project directory:

```json
{
  "openai": "your-openai-api-key",
  "anthropic": "your-anthropic-api-key", 
  "gemini": "your-google-api-key"
}
```

Or generate a sample config:
```bash
llminfer --create-config
# This creates config.json.example - copy to config.json and add your keys
```

### Option 2: Environment Variables

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

**Note**: config.json takes priority over environment variables.

## Examples

See `examples.py` for complete working examples:

```bash
cd llminfer
python examples.py
```

## Error Handling

The package includes robust error handling:
- API errors return '[ERROR]' as the response
- Individual item failures don't stop batch processing
- Detailed error messages are printed to help with debugging

## License

This package is provided as-is for research and development purposes. 