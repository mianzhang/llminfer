import llminfer


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

llminfer.process_jsonl(
    "prompts.jsonl",
    "output.jsonl", 
    provider="vllm",
    model="/localdisk/models/Qwen2.5-1.5B-Instruct",
    input_key="prompt",  # Key pointing to string prompts
    temperature=0.7,
)