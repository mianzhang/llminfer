import llminfer


import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

llminfer.process_jsonl(
    "prompts.jsonl",
    "output.jsonl", 
    provider="openai",
    model="gpt-5",
    input_key="prompt",  # Key pointing to string prompts
    service_tier="flex",
)