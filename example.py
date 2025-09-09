import llminfer

llminfer.process_jsonl(
    "prompts.jsonl",
    "output.jsonl", 
    provider="openai",
    model="gpt-4",
    input_key="prompt",  # Key pointing to string prompts
    temperature=0.7,
    max_workers=10
)