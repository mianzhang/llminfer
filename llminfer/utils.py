"""
Utility functions for file handling and data processing.
"""

import json
from typing import List, Dict, Any


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Read a JSONL file and return a list of dictionaries.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries, one for each line in the file
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line.strip()) for line in file if line.strip()]


def write_jsonl(file_path: str, data: List[Dict[str, Any]]) -> None:
    """
    Write a list of dictionaries to a JSONL file.
    
    Args:
        file_path: Path to the output JSONL file
        data: List of dictionaries to write
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n') 