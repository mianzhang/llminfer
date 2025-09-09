"""
Simple configuration loader for API keys.
"""

import json
import os
from typing import Optional


def load_api_key(provider: str) -> Optional[str]:
    """
    Load API key for a provider.
    
    Priority:
    1. config.json file
    2. Environment variables
    
    Args:
        provider: 'openai', 'anthropic', or 'gemini'
        
    Returns:
        API key or None
    """
    # Try config.json first
    if os.path.exists('config.json'):
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
                return config.get(provider)
        except:
            pass
    
    # Fallback to environment variables
    env_vars = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY', 
        'gemini': 'GOOGLE_API_KEY'
    }
    
    return os.getenv(env_vars.get(provider, ''))


def create_sample_config():
    """Create a sample config.json file."""
    sample = {
        "openai": "your-openai-api-key-here",
        "anthropic": "your-anthropic-api-key-here",
        "gemini": "your-google-api-key-here"
    }
    
    with open('config.json.example', 'w') as f:
        json.dump(sample, f, indent=2)
    
    print("Created config.json.example - copy to config.json and add your keys") 