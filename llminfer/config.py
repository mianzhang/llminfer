"""
Simple configuration loader for API keys (environment variables only).
"""

import os
from typing import Optional


def load_api_key(provider: str) -> Optional[str]:
    """
    Load API key for a provider from environment variables only.
    
    Args:
        provider: 'openai', 'anthropic', or 'gemini'
        
    Returns:
        API key or None
    """
    env_vars = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY', 
        'gemini': 'GOOGLE_API_KEY'
    }
    
    return os.getenv(env_vars.get(provider, ''))


def create_sample_config():
    """Create a sample .env.example file for API keys."""
    sample = (
        "OPENAI_API_KEY=your-openai-api-key-here\n"
        "ANTHROPIC_API_KEY=your-anthropic-api-key-here\n"
        "GOOGLE_API_KEY=your-google-api-key-here\n"
    )

    with open('.env.example', 'w') as f:
        f.write(sample)

    print("Created .env.example - export these environment variables before running.")