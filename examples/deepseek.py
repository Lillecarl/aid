"""A pydantic-ai agent on DeepSeek. The provider reads DEEPSEEK_API_KEY from the worker's environment.

    aid new-py deepseek --python-path examples deepseek:agent
    aid prompt deepseek "What is aid?"
"""

from __future__ import annotations

from pydantic_ai import Agent

agent = Agent("deepseek:deepseek-chat", instructions="Answer briefly and plainly.")
