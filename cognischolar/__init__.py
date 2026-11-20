"""CogniScholar: a cognitive agent with commitment-reconsideration and
knowledge compilation for multi-step analysis of scientific papers."""

from .agent import CogniScholarAgent, AgentConfig
from .llm import make_backend

__all__ = ["CogniScholarAgent", "AgentConfig", "make_backend"]
__version__ = "0.1.0"
