"""Project Sherlock — domain-agnostic context-curation library.

M1 exposes the minimum public surface required by SPEC.md § 8.1:

    from sherlock import Sherlock, Config
    config = Config.from_yaml("sherlock.yaml")
    agent = Sherlock(config)
    response = agent.chat("user input")
"""

from sherlock.agent import Sherlock, TurnState
from sherlock.config import Config

__version__ = "0.1.0"
__all__ = ["Config", "Sherlock", "TurnState", "__version__"]
